import logging
import os
import time

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RolloutScheduler(Scheduler):
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        req_id_to_prompt_token_num = {req_id: req.num_prompt_tokens for req_id, req in self.requests.items()}
        req_id_to_response_token_num = {req_id: req.num_output_tokens for req_id, req in self.requests.items()}
        # NOTE(lhy): we need to patch the original vllm SchedulerStats to add:
        # 1. `need_to_abort_reqs` field. This is a set of request IDs that need to be aborted.
        # 2. `req_id_to_prompt_token_num` field. This is a dictionary of request ID to the number of prompt tokens.
        # 3. `req_id_to_response_token_num` field. This is a dictionary of request ID to the number of response tokens.
        return SchedulerStats(
            need_to_abort_reqs=self.need_to_abort_reqs,
            req_id_to_prompt_token_num=req_id_to_prompt_token_num,
            req_id_to_response_token_num=req_id_to_response_token_num,
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, "Only running requests can be preempted"
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # NOTE(lhy): We examine the number of waiting requests to
        # determine whether to abort the preempted request.
        # Once aborted, the preempted request will be put back to
        # the rollout router to be scheduled again.
        max_num_waiting_reqs_after_preemption = self.vllm_config.additional_config.get(
            "max_num_waiting_reqs_after_preemption", 0
        )
        if len(self.waiting) > max_num_waiting_reqs_after_preemption:
            # NOTE(lhy): the `need_to_abort_reqs` is set and put
            # into the scheduler stats. Afterwards inside vllm
            # rollout, the abortion will be performed.
            print(
                f"Preempted request {request.request_id} is "
                f"aborted because of "
                f"max_num_waiting_reqs_after_preemption is "
                f"{max_num_waiting_reqs_after_preemption}"
            )
            self.need_to_abort_reqs.append(request.request_id)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    # NOTE(lhy): The most of the code is copied from the vLLM 10.0.2 scheduler.
    # We refactor the logic of preemption to allow preempted sequences to directly returned as aborted.
    # So that we can allow rollout migration of the waiting requests between different instances.
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        self.need_to_abort_reqs: list[str] = list()
        return super().schedule()

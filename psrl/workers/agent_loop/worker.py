import asyncio
import logging
import os
from collections import deque

import torch
import hydra
import numpy as np
import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.distillation import is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.workers.config.model import HFModelConfig

from psrl.utils.common.http_utils import init_http_client
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events
from psrl.utils.rollout.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from psrl.workers.gen_dplb.utils import TokenOutput
from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY, DictConfigWrap, TerminateReason
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class PSRL_AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle: ray.actor.ActorHandle,
        rollout_router: ray.actor.ActorHandle | str,
    ):
        """Initialize agent loop worker.

        Args:
            config (DictConfig): Configuration containing model and rollout settings.
            ps_manager_handle (ray.actor.ActorHandle): Handle to the parameter server manager.
            rollout_router (ray.actor.ActorHandle | str): Handle to the rollout router actor.
        """
        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        # TransferQueue bootstrap (connects to controller/storage spun up by the driver).
        tq.init()

        self.distillation_config = config.get("distillation", None)
        self.distillation_enabled = is_distillation_enabled(self.distillation_config)
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not supported in PSRL yet.")

        self.dataset_cls = get_dataset_class(config.data)

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.rollout_router = rollout_router
        self.ps_manager_handle = ps_manager_handle
        self.agent_loop_manager = None
        self.reward_manager = None

        n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        init_http_client(
            server_concurrency=self.config.psrl.rollout_gateway.max_concurrency,
            rollout_engine_num=n_rollout_instances + n_validate_instances,
        )

        self.agent_programs = set()
        self.pending_program_queue = deque()

        self.running_loop = None
        self.busy_loop_task = None
        self.stop_busy_loop_task = False

        # Register agent loop configs from file
        agent_loop_config_path = config.gen_actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                AGENT_LOOP_REGISTRY[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        # Initialize rollout trace config
        trace_config = self.config.gen_actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

        # Build logger
        # TODO(lhy): support >1 workers
        self.log_prefix = "AgentLoopWorker"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def set_agent_loop_manager(self, agent_loop_manager: ray.actor.ActorHandle):
        """Set the agent loop manager handle for communication.

        Args:
            agent_loop_manager: Handle to the agent loop manager actor.
        """
        self.agent_loop_manager = agent_loop_manager

    def set_reward_manager(self, reward_manager: ray.actor.ActorHandle):
        """Set the reward manager handle for sending processed data.

        Args:
            reward_manager: Handle to the reward manager actor.
        """
        self.reward_manager = reward_manager

    def add_agent_program(self, batch: KVBatchMeta | None):
        """Add a new agent program to the pending queue for processing.

        Args:
            batch (KVBatchMeta or None): Data to process, or None to signal termination.
        """
        if batch is None:
            self.pending_program_queue.append(None)
        elif isinstance(batch, KVBatchMeta):
            self.pending_program_queue.append(batch)
        else:
            raise TypeError(
                f"Unsupported data type: {type(batch)}. Expected KVBatchMeta or None."
            )

    def start_busy_loop(self):
        """Start the busy loop to continuously process agent programs from the queue."""
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        self.stop_busy_loop_task = False
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._launch_agent_loop())
        self.busy_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

    async def stop_busy_loop(self):
        """Stop the busy loop and wait for the current task to complete."""
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        await asyncio.gather(self.busy_loop_task)

    async def _launch_agent_loop(self):
        """Main loop that processes agent programs from the pending queue."""
        while not self.stop_busy_loop_task:
            if len(self.pending_program_queue) > 0:
                program = self.pending_program_queue.popleft()
                # psrl_logger.info(f"Processing program: {program.non_tensor_batch['uid'].tolist()[0]}")
                if program is None:
                    self.stop_busy_loop_task = True
                    continue
                await self.generate_trajectory(program)
            await asyncio.sleep(0)

    def _create_task_done_callback(self, task):
        """Create a callback function to handle task completion."""

        def task_done_callback(future):
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                psrl_logger.error(f"Task {task} failed with exception: {e}")
            finally:
                self.agent_programs.discard(task)

        return task_done_callback

    async def generate_trajectory(self, batch: KVBatchMeta):
        """Generate trajectories using the specified agent type based on configuration.

        This method only create the task (agent_loop) and add the task to the agent_programs set.
        But the task is not await here so different agent_loop can be run in parallel.

        Args:
            batch (KVBatchMeta): Input batch metadata containing prompts and metadata.
        """
        # by default, we assume it's a generation-only agent
        default_agent_name = "generate_only_agent"
        assert len(batch) == 1, "Only support single request for generation"

        fields = ["agent_name"]
        data = await tq.async_kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        version_tag = [tag.get("version_tag", -1) for tag in batch.tags]
        tu.assign_non_tensor_stack(data, "version_tag", version_tag)
        await tq.async_kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("version_tag"))
        agent_name = data.get("agent_name", [default_agent_name])[0]

        task = asyncio.create_task(self._run_agent_loop(agent_name, batch))
        task.add_done_callback(self._create_task_done_callback(task))
        self.agent_programs.add(task)

    async def _run_agent_loop(
        self,
        agent_name: str,
        batch: KVBatchMeta,
    ):
        """Execute the specified agent loop on the given requests.

        This method instantiates the agent loop based on the registered configuration
        and runs it with the provided requests. It handles retries based on termination reasons.

        Args:
            agent_name (str): Name of the agent loop to run.
            batch (KVBatchMeta): Input batch metadata containing prompts and metadata.
        """
        assert len(batch) == 1, "Only support single request for generation"

        fields = ["uid", "parent_id", "global_steps", "validate"]
        data = await tq.async_kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        request_ids = tu.get(data, "uid")
        if "parent_id" in data:
            prompt_index = tu.get(data, "parent_id")[0]
            request_index = tu.get(data, "uid")[0]
        else:
            prompt_index = tu.get(data, "uid")[0]
            request_index = tu.get(data, "uid")[0]

        with rollout_trace_attr(
            prompt_index=prompt_index,
            request_index=request_index,
            step=tu.get(data, "global_steps", default=-1),
            name=agent_name,
            validate=tu.get(data, "validate", default=False),
        ):
            assert agent_name in AGENT_LOOP_REGISTRY, (
                f"Agent loop {agent_name} not registered, registered agent loops: {AGENT_LOOP_REGISTRY.keys()}"
            )
            agent_loop_config = AGENT_LOOP_REGISTRY[agent_name]

            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                rollout_router=self.rollout_router,
                reward_manager=self.reward_manager,
                ps_manager_handle=self.ps_manager_handle,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
            )

            with log_dual_events(
                f"Agent loop with requests {tu.get(data, 'uid')}",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.GEN,
            ):
                retry_limit = self.config.gen_actor_rollout_ref.rollout.agent.retry_limit
                for retry_attempt in range(1, retry_limit + 1):
                    raise_on_error = (
                        retry_attempt == retry_limit
                    ) and self.config.gen_actor_rollout_ref.rollout.agent.raise_on_error
                    output, terminate_reason = await agent_loop.run_with_termination_handling(
                        batch, raise_on_error=raise_on_error
                    )

                    if terminate_reason not in (
                        TerminateReason.TIMEOUT,
                        TerminateReason.ENV_TIMEOUT,
                        TerminateReason.ERROR,
                        TerminateReason.UNKNOWN,
                    ):
                        break

                    # Retry if applicable
                    if retry_attempt < retry_limit:
                        psrl_logger.warning(
                            f"Agent loop for requests {tu.get(data, 'uid')} "
                            f"terminated with reason {terminate_reason.value} on "
                            f"attempt {retry_attempt}/{retry_limit}, retrying..."
                        )
                        continue

                if terminate_reason in (
                    TerminateReason.TIMEOUT,
                    TerminateReason.ENV_TIMEOUT,
                    TerminateReason.ERROR,
                    TerminateReason.UNKNOWN,
                    TerminateReason.ABORTED,
                ):
                    psrl_logger.warning(
                        f"Agent loop for requests {tu.get(data, 'uid')} "
                        f"terminated with reason {terminate_reason.value} "
                        f"after {retry_limit} attempts."
                    )
                    output = None
                else:
                    psrl_logger.debug(
                        f"Agent loop for requests {tu.get(data, 'uid')} "
                        f"terminated with reason {terminate_reason.value}."
                    )

            # Put the output into the TransferQueue and notify PSManager
            # + AgentLoopManager via metadata-only RPCs.
            if output is not None:
                is_validate = tu.get(data, "validate", default=False)
                with log_dual_events(
                    "Update request status",
                    psrl_logger,
                    level=logging.DEBUG,
                    event_type=EventType.OTHER,
                ):
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(
                        request_ids,
                        PSRL_RequestStatus.COMPLETED,
                        is_validate=is_validate,
                    )

                if update_status_success:
                    with log_dual_events(
                        f"Put requests {request_ids} into TransferQueue",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        await self.postprocess_output(output, batch)
            elif terminate_reason != TerminateReason.ABORTED:
                # Generation failed (e.g. HTTP error, timeout) without PSManager being notified.
                # The SMG already reserved a staleness-inventory entry for this request.
                # Abort it now so the RESERVED entry is freed and the buffer can make progress.
                psrl_logger.warning(
                    f"Generation failed for requests {request_ids} "
                    f"(terminate_reason={terminate_reason.value}), aborting in PSManager."
                )
                await self.ps_manager_handle.abort_requests.remote(request_ids)

    async def postprocess_output(self, output: TokenOutput, batch: KVBatchMeta):
        fields = []
        prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
        responses = torch.tensor(output.response_ids, dtype=torch.int64)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        
        field = output.as_dict()
        # do not store raw image/video
        field.pop("multi_modal_data", None)
        field["loss_mask"] = field["response_mask"]
        field["input_ids"] = input_ids
        field["position_ids"] = position_ids
        field["multi_modal_inputs"] = multi_modal_inputs
        prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
        field["seq_len"] = prompt_len + response_len
        field["prompt_len"] = prompt_len
        field["response_len"] = response_len
        fields.append(field)
        await tq.async_kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=list_of_dict_to_tensordict(fields),
        )
        
        await self.agent_loop_manager.occupy_requests.remote(
            request_id=batch.tags[0]["uid"],
            prompt_id=batch.tags[0].get("parent_id", batch.tags[0]["uid"]),
            rollout_instance_id=output.rollout_instance_id,
            version_tag=batch.tags[0]["version_tag"],
            is_validate=batch.partition_id == "val",
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs
        if output.multi_modal_data is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[0][input_ids[0] == self.processor.image_token_id] = 1
            mm_token_type_ids[0][input_ids[0] == self.processor.video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

import argparse
from datetime import datetime, timezone
from typing import Any


WORKERS_UPDATE_STATS_PATH = "/workers/update_stats"
WORKERS_UPDATE_WEIGHT_VERSION_PATH = "/workers/update_weight_version"
WORKERS_STATS_PATH = "/workers/stats"
ROUTING_LOOP_STATUS_PATH = "/routing_loop/status"
TITO_SESSIONS_PATH = "tito/sessions"


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    node = config
    for part in path.split("."):
        if node is None or not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node if node is not None else default


def build_rollout_router_args(config: Any, host: str, port: int, ps_manager_addr: str):
    from smg.launch_router import RouterArgs

    routing_method = str(cfg_get(config, "psrl.routing_strategy.method", "request_num_balance"))
    request_budget = int(cfg_get(config, "psrl.routing_strategy.request_budget", 1024))
    enable_group_sampling = bool(
        cfg_get(config, "psrl.routing_strategy.enable_group_sampling_on_multi_instances", False)
    )

    cli_args = argparse.Namespace(
        host=host,
        port=port,
        dp_aware=True,
        connection_mode="grpc",
        pd_disaggregation=False,
        prefill=None,
        decode=None,
        policy=routing_method,
        prefill_policy=None,
        decode_policy=None,
        disable_retries=True,
        max_concurrent_seqs_per_instance=int(
            cfg_get(config, "psrl.routing_strategy.max_concurrent_seqs_per_instance", 1024)
        ),
        cost_model_path=cfg_get(config, "psrl.routing_strategy.cost_model_path", None),
        max_num_waiting_reqs_after_preemption=int(
            cfg_get(config, "psrl.routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
        ),
        delta_throughput_threshold=float(
            cfg_get(config, "psrl.routing_strategy.delta_throughput_threshold", 0.5)
        ),
        max_prompt_length=int(
            cfg_get(config, "data.max_prompt_length", cfg_get(config, "rollout.prompt_length", 8192))
        ),
        request_budget=request_budget,
        enable_routing_loop=routing_method
        in {"request_num_balance", "throughput_optimal", "throughput_optimal_with_budget"},
        routing_loop_check_interval_ms=int(cfg_get(config, "psrl.routing_strategy.check_interval_in_ms", 10)),
        routing_loop_request_sort_key=str(
            cfg_get(config, "psrl.routing_strategy.request_sort_indicator", "short_length")
        ),
        routing_loop_multi_priority_queue=bool(
            cfg_get(config, "psrl.routing_strategy.enable_multi_priority_queue", False)
        ),
        worker_selection_strategy="psrl",
        psrl_ps_manager_addr=ps_manager_addr,
        psrl_enable_mig_strategy=bool(cfg_get(config, "psrl.sync_and_mig_strategy.mig.enable", False)),
        psrl_candidate_sort_key=str(cfg_get(config, "psrl.routing_strategy.candidate_sort_indicator", "version")),
        psrl_enable_group_sticky=not enable_group_sampling,
        enable_tito=True,
        tito_debug=bool(cfg_get(config, "psrl.rollout_gateway.tito_debug", False)),
        tito_gc_threshold=cfg_get(config, "psrl.rollout_gateway.tito_gc_threshold", None),
        service_discovery=False,
        prometheus_port=None,
        request_timeout_secs=2**64 - 1,
        log_level="warn",
        log_dir=cfg_get(config, "psrl.logging_path", None),
        api_key=None,
        disable_health_check=True,
    )
    return RouterArgs.from_cli_args(cli_args, use_router_prefix=False)


def build_reward_router_args(config: Any, host: str, port: int, prometheus_port: int | None):
    from smg.launch_router import RouterArgs

    cli_args = argparse.Namespace(
        host=host,
        port=port,
        dp_aware=False,
        connection_mode="grpc",
        pd_disaggregation=False,
        prefill=None,
        decode=None,
        policy="round_robin",
        prefill_policy=None,
        decode_policy=None,
        disable_retries=True,
        max_concurrent_seqs_per_instance=1024,
        cost_model_path=None,
        max_num_waiting_reqs_after_preemption=int(
            cfg_get(config, "psrl.routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
        ),
        delta_throughput_threshold=float(
            cfg_get(config, "psrl.routing_strategy.delta_throughput_threshold", 0.5)
        ),
        max_prompt_length=32768,
        request_budget=1024,
        enable_routing_loop=False,
        worker_selection_strategy="naive",
        service_discovery=False,
        prometheus_port=prometheus_port,
        request_timeout_secs=2**64 - 1,
        log_level="warn",
        log_dir=cfg_get(config, "psrl.logging_path", None),
        api_key=None,
        disable_health_check=True,
    )
    return RouterArgs.from_cli_args(cli_args, use_router_prefix=False)


def build_worker_registration_payload(
    *,
    url: str,
    model_id: str,
    max_model_len: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
) -> dict[str, Any]:
    return {
        "url": url,
        "worker_type": "regular",
        "connection_mode": "grpc",
        "runtime_type": "vllm",
        "models": [{"id": model_id}],
        "labels": {
            "max_model_len": str(max_model_len),
            "dp_size": str(dp_size),
            "tp_size": str(tp_size),
            "pp_size": str(pp_size),
        },
    }


def build_pause_resume_payload(instance_ids: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for instance_id in instance_ids:
        if isinstance(instance_id, tuple):
            base_worker_id, dp_rank = instance_id
            payload.append({"base_worker_id": base_worker_id, "dp_rank": dp_rank})
        else:
            payload.append({"base_worker_id": instance_id})
    return payload


def build_weight_version_updates(instance_ids: list[tuple[str, int]], weight_version: int) -> list[dict[str, Any]]:
    return [
        {"worker_id": base_worker_id, "dp_rank": dp_rank, "weight_version": weight_version}
        for base_worker_id, dp_rank in instance_ids
    ]


def build_worker_stats_update(worker_id: str, dp_rank: int, snapshot: dict[str, Any]) -> dict[str, Any]:
    scheduler_stats = snapshot.get("scheduler_stats", {}) if isinstance(snapshot, dict) else {}
    return {
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "timestamp": _parse_timestamp(snapshot.get("timestamp") if isinstance(snapshot, dict) else None),
        "scheduler_stats": {
            "req_id_to_prompt_token_num": scheduler_stats.get("req_id_to_prompt_token_num", {}),
            "req_id_to_response_token_num": scheduler_stats.get("req_id_to_response_token_num", {}),
            "num_running_reqs": int(scheduler_stats.get("num_running_reqs", 0)),
            "num_waiting_reqs": int(scheduler_stats.get("num_waiting_reqs", 0)),
            "kv_cache_usage": float(scheduler_stats.get("kv_cache_usage", 0.0)),
        },
    }

def _parse_timestamp(value: Any) -> str:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

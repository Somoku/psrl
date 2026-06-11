import asyncio
import logging
import math
import os
import time
from collections import defaultdict

import ray

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.elastic_rm.cluster_topology import ClusterTopology, GPUSlot, InstanceIdentifier, InstanceStatus
from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.elastic_rm.scaling_policy import InstanceSignal, ScalingPolicy
from psrl.utils.logger import DualOutputHandler, FileOnlyHandler
from psrl.utils.server.command import Command, CommandType
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

monitor_logger = logging.getLogger("ElasticMonitor")
monitor_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class ElasticExecutor:
    def __init__(
        self,
        config,
        roles: list[tuple[PSRL_Role, str]],
        coordinators: dict[PSRL_Role, dict[str, ray.actor.ActorHandle]],
        agent_loop_manager: ray.actor.ActorHandle | None = None,
        elastic_rm_config: dict | None = None,
    ):
        self.coordinators = coordinators
        self.roles = roles
        self.elastic_rm_config = elastic_rm_config or {}
        self.config = config
        self.agent_loop_manager = agent_loop_manager

        self.instances_status_flags: dict[PSRL_Role, dict[str, dict[RolloutInstanceId, InstanceStatus]]] = {}
        self.instances_engine_stats: dict[PSRL_Role, dict[str, dict[RolloutInstanceId, dict]]] = {}
        self.topology = ClusterTopology()

        for role_name, model_name in self.roles:
            self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
            self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})

        self.scale_up_task_queue: asyncio.Queue = asyncio.Queue()
        self.scale_down_task_queue: asyncio.Queue = asyncio.Queue()

        # Task management
        self.running_loop = None
        self.monitor_task = None
        self.scale_up_task = None
        self.scale_down_task = None
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

        # Scaling policy
        self.scaling_policy = ScalingPolicy(config=self.config, policy_config=self.elastic_rm_config)

        # Whether current decision is being executed.
        self._decision_execution_in_progress = False
        self._next_decision_id = 1
        self._decision_pending_action_counts: dict[int, int] = {}

        # Consecutive monitor ticks where policy is blocked by an unfinished scale decision.
        self._execution_in_progress_stall_ticks: int = 0
        # When stall ticks reach this threshold, clear in-flight decision state so policy can proceed.
        # 0 = disabled (only warnings, no abandon).
        self._decision_abandon_stall_ticks: int = int(
            self.elastic_rm_config.get("decision_execution_abandon_stall_ticks", 0)
        )

        self._last_monitor_instance_log_ms: float = 0.0
        self._monitor_instance_log_interval_ms: int = int(
            self.elastic_rm_config.get("monitor_instance_log_interval_ms", 5000)
        )
        self._enable_monitor_instance_log: bool = bool(self.elastic_rm_config.get("enable_monitor_instance_log", True))
        self.router_backlog_by_role: dict[str, int] = {}
        self.trainer_waiting_hint: dict[str, object] = {
            "trainer_busy": True,
            "waiting_buffer_id": None,
            "waiting_on": "none",
            "breakdown": {},
        }

        # Fraction [0,1] of waiting uids to ABORT per instance after scale-up (FIFO / queue head).
        _r = float(self.elastic_rm_config.get("post_scale_up_abort_waiting_ratio", 1.0))
        self._post_scale_up_abort_waiting_ratio = max(0.0, min(1.0, _r))
        # Per-tick timeout for coordinator Ray RPCs; avoids monitor loop hanging forever when coordinators stall.
        self._coordinator_sync_timeout_s = float(self.elastic_rm_config.get("coordinator_sync_timeout_s", 60.0))
        # Optional timeout for SLEEP/WAKE_UP/ABORT issued during elastic scale (None = wait forever).
        _cmd_to = self.elastic_rm_config.get("coordinator_command_timeout_s", None)
        if _cmd_to is None or float(_cmd_to) <= 0:
            self._coordinator_command_timeout_s: float | None = None
        else:
            self._coordinator_command_timeout_s = float(_cmd_to)

        # Logger
        self.log_prefix = "ElasticExecutor"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        monitor_logger.propagate = False
        monitor_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, "ElasticMonitor"))

    def register_role(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_ids: list[RolloutInstanceId],
        gpu_slots_per_instance: list[frozenset[GPUSlot]],
    ):
        """Register all instances for one (role, model) pair with their GPU placements.

        Args:
            instance_ids: list of RolloutInstanceId tuples, e.g. [(replica_id, dp_rank), ...].
            gpu_slots_per_instance: parallel list of frozenset[GPUSlot], one per instance.
        """
        if len(instance_ids) != len(gpu_slots_per_instance):
            raise ValueError(
                f"instance_ids ({len(instance_ids)}) and gpu_slots_per_instance "
                f"({len(gpu_slots_per_instance)}) must have the same length."
            )
        self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
        self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
        for instance_id, gpu_slots in zip(instance_ids, gpu_slots_per_instance):
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP
            self.instances_engine_stats[role_name][model_name][instance_id] = EngineStats.get_default_snapshot()
            self.topology.register(
                role=role_name,
                model_name=model_name,
                instance_id=instance_id,
                gpu_slots=gpu_slots,
            )

    def select_initial_awake_ids(
        self,
        role_name: PSRL_Role,
        model_name: str,
        target_awake_num: int,
        min_awake_num: int = 0,
    ) -> list[RolloutInstanceId]:
        """Select non-conflicting instances to wake during initial bootstrap.

        The topology derives the occupied GPU set from all currently-AWAKEN
        instances, so successive calls for different roles automatically avoid
        GPUs already claimed by earlier calls (as long as selected instances are
        marked AWAKEN before the next call — which this method does).

        Returns:
            list of RolloutInstanceId that should be woken up.
        """
        status_dict = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        all_ids = list(status_dict.keys())
        selected_ids = self.topology.select_non_conflicting_awake_ids(
            role=role_name,
            model_name=model_name,
            instance_ids=all_ids,
            target_awake_num=target_awake_num,
            min_awake_num=min_awake_num,
        )
        for instance_id in selected_ids:
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.AWAKEN
            self.topology.set_status(role_name, model_name, instance_id, InstanceStatus.AWAKEN)
        return selected_ids

    def snapshot(self) -> dict:
        return {
            "status": self.instances_status_flags,
            "topology": self.topology.snapshot(),
        }

    async def start_busy_loop(self):
        if self.monitor_task is not None and not self.monitor_task.done():
            return

        self.running_loop = asyncio.get_running_loop()
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

        self.monitor_task = self.running_loop.create_task(self._monitor_loop())
        self.monitor_task.add_done_callback(lambda f: f.result())

        self.scale_up_task = self.running_loop.create_task(self._scale_up_handler_loop())
        self.scale_up_task.add_done_callback(lambda f: f.result())

        self.scale_down_task = self.running_loop.create_task(self._scale_down_handler_loop())
        self.scale_down_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        if self.monitor_task is None and self.scale_up_task is None and self.scale_down_task is None:
            return

        self.stop_monitor = True
        self.stop_scale_up = True
        self.stop_scale_down = True

        tasks_to_wait = []
        if self.monitor_task is not None:
            tasks_to_wait.append(self.monitor_task)
        if self.scale_up_task is not None:
            tasks_to_wait.append(self.scale_up_task)
        if self.scale_down_task is not None:
            tasks_to_wait.append(self.scale_down_task)
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _monitor_loop(self):
        while not self.stop_monitor:
            try:
                # Pull fresh engine status from coordinators first, then decide scaling.
                await self._sync_engine_status_from_coordinators()
                await self._sync_router_backlog_from_coordinators()
                await self._sync_trainer_waiting_hint()
                signals = self._build_instance_signals()
                self._maybe_log_instance_signals(signals)
                decision = self.scaling_policy.decide(
                    signals,
                    execution_in_progress=self._decision_execution_in_progress,
                    router_backlog_by_role=self.router_backlog_by_role,
                    trainer_waiting_hint=self.trainer_waiting_hint,
                )
                if not decision.actions and decision.reason == "decision_execution_in_progress":
                    self._execution_in_progress_stall_ticks += 1
                    if self._decision_abandon_stall_ticks > 0 and (
                        self._execution_in_progress_stall_ticks >= self._decision_abandon_stall_ticks
                    ):
                        self._abandon_in_flight_decision(
                            reason="decision_execution_stall_ticks_exceeded",
                            stall_ticks=self._execution_in_progress_stall_ticks,
                        )
                    elif self._execution_in_progress_stall_ticks in (1, 30, 60) or (
                        self._execution_in_progress_stall_ticks > 60
                        and self._execution_in_progress_stall_ticks % 120 == 0
                    ):
                        psrl_logger.warning(
                            "elastic_rm: policy blocked by in-flight scaling for %d monitor ticks; "
                            "pending=%s. Likely causes: (1) coordinator.exec_command stuck inside "
                            "SLEEP/WAKE_UP/ABORT (gen/RM workers or router not returning); "
                            "(2) another client's command ahead in the same coordinator queue never finishes "
                            "(head-of-line blocking). Check ElasticExecutor.log for the last "
                            "coordinator_cmd START line without matching END.",
                            self._execution_in_progress_stall_ticks,
                            dict(self._decision_pending_action_counts),
                        )
                else:
                    self._execution_in_progress_stall_ticks = 0

                if decision.actions:
                    decision_id = self._next_decision_id
                    self._next_decision_id += 1
                    self._decision_execution_in_progress = True
                    self._decision_pending_action_counts[decision_id] = len(decision.actions)
                    psrl_logger.info(
                        "Policy decision accepted: decision_id=%d, reason=%s, "
                        "lambda=%.6f, role_to_total_mu=%s, actions=%s",
                        decision_id,
                        decision.reason,
                        decision.estimated_lambda,
                        decision.role_to_total_mu,
                        [
                            {
                                "action_type": action.action_type,
                                "role_name": action.role_name.name,
                                "model_name": action.model_name,
                                "num_instances": action.num_instances,
                                "preferred_instance_ids": action.preferred_instance_ids,
                                "reason": action.reason,
                            }
                            for action in decision.actions
                        ],
                    )
                    psrl_logger.info("elastic_rm trainer_waiting_hint=%s", self.trainer_waiting_hint)
                    for action in decision.actions:
                        task = {
                            "role_name": action.role_name,
                            "model_name": action.model_name,
                            "num_instances": action.num_instances,
                            "preferred_instance_ids": action.preferred_instance_ids or [],
                            "reason": action.reason,
                            "decision_id": decision_id,
                        }
                        if action.action_type == "scale_up":
                            self.scale_up_task_queue.put_nowait(task)
                        elif action.action_type == "scale_down":
                            self.scale_down_task_queue.put_nowait(task)
                        else:
                            psrl_logger.warning(
                                "Unknown action_type=%s for decision_id=%d, mark as finished immediately.",
                                action.action_type,
                                decision_id,
                            )
                            self._mark_decision_action_finished(decision_id)
            except Exception:
                # Without this, a single Ray/sync/decide exception kills the monitor task forever
                # (see start_busy_loop done_callback calling task.result()).
                psrl_logger.exception("elastic_rm _monitor_loop iteration failed; will retry after sleep.")
            await asyncio.sleep(max(self.scaling_policy.monitor_interval_ms / 1000, 0.01))

    async def _scale_up_handler_loop(self):
        while not self.stop_scale_up:
            if self.scale_up_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_up = self.scale_up_task_queue.get_nowait()
            decision_id = role_need_to_scale_up.get("decision_id")
            try:
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_up,
                )
                instances_to_scaled_down = self._find_instances_to_scaled_down_for_other_roles(role_need_to_scale_up)
                if instances_to_scaled_down:
                    psrl_logger.info(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other count=%s detail=%s",
                        decision_id,
                        len(instances_to_scaled_down),
                        instances_to_scaled_down,
                    )
                    await asyncio.gather(
                        *[self._scale_down_instance(instance) for instance in instances_to_scaled_down]
                    )
                    psrl_logger.info(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other done",
                        decision_id,
                    )

                instances_to_scaled_up = self._find_instances_to_scaled_up(role_need_to_scale_up)
                if not instances_to_scaled_up:
                    psrl_logger.warning("No instances can be scaled up for role %s", role_need_to_scale_up)
                    continue
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s wake_targets=%s",
                    decision_id,
                    instances_to_scaled_up,
                )
                await asyncio.gather(*[self._scale_up_instance(instance) for instance in instances_to_scaled_up])
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s wake_targets done; post_scale_up_abort",
                    decision_id,
                )
                await self._interrupt_waiting_after_scale_up(
                    role_name=role_need_to_scale_up["role_name"],
                    model_name=role_need_to_scale_up["model_name"],
                )
                psrl_logger.info("elastic_rm scale_up_handler decision_id=%s post_scale_up_abort done", decision_id)
            finally:
                self._mark_decision_action_finished(decision_id)

    async def _scale_down_handler_loop(self):
        while not self.stop_scale_down:
            if self.scale_down_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_down = self.scale_down_task_queue.get_nowait()
            decision_id = role_need_to_scale_down.get("decision_id")
            try:
                psrl_logger.info(
                    "elastic_rm scale_down_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_down,
                )
                instances_to_scaled_down = self._find_instances_to_scaled_down_in_role(role_need_to_scale_down)
                if not instances_to_scaled_down:
                    psrl_logger.warning("No instances can be scaled down for role %s", role_need_to_scale_down)
                    continue
                psrl_logger.info(
                    "elastic_rm scale_down_handler decision_id=%s sleep_targets=%s",
                    decision_id,
                    instances_to_scaled_down,
                )
                await asyncio.gather(*[self._scale_down_instance(instance) for instance in instances_to_scaled_down])
                psrl_logger.info("elastic_rm scale_down_handler decision_id=%s sleep_targets done", decision_id)
            finally:
                self._mark_decision_action_finished(decision_id)

    def _abandon_in_flight_decision(self, *, reason: str, stall_ticks: int) -> None:
        """Clear local decision bookkeeping so scaling policy is no longer blocked.

        Scale handler tasks may still complete later; their _mark_decision_action_finished
        calls become no-ops if the decision was already cleared here. Cluster state may
        diverge from ElasticExecutor flags until the next sync — same as coordinator timeouts.
        """
        if not self._decision_execution_in_progress and not self._decision_pending_action_counts:
            return
        pending = dict(self._decision_pending_action_counts)
        psrl_logger.error(
            "elastic_rm: abandoning in-flight scaling decision (%s): stall_ticks=%d >= threshold=%d, "
            "pending_action_counts=%s. Policy will accept new decisions; handlers may still finish RPCs.",
            reason,
            stall_ticks,
            self._decision_abandon_stall_ticks,
            pending,
        )
        self._decision_pending_action_counts.clear()
        self._decision_execution_in_progress = False
        self._execution_in_progress_stall_ticks = 0

    def _mark_decision_action_finished(self, decision_id: int | None):
        if decision_id is None:
            return
        if decision_id not in self._decision_pending_action_counts:
            return
        remaining = self._decision_pending_action_counts[decision_id] - 1
        if remaining > 0:
            self._decision_pending_action_counts[decision_id] = remaining
            return
        self._decision_pending_action_counts.pop(decision_id, None)
        if not self._decision_pending_action_counts:
            self._decision_execution_in_progress = False
            psrl_logger.info("elastic_rm decision_id=%d execution completed.", decision_id)

    def _maybe_log_instance_signals(self, signals: list[InstanceSignal]):
        if not self._enable_monitor_instance_log:
            return
        if not signals:
            return
        now_ms = asyncio.get_running_loop().time() * 1000
        if now_ms - self._last_monitor_instance_log_ms < max(self._monitor_instance_log_interval_ms, 0):
            return
        self._last_monitor_instance_log_ms = now_ms

        rows: list[dict] = []
        for s in signals:
            role_name = getattr(s.role_name, "name", str(s.role_name))
            rows.append(
                {
                    "role": role_name,
                    "model": s.model_name,
                    "instance": str(s.instance_id),
                    "status": "AWAKEN" if bool(s.is_awaken) else "ASLEEP",
                    "running": int(s.running_queue_num),
                    "waiting": int(s.waiting_queue_num),
                    "kv_cache": float(s.kv_cache_utilization),
                    "throughput": float(s.generation_throughput),
                    "ts": s.snapshot_timestamp,
                }
            )
        rows.sort(key=lambda r: (r["status"] != "AWAKEN", r["role"], r["model"], r["instance"]))
        awake_by_role: dict[str, int] = {}
        awake_by_role_model: dict[str, int] = {}
        for row in rows:
            if row["status"] != "AWAKEN":
                continue
            role_key = str(row["role"])
            role_model_key = f"{role_key}/{row['model']}"
            awake_by_role[role_key] = awake_by_role.get(role_key, 0) + 1
            awake_by_role_model[role_model_key] = awake_by_role_model.get(role_model_key, 0) + 1
        monitor_logger.info("------------------------------------------------------------")
        for row in rows:
            monitor_logger.info("Instance current Status: %s", row)
        monitor_logger.info("------------------------------------------------------------")
        psrl_logger.info(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )
        monitor_logger.info(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )

    async def _await_elastic_coordinator_command(
        self,
        coordinator: ray.actor.ActorHandle,
        command: Command,
        *,
        stage: str,
    ) -> object | None:
        """Run coordinator.exec_command with optional timeout and structured tracing for elastic_rm debugging."""
        t0 = time.monotonic()
        psrl_logger.info(
            "elastic_rm coordinator_cmd START stage=%s type=%s args=%s timeout_s=%s",
            stage,
            command.type.name,
            command.get_args(),
            self._coordinator_command_timeout_s,
        )
        kwargs: dict = {"blocking": True}
        if self._coordinator_command_timeout_s is not None:
            kwargs["timeout"] = self._coordinator_command_timeout_s
        try:
            result = await coordinator.exec_command.remote(command, **kwargs)
        except Exception:
            psrl_logger.exception(
                "elastic_rm coordinator_cmd EXCEPTION stage=%s type=%s elapsed_s=%.3f",
                stage,
                command.type.name,
                time.monotonic() - t0,
            )
            raise
        elapsed = time.monotonic() - t0
        psrl_logger.info(
            "elastic_rm coordinator_cmd END stage=%s type=%s elapsed_s=%.3f result=%r",
            stage,
            command.type.name,
            elapsed,
            result,
        )
        if result is None and self._coordinator_command_timeout_s is not None:
            psrl_logger.error(
                "elastic_rm coordinator_cmd got None (timeout or failure) stage=%s — "
                "ElasticExecutor local flags may diverge from cluster; consider coordinator_command_timeout_s "
                "and check coordinator logs for stuck SLEEP/WAKE_UP/ABORT.",
                stage,
            )
        return result

    async def _scale_up_instance(self, instance_to_scaled_up: dict):
        instance_role = instance_to_scaled_up["role_name"]
        instance_model_name = instance_to_scaled_up["model_name"]
        instance_id = instance_to_scaled_up["instance_id"]  # RolloutInstanceId tuple

        coordinator = self.coordinators[instance_role][instance_model_name]
        result = await self._await_elastic_coordinator_command(
            coordinator,
            Command(
                type=CommandType.WAKE_UP,
                instance_ids=[instance_id],
            ),
            stage=(
                f"WAKE_UP role={getattr(instance_role, 'name', instance_role)} "
                f"model={instance_model_name} instance={instance_id}"
            ),
        )
        if result is None:
            return
        self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.AWAKEN
        self.topology.set_status(instance_role, instance_model_name, instance_id, InstanceStatus.AWAKEN)

    def _waiting_uids_for_abort_by_ratio(self, normalized_waiting_uids: list[int]) -> list[int]:
        """Take the first k waiting uids (FIFO vs queue order); k = floor(n * ratio)."""
        if not normalized_waiting_uids:
            return []
        r = self._post_scale_up_abort_waiting_ratio
        if r <= 0.0:
            return []
        if r >= 1.0:
            return list(normalized_waiting_uids)
        k = min(
            len(normalized_waiting_uids),
            int(math.floor(float(len(normalized_waiting_uids)) * r + 1e-12)),
        )
        return normalized_waiting_uids[:k]

    async def _interrupt_waiting_after_scale_up(self, role_name: PSRL_Role, model_name: str):
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        awaken_instance_ids = [
            instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN
        ]
        if not awaken_instance_ids:
            return
        instance_to_uids: dict = {}
        role_engine_stats = self.instances_engine_stats.get(role_name, {}).get(model_name, {})
        for instance_id in awaken_instance_ids:
            snapshot = role_engine_stats.get(instance_id, {})
            if not isinstance(snapshot, dict):
                continue
            scheduler_stats = snapshot.get("scheduler_stats", {})
            if not isinstance(scheduler_stats, dict):
                continue
            waiting_uids = scheduler_stats.get("req_id_in_waiting", [])
            normalized_waiting_uids: list[int] = []
            for uid in waiting_uids:
                try:
                    normalized_waiting_uids.append(int(uid))
                except (TypeError, ValueError):
                    psrl_logger.warning(
                        "Skip non-integer waiting uid during post-scale-up ABORT: role=%s model=%s instance=%s uid=%r",
                        role_name,
                        model_name,
                        instance_id,
                        uid,
                    )
            selected = self._waiting_uids_for_abort_by_ratio(normalized_waiting_uids)
            if selected:
                instance_to_uids[instance_id] = selected

        coordinator = self.coordinators.get(role_name, {}).get(model_name)
        if coordinator is None:
            psrl_logger.warning(
                "Skip post-scale-up ABORT: coordinator missing for role=%s model=%s.",
                role_name,
                model_name,
            )
            return

        if not instance_to_uids:
            psrl_logger.info(
                "Skip post-scale-up ABORT: no waiting uids selected (ratio=%.4f) role=%s model=%s.",
                self._post_scale_up_abort_waiting_ratio,
                role_name,
                model_name,
            )
            return

        try:
            interrupted_request_num = await self._await_elastic_coordinator_command(
                coordinator,
                Command(
                    type=CommandType.ABORT,
                    instance_to_uids=instance_to_uids,
                ),
                stage=(
                    f"ABORT_post_scale_up role={getattr(role_name, 'name', role_name)} model={model_name} "
                    f"instances={sorted(instance_to_uids.keys())}"
                ),
            )
            if interrupted_request_num is None:
                psrl_logger.warning(
                    "elastic_rm post-scale-up ABORT returned None (timeout?); role=%s model=%s",
                    role_name,
                    model_name,
                )
                return
            psrl_logger.info(
                (
                    "Post-scale-up rebalancing abort executed: role=%s model=%s "
                    "awake_instances=%s uid_sources=%s interrupted=%s abort_waiting_ratio=%.4f"
                ),
                role_name,
                model_name,
                awaken_instance_ids,
                sorted(instance_to_uids.keys()),
                interrupted_request_num,
                self._post_scale_up_abort_waiting_ratio,
            )
        except Exception as exc:
            psrl_logger.warning(
                "elastic_rm post-scale-up ABORT failed for role=%s model=%s instances=%s: %s",
                role_name,
                model_name,
                awaken_instance_ids,
                exc,
            )

    async def _scale_down_instance(self, instance_to_scaled_down: dict):
        instance_role = instance_to_scaled_down["role_name"]
        instance_model_name = instance_to_scaled_down["model_name"]
        instance_id = instance_to_scaled_down["instance_id"]  # RolloutInstanceId tuple
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        role_status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {})
        awaken_count = sum(1 for status in role_status.values() if status == InstanceStatus.AWAKEN)
        # Concurrency safety: even if multiple scale-down tasks race, keep at least
        # `min_awake_per_role` alive for each role/model.
        if awaken_count <= min_awake_per_role:
            psrl_logger.info(
                "Skip scale down for role=%s model=%s instance=%s: keep at least %d awaken instances.",
                instance_role,
                instance_model_name,
                instance_id,
                min_awake_per_role,
            )
            return

        # When min_awake_per_role==0 only: do not sleep the last awake instance if the engine
        # still has running/waiting work. When min_awake_per_role>0, allow shrinking straight
        # down to the configured floor without this queue gate.
        if min_awake_per_role == 0 and awaken_count == 1:
            running_n, waiting_n = self.get_instance_running_waiting(
                InstanceIdentifier(role=instance_role, model_name=instance_model_name, instance_id=instance_id)
            )
            if running_n > 0 or waiting_n > 0:
                psrl_logger.info(
                    "Skip scale down for role=%s model=%s instance=%s: sole awake instance still has "
                    "running=%d waiting=%d (min_awake_per_role=0).",
                    instance_role,
                    instance_model_name,
                    instance_id,
                    running_n,
                    waiting_n,
                )
                return

        coordinator = self.coordinators[instance_role][instance_model_name]
        result = await self._await_elastic_coordinator_command(
            coordinator,
            Command(
                type=CommandType.SLEEP,
                instance_ids=[instance_id],
            ),
            stage=(
                f"SLEEP role={getattr(instance_role, 'name', instance_role)} "
                f"model={instance_model_name} instance={instance_id}"
            ),
        )
        if result is None:
            return
        self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.ASLEEP
        self.topology.set_status(instance_role, instance_model_name, instance_id, InstanceStatus.ASLEEP)
        self.instances_engine_stats[instance_role][instance_model_name][instance_id] = None

    def _find_instances_to_scaled_up(self, role_need_to_scale_up: dict):
        role_name = role_need_to_scale_up["role_name"]
        model_name = role_need_to_scale_up["model_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        preferred_instance_ids = list(role_need_to_scale_up.get("preferred_instance_ids", []))
        status_dict = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        all_asleep_ids = [
            instance_id for instance_id, status in status_dict.items() if status == InstanceStatus.ASLEEP
        ]
        if not all_asleep_ids:
            return None
        # Prefer the suggested instances; if none are available (e.g. already awake due to state race),
        # fall back to any asleep instance rather than failing the entire scale-up.
        if preferred_instance_ids:
            preferred_available = [
                instance_id for instance_id in all_asleep_ids if instance_id in preferred_instance_ids
            ]
            if preferred_available:
                candidate_ids = preferred_available
            else:
                psrl_logger.info(
                    "Preferred instances %s not asleep for role=%s model=%s, falling back to all asleep instances.",
                    preferred_instance_ids,
                    role_name,
                    model_name,
                )
                candidate_ids = all_asleep_ids
        else:
            candidate_ids = all_asleep_ids

        # First pass: respect preferred ordering and cross-role GPU conflict guard.
        filtered_ids = [
            instance_id
            for instance_id in candidate_ids
            if not self.topology.has_other_role_awaken_on_shared_gpu(role_name, model_name, instance_id)
        ]

        # Fallback: when preferred candidates are all filtered out by conflict guard,
        # try all asleep instances. This avoids force-wake starvation where policy
        # keeps requesting a fixed preferred instance id that is temporarily conflicted.
        if not filtered_ids and preferred_instance_ids:
            fallback_filtered_ids = [
                instance_id
                for instance_id in all_asleep_ids
                if not self.topology.has_other_role_awaken_on_shared_gpu(role_name, model_name, instance_id)
            ]
            if fallback_filtered_ids:
                psrl_logger.info(
                    (
                        "Preferred instances %s are conflict-filtered for role=%s model=%s; "
                        "fallback to non-conflicting asleep instances %s."
                    ),
                    preferred_instance_ids,
                    role_name,
                    model_name,
                    fallback_filtered_ids,
                )
                filtered_ids = fallback_filtered_ids
        if not filtered_ids:
            return None
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": instance_id}
            for instance_id in filtered_ids[:num_instances]
        ]

    def _find_instances_to_scaled_down_for_other_roles(self, role_need_to_scale_up: dict):
        target_role = role_need_to_scale_up["role_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        candidates: list[dict] = []
        removable_budget: dict[tuple[PSRL_Role, str], int] = {}
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))

        for role_name, model_name in self.roles:
            if role_name == target_role:
                continue
            role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
            awaken_ids = [
                instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN
            ]
            # Do not cede more than the removable budget of this role/model.
            max_removable = max(0, len(awaken_ids) - min_awake_per_role)
            if max_removable <= 0:
                continue
            removable_budget[(role_name, model_name)] = max_removable
            for instance_id, status in role_status.items():
                if status == InstanceStatus.AWAKEN:
                    candidates.append({"role_name": role_name, "model_name": model_name, "instance_id": instance_id})

        if not candidates:
            return None
        candidates.sort(
            key=lambda item: self.get_instance_kv_cache_usage(
                InstanceIdentifier(
                    role=item["role_name"],
                    model_name=item["model_name"],
                    instance_id=item["instance_id"],
                )
            )
        )
        picked: list[dict] = []
        for candidate in candidates:
            key = (candidate["role_name"], candidate["model_name"])
            if removable_budget.get(key, 0) <= 0:
                continue
            picked.append(candidate)
            removable_budget[key] -= 1
            if len(picked) >= num_instances:
                break
        return picked if picked else None

    def _find_instances_to_scaled_down_in_role(self, role_need_to_scale_down: dict):
        role_name = role_need_to_scale_down["role_name"]
        model_name = role_need_to_scale_down["model_name"]
        num_instances = int(role_need_to_scale_down.get("num_instances", 1))
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        preferred_instance_ids = list(role_need_to_scale_down.get("preferred_instance_ids", []))
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        # Compute max_scalable_down from the TOTAL awake count, not the preferred-filtered subset.
        # Previously this was computed after preferred filtering, which caused max_scalable_down=0
        # whenever only 1 preferred instance was awake (e.g. 1 preferred out of 8 awake total).
        all_awake_ids = [instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN]
        if not all_awake_ids:
            return None
        max_scalable_down = len(all_awake_ids) - min_awake_per_role
        if max_scalable_down <= 0:
            psrl_logger.info(
                "Skip scale down for role=%s model=%s: keep at least %d awaken instances (total awake=%d).",
                role_name,
                model_name,
                min_awake_per_role,
                len(all_awake_ids),
            )
            return None
        # Prefer the suggested instances; if none are awake (e.g. already asleep due to state race),
        # fall back to all awake instances rather than failing the entire scale-down.
        if preferred_instance_ids:
            preferred_available = [
                instance_id for instance_id in all_awake_ids if instance_id in preferred_instance_ids
            ]
            if preferred_available:
                candidate_ids = preferred_available
            else:
                psrl_logger.info(
                    "Preferred instances %s not awake for role=%s model=%s, falling back to all awake instances.",
                    preferred_instance_ids,
                    role_name,
                    model_name,
                )
                candidate_ids = list(all_awake_ids)
        else:
            candidate_ids = list(all_awake_ids)
        candidate_ids = sorted(
            candidate_ids,
            key=lambda instance_id: self.get_instance_kv_cache_usage(
                InstanceIdentifier(
                    role=role_name,
                    model_name=model_name,
                    instance_id=instance_id,
                ),
            ),
        )
        effective_num_instances = min(num_instances, max_scalable_down)
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": instance_id}
            for instance_id in candidate_ids[:effective_num_instances]
        ]

    async def _await_coordinator_refs_with_per_ref_timeout(
        self,
        refs: list,
        task_keys: list,
        op_label: str,
        *,
        trace_router_backlog: bool = False,
    ) -> list:
        """Await each coordinator Ray ObjectRef with its own timeout (parallel).

        Wrapping a single ``asyncio.gather`` in one ``wait_for`` lets the slowest
        RPC drop every other result for that tick; elastic_rm then mis-reads load.
        """
        timeout_s = self._coordinator_sync_timeout_s

        async def _one(ref, key):
            t0 = time.monotonic()
            if trace_router_backlog:
                log_elastic_rm_backlog_diag(
                    psrl_logger,
                    "stage=ElasticExecutor_wait_ref_begin op=%s key=%s timeout_s=%s",
                    op_label,
                    key,
                    timeout_s if timeout_s > 0 else "none",
                )
            try:
                if timeout_s > 0:
                    out = await asyncio.wait_for(ref, timeout=timeout_s)
                else:
                    out = await ref
                if trace_router_backlog:
                    log_elastic_rm_backlog_diag(
                        psrl_logger,
                        "stage=ElasticExecutor_wait_ref_done op=%s key=%s total_s=%.3f",
                        op_label,
                        key,
                        time.monotonic() - t0,
                    )
                return out
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    "elastic_rm: %s RPC timed out after %.1fs for key=%s; skipped for this tick. "
                    "(If PSRL_ELASTIC_RM_BACKLOG_DIAG=1, compare coordinator/router stages above.)",
                    op_label,
                    timeout_s,
                    key,
                )
                return TimeoutError(f"{op_label} key={key}")

        return await asyncio.gather(*[_one(r, k) for r, k in zip(refs, task_keys)], return_exceptions=True)

    async def _sync_engine_status_from_coordinators(self):
        task_refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            task_refs.append(coordinator.get_instance_engine_status_snapshot.remote())
            task_keys.append((role_name, model_name))

        if not task_refs:
            return
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            task_refs, task_keys, op_label="get_instance_engine_status_snapshot"
        )
        for (role_name, model_name), result in zip(task_keys, results):
            if isinstance(result, Exception):
                psrl_logger.warning(
                    "Failed to fetch engine status from coordinator role=%s model=%s: %s",
                    role_name,
                    model_name,
                    result,
                )
                continue
            for instance_id, snapshot in result.items():
                # instance_id is a RolloutInstanceId tuple (str, int) returned by coordinator
                self.instances_engine_stats[role_name][model_name][instance_id] = snapshot

    async def _sync_router_backlog_from_coordinators(self):
        refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            refs.append(coordinator.get_router_backlog_size.remote())
            task_keys.append((role_name, model_name))
        if not refs:
            self.router_backlog_by_role = {}
            return
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            refs,
            task_keys,
            op_label="get_router_backlog_size",
            trace_router_backlog=True,
        )
        prev_by_role = dict(self.router_backlog_by_role)
        grouped: dict[str, list] = defaultdict(list)
        for (role_name, model_name), result in zip(task_keys, results):
            role_key = getattr(role_name, "name", str(role_name))
            grouped[role_key].append(result)

        role_backlog: dict[str, int] = {}
        for role_key, result_list in grouped.items():
            total = 0
            for result in result_list:
                if isinstance(result, Exception):
                    psrl_logger.warning("Failed to fetch router backlog for role=%s: %s", role_key, result)
                    role_backlog[role_key] = prev_by_role.get(role_key, 0)
                    break
                total += max(0, int(result))
            else:
                role_backlog[role_key] = total
        self.router_backlog_by_role = role_backlog

    async def _sync_trainer_waiting_hint(self):
        if self.agent_loop_manager is None:
            self.trainer_waiting_hint = {
                "trainer_busy": True,
                "waiting_buffer_id": None,
                "waiting_on": "none",
                "breakdown": {},
            }
            return
        try:
            hint = await self.agent_loop_manager.get_trainer_waiting_hint.remote()
            if not isinstance(hint, dict):
                raise TypeError(f"trainer waiting hint must be dict, got {type(hint)}")
            self.trainer_waiting_hint = hint
        except Exception as exc:
            psrl_logger.warning("Failed to sync trainer waiting hint from agent_loop_manager: %s", exc)
            self.trainer_waiting_hint = {
                "trainer_busy": True,
                "waiting_buffer_id": None,
                "waiting_on": "none",
                "breakdown": {},
            }

    def get_instance_kv_cache_usage(
        self,
        instance_identifier: InstanceIdentifier,
    ) -> float:
        snapshot = (
            self.instances_engine_stats.get(instance_identifier.role, {})
            .get(instance_identifier.model_name, {})
            .get(instance_identifier.instance_id, {})
        )
        # TODO(linsh): why use 1.0 instead of 0.0 as the default?
        if not isinstance(snapshot, dict):
            return 1.0
        scheduler_stats = snapshot.get("scheduler_stats", {})
        return scheduler_stats.get("kv_cache_usage", 1.0)

    def get_instance_running_waiting(
        self,
        instance_identifier: InstanceIdentifier,
    ) -> tuple[int, int]:
        """Engine scheduler queue depth for elastic sleep gating (same keys as InstanceSignal)."""
        snapshot = (
            self.instances_engine_stats.get(instance_identifier.role, {})
            .get(instance_identifier.model_name, {})
            .get(instance_identifier.instance_id, {})
        )
        if not isinstance(snapshot, dict):
            return (0, 0)
        scheduler_stats = snapshot.get("scheduler_stats", {})
        return (
            int(scheduler_stats.get("num_running_reqs", 0)),
            int(scheduler_stats.get("num_waiting_reqs", 0)),
        )

    def _build_instance_signals(self) -> list[InstanceSignal]:
        signals: list[InstanceSignal] = []
        for role_name, role_data in self.instances_status_flags.items():
            for model_name, instance_status in role_data.items():
                for instance_id, status in instance_status.items():
                    snapshot = self.instances_engine_stats.get(role_name, {}).get(model_name, {}).get(instance_id, {})
                    scheduler_stats = snapshot.get("scheduler_stats", {})
                    gpu_keys = self.topology.get_instance_gpu_keys_frozenset(role_name, model_name, instance_id)
                    signal = InstanceSignal(
                        role_name=role_name,
                        model_name=model_name,
                        instance_id=instance_id,
                        is_awaken=status == InstanceStatus.AWAKEN,
                        kv_cache_utilization=float(scheduler_stats.get("kv_cache_usage", 0.0)),
                        running_queue_num=int(scheduler_stats.get("num_running_reqs", 0)),
                        waiting_queue_num=int(scheduler_stats.get("num_waiting_reqs", 0)),
                        generation_throughput=float(snapshot.get("generation_throughput", 0.0)),
                        total_token_num=int(
                            sum((scheduler_stats.get("req_id_to_prompt_token_num") or {}).values())
                            + sum((scheduler_stats.get("req_id_to_response_token_num") or {}).values())
                        ),
                        snapshot_timestamp=snapshot.get("timestamp"),
                        gpu_keys=gpu_keys,
                    )
                    signals.append(signal)
        return signals

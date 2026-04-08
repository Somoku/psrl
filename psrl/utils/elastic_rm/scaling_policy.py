import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psrl.utils.logger import FileOnlyHandler
from psrl.workers.gen_dplb.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _role_name_to_str(role_name: Any) -> str:
    if hasattr(role_name, "name"):
        return str(role_name.name)
    return str(role_name)


@dataclass
class InstanceSignal:
    role_name: Any
    model_name: str
    instance_id: RolloutInstanceId
    is_awaken: bool
    kv_cache_utilization: float
    running_queue_num: int
    waiting_queue_num: int
    generation_throughput: float
    total_token_num: int
    snapshot_timestamp: str | None = None
    # Set of (node_id, gpu_id) pairs occupied by this instance.
    # Populated by ElasticExecutor; None means mapping unavailable.
    gpu_keys: frozenset | None = None


@dataclass
class ScalingAction:
    action_type: str  # "scale_up" or "scale_down"
    role_name: Any
    model_name: str
    num_instances: int = 1
    preferred_instance_ids: list[RolloutInstanceId] | None = None
    reason: str = ""


@dataclass
class ScalingDecision:
    actions: list[ScalingAction]
    reason: str
    estimated_lambda: float
    role_to_total_mu: dict[str, float]


class ThroughputProfileLoader:
    """
    Loader for fitted throughput formulas in throughput_model/*_token.json.
    The formula is evaluated by running-queue length x:
        mu(x) = A * (1 - (B * x + 1)^(-k))
    """

    def __init__(
        self,
        profile_paths: dict[str, str] | None = None,
        throughput_model_dir: str | None = None,
        preferred_output_len: int = 1024,
    ):
        self.profile_paths = profile_paths or {}
        self.throughput_model_dir = throughput_model_dir
        self.preferred_output_len = int(preferred_output_len)
        self._profile_cache: dict[str, dict] = {}

    def _load_json(self, path: str) -> dict:
        if path in self._profile_cache:
            return self._profile_cache[path]
        if not path or not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self._profile_cache[path] = payload
        return payload

    def _load_throughput_formula_params(self, model_name: str) -> dict | None:
        if not self.throughput_model_dir:
            return None
        token_profile_path = os.path.join(self.throughput_model_dir, f"{model_name}_token.json")
        payload = self._load_json(token_profile_path)
        if not payload:
            return None
        if payload.get("metric") != "tok":
            return None
        fit_by_output_len = payload.get("fit_by_output_len", {})
        preferred_key = str(self.preferred_output_len)
        if preferred_key in fit_by_output_len:
            return fit_by_output_len[preferred_key]
        if fit_by_output_len:
            # Deterministic fallback: use smallest output-len bucket.
            sorted_keys = sorted(fit_by_output_len.keys(), key=lambda x: int(x))
            return fit_by_output_len[sorted_keys[0]]
        return None

    def estimate_mu_by_running_queue(
        self,
        model_name: str,
        running_queue_num: float,
        fallback_mu: float | None = None,
    ) -> float | None:
        running_queue_num = max(float(running_queue_num), 0.0)
        # Priority 1: throughput_model fitted formula (token throughput, output-len=1024).
        formula_params = self._load_throughput_formula_params(model_name)
        if formula_params is not None:
            required = {"A", "B", "k"}
            if required.issubset(set(formula_params.keys())):
                A = float(formula_params["A"])
                B = float(formula_params["B"])
                k = float(formula_params["k"])
                # Formula: A*(1-(B*x+1)^(-k)), x = running queue num.
                mu = A * (1.0 - math.pow(B * running_queue_num + 1.0, -k))
                return max(mu, 0.0)

        # Priority 2: fallback to provided scalar (typically runtime throughput).
        if fallback_mu is None:
            return None
        return max(float(fallback_mu), 0.0)

    @staticmethod
    def _resolve_new_schema_entry(payload: dict, signal: InstanceSignal, role_key: str) -> dict | None:
        role_section = payload.get("roles", {}).get(role_key, {})
        model_section = role_section.get(signal.model_name, {})
        if not model_section:
            return None
        # Optional per-instance override in new schema
        instance_section = model_section.get("instances", {}).get(str(signal.instance_id), {})
        if instance_section:
            return instance_section
        return model_section

    @staticmethod
    def _lookup_throughput_from_table(entry: dict, signal: InstanceSignal) -> float | None:
        # Priority: explicit scalar -> queue-based map -> threshold table.
        if "default_mu" in entry:
            return float(entry["default_mu"])

        running_queue_mu = entry.get("mu_by_running_queue", {})
        if running_queue_mu:
            key = str(signal.running_queue_num)
            if key in running_queue_mu:
                return float(running_queue_mu[key])
            # nearest lower key as a stable fallback
            numeric_keys = sorted(int(k) for k in running_queue_mu.keys() if str(k).isdigit())
            lower_keys = [k for k in numeric_keys if k <= signal.running_queue_num]
            if lower_keys:
                return float(running_queue_mu[str(lower_keys[-1])])

        mu_table = entry.get("mu_table", [])
        if not mu_table:
            return None
        for row in mu_table:
            max_q = int(row.get("max_running_queue_num", 10**9))
            max_tokens = int(row.get("max_total_token_num", 10**18))
            if signal.running_queue_num <= max_q and signal.total_token_num <= max_tokens:
                if "throughput" in row:
                    return float(row["throughput"])
                if "mu" in row:
                    return float(row["mu"])
        return None

    def estimate_instance_mu(self, signal: InstanceSignal) -> float:
        formula_mu = self.estimate_mu_by_running_queue(
            model_name=signal.model_name,
            running_queue_num=float(signal.running_queue_num),
            fallback_mu=None,
        )
        if formula_mu is not None:
            return formula_mu

        role_key = _role_name_to_str(signal.role_name)
        new_schema_path = self.profile_paths.get(signal.model_name)
        if new_schema_path:
            payload = self._load_json(new_schema_path)
            if payload:
                entry = self._resolve_new_schema_entry(payload, signal, role_key)
                if entry:
                    mu = self._lookup_throughput_from_table(entry, signal)
                    if mu is not None:
                        return max(0.0, float(mu))

        # Final fallback: trust runtime throughput if no formula/profile is found.
        return max(0.0, float(signal.generation_throughput))


class ScalingPolicy:
    def __init__(self, config: dict, policy_config: dict | None = None):
        self.config = config
        cfg = policy_config or {}
        self.enable = bool(cfg.get("enable_policy", cfg.get("enable", False)))
        self.monitor_interval_ms = int(cfg.get("monitor_interval_ms", 1000))
        self.theta_low = float(cfg.get("theta_low", 0.3))
        self.theta_max = float(cfg.get("theta_max", 0.85))
        self.cooldown_ms = int(cfg.get("cooldown_ms", 3000))
        self.hysteresis = float(cfg.get("hysteresis", 0.05))
        self.min_awake_per_role = max(0, int(cfg.get("min_awake_per_role", 0)))
        self.full_load_mode = str(cfg.get("full_load_mode", "any")).lower()
        # Extra guard for Priority-3 spontaneous shrink:
        # even if KV cache is low, do not shrink when waiting queue is still high.
        # This is computed per-role as total waiting queues across awake instances.
        self.max_waiting_queue_for_scale_down = int(cfg.get("max_waiting_queue_for_scale_down", 0))

        profile_paths = cfg.get("profile_paths", {})
        if not isinstance(profile_paths, dict):
            profile_paths = {}
        throughput_model_dir = cfg.get("throughput_model_dir")
        preferred_output_len = int(cfg.get("throughput_model_output_len", 1024))
        self.profile_loader = ThroughputProfileLoader(
            profile_paths=profile_paths,
            throughput_model_dir=throughput_model_dir,
            preferred_output_len=preferred_output_len,
        )

        self.last_action_time_ms: float = 0.0
        self._last_total_queue = 0
        self._last_lambda_time = time.time()
        self._lambda_ewma = 0.0
        self._lambda_ewma_alpha = float(cfg.get("lambda_ewma_alpha", 0.2))
        # Per-tick decision trace (reasons when scale conditions are not met).
        self.log_scaling_decisions = bool(cfg.get("log_scaling_decisions", True))

        self.log_prefix = "ScalingPolicy"
        psrl_logger.propagate = False
        psrl_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, self.log_prefix))

    def _policy_log(self, event: str, **kwargs: Any) -> None:
        if not self.log_scaling_decisions:
            return
        parts = [f"{k}={kwargs[k]!r}" for k in sorted(kwargs.keys())]
        psrl_logger.info("elastic_rm_policy %s | %s", event, " ".join(parts))

    def _policy_log_no_action(self, final_reason: str, diagnostics: list[str], detail: list[str]) -> None:
        if not self.log_scaling_decisions:
            return
        psrl_logger.info("elastic_rm_policy decision | outcome=no_action reason=%s", final_reason)
        if diagnostics:
            psrl_logger.info(
                "elastic_rm_policy no_action | diagnostics=%s",
                "|".join(sorted(diagnostics)),
            )
        for ln in detail:
            psrl_logger.info("elastic_rm_policy no_action_detail | %s", ln)

    @staticmethod
    def _is_snapshot_staled(snapshot: dict, max_staleness_seconds: float = 5.0) -> bool:
        ts = snapshot.get("timestamp")
        if not ts:
            return True
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return True
        return (datetime.now() - dt).total_seconds() > max_staleness_seconds

    @staticmethod
    def _get_role_name(role_name: Any) -> str:
        return _role_name_to_str(role_name)

    @staticmethod
    def _group_by_role(signals: list[InstanceSignal]) -> dict[str, list[InstanceSignal]]:
        grouped: dict[str, list[InstanceSignal]] = {}
        for signal in signals:
            grouped.setdefault(ScalingPolicy._get_role_name(signal.role_name), []).append(signal)
        return grouped

    def _estimate_lambda(self, signals: list[InstanceSignal], role_to_total_mu: dict[str, float]) -> float:
        now = time.time()
        elapsed = max(now - self._last_lambda_time, 1e-6)
        current_total_queue = sum(s.running_queue_num + s.waiting_queue_num for s in signals if s.is_awaken)
        d_queue = current_total_queue - self._last_total_queue
        total_mu = sum(role_to_total_mu.values())
        raw_lambda = max(total_mu + d_queue / elapsed, 0.0)
        self._lambda_ewma = self._lambda_ewma_alpha * raw_lambda + (1 - self._lambda_ewma_alpha) * self._lambda_ewma
        self._last_total_queue = current_total_queue
        self._last_lambda_time = now
        return self._lambda_ewma

    def _build_mu_maps(
        self, signals: list[InstanceSignal]
    ) -> tuple[dict[tuple[str, str, RolloutInstanceId], float], dict[str, float]]:
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float] = {}
        role_total_mu: dict[str, float] = {}
        for signal in signals:
            mu = self.profile_loader.estimate_instance_mu(signal)
            role_name = self._get_role_name(signal.role_name)
            key = (role_name, signal.model_name, signal.instance_id)
            instance_mu[key] = mu
            if signal.is_awaken:
                role_total_mu[role_name] = role_total_mu.get(role_name, 0.0) + mu
        return instance_mu, role_total_mu

    def _estimate_role_total_mu_with_rebalance(
        self,
        role_signals: list[InstanceSignal],
        scale_up_signal: InstanceSignal | None = None,
        scale_down_signal: InstanceSignal | None = None,
    ) -> float:
        # This helper simulates "after-action" throughput for one role.
        # We use it to compare candidate actions before issuing real SLEEP/WAKE_UP.
        awaken_signals = [s for s in role_signals if s.is_awaken]
        if not awaken_signals:
            return 0.0

        if scale_up_signal is not None:
            # Expansion: average (running + waiting) across all active instances.
            total_queue = sum(float(s.running_queue_num + s.waiting_queue_num) for s in awaken_signals)
            new_active = awaken_signals + [scale_up_signal]
            avg_running = total_queue / max(len(new_active), 1)
            return sum(
                self.profile_loader.estimate_mu_by_running_queue(
                    model_name=s.model_name,
                    running_queue_num=avg_running,
                    fallback_mu=s.generation_throughput,
                )
                or 0.0
                for s in new_active
            )

        if scale_down_signal is not None:
            # Shrink: redistribute removed running queue to retained active instances.
            retained = [s for s in awaken_signals if s.instance_id != scale_down_signal.instance_id]
            if len(retained) < self.min_awake_per_role:
                return -1.0
            extra_running = float(scale_down_signal.running_queue_num) / max(len(retained), 1)
            return sum(
                self.profile_loader.estimate_mu_by_running_queue(
                    model_name=s.model_name,
                    running_queue_num=float(s.running_queue_num) + extra_running,
                    fallback_mu=s.generation_throughput,
                )
                or 0.0
                for s in retained
            )

        return sum(
            self.profile_loader.estimate_mu_by_running_queue(
                model_name=s.model_name,
                running_queue_num=float(s.running_queue_num),
                fallback_mu=s.generation_throughput,
            )
            or 0.0
            for s in awaken_signals
        )

    def _pick_scale_down_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """Spontaneous shrink: only cede when KV is already low (below ``theta_low``).

        Used when a role has spare capacity (Priority 1/3), not for forced rebalance
        under mutual full load — see ``_pick_scale_down_candidate_for_bottleneck_transfer``.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        cede_candidates = [s for s in awaken if s.kv_cache_utilization <= self.theta_low]
        if not cede_candidates:
            return None
        cede_candidates.sort(
            key=lambda s: (
                s.kv_cache_utilization,
                instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            )
        )
        return cede_candidates[0]

    def _pick_scale_down_candidate_for_bottleneck_transfer(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """Pick which awake instance to sleep when *both* roles are full-load (Priority 2).

        Does **not** require ``kv_cache_utilization <= theta_low``: mutual full load is exactly
        when we may need to sacrifice a replica to wake the other role. We pick the instance
        with the smallest estimated per-instance throughput (mu) first, then lower KV as tie-break.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        awaken_sorted = sorted(
            awaken,
            key=lambda s: (
                instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
                s.kv_cache_utilization,
            ),
        )
        return awaken_sorted[0]

    def _pick_scale_up_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        asleep = [s for s in role_signals if not s.is_awaken]
        if not asleep:
            return None
        asleep.sort(
            key=lambda s: instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return asleep[0]

    def _pick_scale_up_candidate_on_free_gpu(
        self,
        full_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """Among asleep instances of the full-load role, return the best one whose
        GPUs are completely free of any awake instance from the other role.

        "Free" means: the candidate's gpu_keys have no intersection with the gpu_keys
        of any currently awake other-role instance.  Candidates with unknown GPU
        mapping (gpu_keys is None or empty) are skipped — we can't guarantee they are
        free, so we won't take the risk of double-occupancy.

        Returns the highest-mu free candidate, or None if no such instance exists.
        """
        other_awake_gpu_keys: set = set()
        for s in other_role_signals:
            if s.is_awaken and s.gpu_keys:
                other_awake_gpu_keys.update(s.gpu_keys)

        free_candidates: list[InstanceSignal] = []
        for s in full_role_signals:
            if s.is_awaken:
                continue
            if not s.gpu_keys:
                continue
            if not s.gpu_keys.intersection(other_awake_gpu_keys):
                free_candidates.append(s)

        if not free_candidates:
            return None
        free_candidates.sort(
            key=lambda s: instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return free_candidates[0]

    @staticmethod
    def _role_full_load(role_signals: list[InstanceSignal], theta_max: float, mode: str = "all") -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        if mode == "any":
            return any(s.kv_cache_utilization >= theta_max for s in awaken)
        return all(s.kv_cache_utilization >= theta_max for s in awaken)

    @staticmethod
    def _role_has_low_load(role_signals: list[InstanceSignal], theta_low: float) -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        return any(s.kv_cache_utilization <= theta_low for s in awaken)

    def _no_action_detail_strings(
        self,
        *,
        trainer_busy: bool,
        pending_total: int,
        waiting_on: str,
        rollout_full: bool,
        rm_full: bool,
        rollout_low: bool,
        rm_low: bool,
        rollout_down_waiting: int,
        rm_down_waiting: int,
        rollout_up: InstanceSignal | None,
        rm_up: InstanceSignal | None,
        rollout_down: InstanceSignal | None,
        rm_down: InstanceSignal | None,
        rollout_free_up: InstanceSignal | None,
        rm_free_up: InstanceSignal | None,
        rollout_down_xfer: InstanceSignal | None,
        rm_down_xfer: InstanceSignal | None,
        p2_gain_rejected: float | None,
        p2_branch_notes: list[str],
    ) -> list[str]:
        """Human-readable reasons why lower-priority branches did not scale (for logs)."""
        out: list[str] = []
        out.append(
            f"p-1_skip trainer_busy={trainer_busy} pending_total={pending_total} waiting_on={waiting_on!r} "
            f"(needs idle trainer + pending>0 + waiting_on rollout|reward)"
        )
        # Priority 1: RM-only-full branch
        if rm_full and not rollout_full:
            out.append(
                f"p1_rm_full_only rm_free_up={rm_free_up is not None} "
                f"rm_up={rm_up is not None} rollout_down_theta_low={rollout_down is not None} "
                f"(free_gpu tried first, fallback needs rm_up+rollout_down)"
            )
        # Priority 1: Rollout-only-full branch
        if rollout_full and not rm_full:
            out.append(
                f"p1_rollout_full_only rollout_free_up={rollout_free_up is not None} "
                f"rollout_up={rollout_up is not None} rm_down_theta_low={rm_down is not None} "
                f"(free_gpu tried first, fallback needs rollout_up+rm_down)"
            )
        if not rm_full and not rollout_full:
            out.append("p1_skip neither_side_full")
        # Priority 2
        out.append(
            f"p2_context both_full={rollout_full and rm_full} "
            f"rollout_down_xfer={rollout_down_xfer is not None} rm_down_xfer={rm_down_xfer is not None} "
            f"rm_up={rm_up is not None} rollout_up={rollout_up is not None}"
        )
        out.extend(p2_branch_notes)
        if p2_gain_rejected is not None:
            out.append(f"p2_reject best_gain={p2_gain_rejected:.6f} hysteresis={self.hysteresis}")
        out.append(
            f"p3_rollout_self_down_unmet rollout_low={rollout_low} "
            f"rollout_down_theta_low={rollout_down is not None} rm_full={rm_full} "
            f"down_waiting={rollout_down_waiting} waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        out.append(
            f"p3_rm_self_down_unmet rm_low={rm_low} rm_down_theta_low={rm_down is not None} "
            f"rollout_full={rollout_full} down_waiting={rm_down_waiting} "
            f"waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        return out

    def _make_stepwise_decision(
        self,
        grouped: dict[str, list[InstanceSignal]],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
        role_total_mu: dict[str, float],
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[list[ScalingAction], str]:
        rollout_role = "Rollout"
        rm_role = "RewardModel"
        rollout_signals = grouped.get(rollout_role, [])
        rm_signals = grouped.get(rm_role, [])
        if not rollout_signals or not rm_signals:
            self._policy_log(
                "decision",
                outcome="no_action",
                reason="skip_decision_missing_rollout_or_rm_signals",
                has_rm_signals=bool(rm_signals),
                has_rollout_signals=bool(rollout_signals),
            )
            return [], "skip_decision_missing_rollout_or_rm_signals"

        rollout_full = self._role_full_load(rollout_signals, self.theta_max, mode=self.full_load_mode)
        rm_full = self._role_full_load(rm_signals, self.theta_max, mode=self.full_load_mode)
        rollout_low = self._role_has_low_load(rollout_signals, self.theta_low)
        rm_low = self._role_has_low_load(rm_signals, self.theta_low)

        rollout_up = self._pick_scale_up_candidate(rollout_signals, instance_mu)
        rm_up = self._pick_scale_up_candidate(rm_signals, instance_mu)
        rollout_down = self._pick_scale_down_candidate(rollout_signals, instance_mu)
        rm_down = self._pick_scale_down_candidate(rm_signals, instance_mu)

        actions: list[ScalingAction] = []
        diagnostics: list[str] = []
        rollout_down_waiting = rollout_down.waiting_queue_num if rollout_down is not None else -1
        rm_down_waiting = rm_down.waiting_queue_num if rm_down is not None else -1
        n_rollout_awaken = sum(1 for s in rollout_signals if s.is_awaken)
        n_rollout_asleep = sum(1 for s in rollout_signals if not s.is_awaken)
        n_rm_awaken = sum(1 for s in rm_signals if s.is_awaken)
        n_rm_asleep = sum(1 for s in rm_signals if not s.is_awaken)
        self._policy_log(
            "tick_context",
            full_load_mode=self.full_load_mode,
            n_rm_asleep=n_rm_asleep,
            n_rm_awaken=n_rm_awaken,
            n_rollout_asleep=n_rollout_asleep,
            n_rollout_awaken=n_rollout_awaken,
            rm_full=rm_full,
            rm_low=rm_low,
            rollout_full=rollout_full,
            rollout_low=rollout_low,
            theta_low=self.theta_low,
            theta_max=self.theta_max,
        )
        self._policy_log(
            "tick_context_waiting_guard",
            max_waiting_queue_for_scale_down=self.max_waiting_queue_for_scale_down,
            rm_down_waiting=rm_down_waiting,
            rollout_down_waiting=rollout_down_waiting,
        )

        # Priority -1: keep trainer continuously training.
        # When trainer is idle and blocked by current batch, directly bias scale-up
        # towards the bottleneck stage (rollout/reward).
        hint = trainer_waiting_hint or {}
        trainer_busy = bool(hint.get("trainer_busy", True))
        waiting_on = str(hint.get("waiting_on", "none")).lower()
        pending_total = int((hint.get("breakdown") or {}).get("pending_total", 0))
        if not trainer_busy and pending_total > 0 and waiting_on in {"rollout", "reward"}:
            if waiting_on == "rollout":
                if rollout_up is None:
                    self._policy_log(
                        "decision",
                        outcome="no_action",
                        reason="trainer_idle_waiting_rollout_but_no_scaleup_candidate",
                        pending_total=pending_total,
                        waiting_on=waiting_on,
                    )
                    return [], "trainer_idle_waiting_rollout_but_no_scaleup_candidate"
                actions.append(
                    ScalingAction(
                        action_type="scale_up",
                        role_name=rollout_up.role_name,
                        model_name=rollout_up.model_name,
                        preferred_instance_ids=[rollout_up.instance_id],
                        reason="trainer_idle_waiting_rollout_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    instance_id=rollout_up.instance_id,
                    model_name=rollout_up.model_name,
                    outcome="action",
                    policy_branch="p-1_trainer_idle_rollout",
                    reason="trainer_idle_waiting_rollout",
                )
                return actions, "trainer_idle_waiting_rollout"
            if rm_up is None:
                self._policy_log(
                    "decision",
                    outcome="no_action",
                    reason="trainer_idle_waiting_reward_but_no_scaleup_candidate",
                    pending_total=pending_total,
                    waiting_on=waiting_on,
                )
                return [], "trainer_idle_waiting_reward_but_no_scaleup_candidate"
            actions.append(
                ScalingAction(
                    action_type="scale_up",
                    role_name=rm_up.role_name,
                    model_name=rm_up.model_name,
                    preferred_instance_ids=[rm_up.instance_id],
                    reason="trainer_idle_waiting_reward_scale_up",
                )
            )
            self._policy_log(
                "decision",
                action="scale_up",
                instance_id=rm_up.instance_id,
                model_name=rm_up.model_name,
                outcome="action",
                policy_branch="p-1_trainer_idle_reward",
                reason="trainer_idle_waiting_reward",
            )
            return actions, "trainer_idle_waiting_reward"

        # ── Priority 1 (new) ──────────────────────────────────────────────────────
        # Applies when EXACTLY one side is full.
        # Step A: try to wake an instance of the full side on a GPU that is
        #         completely idle for the other side (no scale_down needed).
        # Step B: fall back to ceding a low-load instance from the other side.
        # Both-full case is handled by Priority 2 below.
        # ─────────────────────────────────────────────────────────────────────────

        # Pre-compute free-GPU candidates (only needed when exactly one side full).
        rm_free_up: InstanceSignal | None = None
        rollout_free_up: InstanceSignal | None = None
        if rm_full and not rollout_full:
            rm_free_up = self._pick_scale_up_candidate_on_free_gpu(rm_signals, rollout_signals, instance_mu)
        if rollout_full and not rm_full:
            rollout_free_up = self._pick_scale_up_candidate_on_free_gpu(rollout_signals, rm_signals, instance_mu)

        # P1a: only RM is full
        if rm_full and not rollout_full:
            # Step A: free-GPU scale-up — no resource taken from Rollout
            if rm_free_up is not None:
                actions.append(
                    ScalingAction(
                        action_type="scale_up",
                        role_name=rm_free_up.role_name,
                        model_name=rm_free_up.model_name,
                        preferred_instance_ids=[rm_free_up.instance_id],
                        reason="rm_full_free_gpu_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=True,
                    instance_id=rm_free_up.instance_id,
                    model_name=rm_free_up.model_name,
                    outcome="action",
                    policy_branch="p1_rm_full_free_gpu",
                    reason="rm_full_free_gpu_scale_up",
                )
                return actions, "rm_full_free_gpu_scale_up"
            # Step B: fall back — cede a low-load Rollout instance
            if rm_up is not None and rollout_down is not None:
                actions.append(
                    ScalingAction(
                        action_type="scale_up",
                        role_name=rm_up.role_name,
                        model_name=rm_up.model_name,
                        preferred_instance_ids=[rm_up.instance_id],
                        reason="rm_full_rollout_cede_transfer",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=False,
                    instance_id=rm_up.instance_id,
                    model_name=rm_up.model_name,
                    outcome="action",
                    policy_branch="p1_transfer_rollout_to_rm",
                    reason="transfer_rollout_to_rm",
                )
                return actions, "transfer_rollout_to_rm"

        # P1b: only Rollout is full
        if rollout_full and not rm_full:
            # Step A: free-GPU scale-up — no resource taken from RM
            if rollout_free_up is not None:
                actions.append(
                    ScalingAction(
                        action_type="scale_up",
                        role_name=rollout_free_up.role_name,
                        model_name=rollout_free_up.model_name,
                        preferred_instance_ids=[rollout_free_up.instance_id],
                        reason="rollout_full_free_gpu_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=True,
                    instance_id=rollout_free_up.instance_id,
                    model_name=rollout_free_up.model_name,
                    outcome="action",
                    policy_branch="p1_rollout_full_free_gpu",
                    reason="rollout_full_free_gpu_scale_up",
                )
                return actions, "rollout_full_free_gpu_scale_up"
            # Step B: fall back — cede a low-load RM instance
            if rollout_up is not None and rm_down is not None:
                actions.append(
                    ScalingAction(
                        action_type="scale_up",
                        role_name=rollout_up.role_name,
                        model_name=rollout_up.model_name,
                        preferred_instance_ids=[rollout_up.instance_id],
                        reason="rollout_full_rm_cede_transfer",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=False,
                    instance_id=rollout_up.instance_id,
                    model_name=rollout_up.model_name,
                    outcome="action",
                    policy_branch="p1_transfer_rm_to_rollout",
                    reason="transfer_rm_to_rollout",
                )
                return actions, "transfer_rm_to_rollout"

        # ── Priority 2 ────────────────────────────────────────────────────────────
        # Both sides full → optimize bottleneck throughput by one-step transfer.
        # We compare candidate gains with queue-rebalance simulation instead of naive
        # add/subtract current mu, because queue pressure changes after scaling.
        # Sleep-side candidates ignore theta_low: full KV on both sides is normal
        # here; theta_low remains the gate for *spontaneous* shrink in Priority 3.
        # ─────────────────────────────────────────────────────────────────────────
        bottleneck_before = min(
            self._estimate_role_total_mu_with_rebalance(rollout_signals),
            self._estimate_role_total_mu_with_rebalance(rm_signals),
        )
        transfer_candidates: list[tuple[float, ScalingAction, str]] = []
        rollout_down_xfer: InstanceSignal | None = None
        rm_down_xfer: InstanceSignal | None = None
        p2_branch_notes: list[str] = []
        p2_gain_rejected: float | None = None
        if rollout_full and rm_full:
            rollout_down_xfer = self._pick_scale_down_candidate_for_bottleneck_transfer(rollout_signals, instance_mu)
            rm_down_xfer = self._pick_scale_down_candidate_for_bottleneck_transfer(rm_signals, instance_mu)

        if rollout_down_xfer is not None and rm_up is not None:
            rollout_mu = self._estimate_role_total_mu_with_rebalance(
                rollout_signals,
                scale_down_signal=rollout_down_xfer,
            )
            rm_mu = self._estimate_role_total_mu_with_rebalance(
                rm_signals,
                scale_up_signal=rm_up,
            )
            if rollout_mu >= 0 and rm_mu >= 0:
                bottleneck_after = min(rollout_mu, rm_mu)
                transfer_candidates.append(
                    (
                        bottleneck_after - bottleneck_before,
                        ScalingAction(
                            action_type="scale_up",
                            role_name=rm_up.role_name,
                            model_name=rm_up.model_name,
                            preferred_instance_ids=[rm_up.instance_id],
                            reason="optimize_bottleneck_rollout_to_rm",
                        ),
                        "optimize_rollout_to_rm",
                    )
                )
            else:
                p2_branch_notes.append(f"p2_opt_rollout_to_rm_sim_invalid rollout_mu={rollout_mu} rm_mu={rm_mu}")
        elif rollout_full and rm_full:
            if rollout_down_xfer is None:
                p2_branch_notes.append("p2_opt_rollout_to_rm_skip_no_rollout_down_xfer")
            if rm_up is None:
                p2_branch_notes.append("p2_opt_rollout_to_rm_skip_no_rm_up")

        if rm_down_xfer is not None and rollout_up is not None:
            rollout_mu = self._estimate_role_total_mu_with_rebalance(
                rollout_signals,
                scale_up_signal=rollout_up,
            )
            rm_mu = self._estimate_role_total_mu_with_rebalance(
                rm_signals,
                scale_down_signal=rm_down_xfer,
            )
            if rollout_mu >= 0 and rm_mu >= 0:
                bottleneck_after = min(rollout_mu, rm_mu)
                transfer_candidates.append(
                    (
                        bottleneck_after - bottleneck_before,
                        ScalingAction(
                            action_type="scale_up",
                            role_name=rollout_up.role_name,
                            model_name=rollout_up.model_name,
                            preferred_instance_ids=[rollout_up.instance_id],
                            reason="optimize_bottleneck_rm_to_rollout",
                        ),
                        "optimize_rm_to_rollout",
                    )
                )
            else:
                p2_branch_notes.append(f"p2_opt_rm_to_rollout_sim_invalid rollout_mu={rollout_mu} rm_mu={rm_mu}")
        elif rollout_full and rm_full:
            if rm_down_xfer is None:
                p2_branch_notes.append("p2_opt_rm_to_rollout_skip_no_rm_down_xfer")
            if rollout_up is None:
                p2_branch_notes.append("p2_opt_rm_to_rollout_skip_no_rollout_up")

        if transfer_candidates:
            transfer_candidates.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_action, reason = transfer_candidates[0]
            if best_gain > self.hysteresis:
                actions.append(best_action)
                self._policy_log(
                    "decision",
                    action=best_action.action_type,
                    bottleneck_before=bottleneck_before,
                    gain=best_gain,
                    instance_id=(best_action.preferred_instance_ids or [None])[0],
                    model_name=best_action.model_name,
                    outcome="action",
                    policy_branch="p2_bottleneck_transfer",
                    reason=f"{reason}_gain_{best_gain:.6f}",
                )
                return actions, f"{reason}_gain_{best_gain:.6f}"
            p2_gain_rejected = best_gain

        # Priority 3: purely underloaded side can scale down itself.
        if (
            rollout_low
            and rollout_down is not None
            and not rm_full
            and rollout_down.waiting_queue_num <= self.max_waiting_queue_for_scale_down
        ):
            actions.append(
                ScalingAction(
                    action_type="scale_down",
                    role_name=rollout_down.role_name,
                    model_name=rollout_down.model_name,
                    preferred_instance_ids=[rollout_down.instance_id],
                    reason="rollout_low_scale_down",
                )
            )
            self._policy_log(
                "decision",
                action="scale_down",
                instance_id=rollout_down.instance_id,
                model_name=rollout_down.model_name,
                outcome="action",
                policy_branch="p3_rollout_self_down",
                reason="rollout_low_scale_down",
            )
            return actions, "rollout_low_scale_down"
        if (
            rm_low
            and rm_down is not None
            and not rollout_full
            and rm_down.waiting_queue_num <= self.max_waiting_queue_for_scale_down
        ):
            actions.append(
                ScalingAction(
                    action_type="scale_down",
                    role_name=rm_down.role_name,
                    model_name=rm_down.model_name,
                    preferred_instance_ids=[rm_down.instance_id],
                    reason="rm_low_scale_down",
                )
            )
            self._policy_log(
                "decision",
                action="scale_down",
                instance_id=rm_down.instance_id,
                model_name=rm_down.model_name,
                outcome="action",
                policy_branch="p3_rm_self_down",
                reason="rm_low_scale_down",
            )
            return actions, "rm_low_scale_down"

        detail = self._no_action_detail_strings(
            trainer_busy=trainer_busy,
            pending_total=pending_total,
            waiting_on=waiting_on,
            rollout_full=rollout_full,
            rm_full=rm_full,
            rollout_low=rollout_low,
            rm_low=rm_low,
            rollout_down_waiting=rollout_down_waiting,
            rm_down_waiting=rm_down_waiting,
            rollout_up=rollout_up,
            rm_up=rm_up,
            rollout_down=rollout_down,
            rm_down=rm_down,
            rollout_free_up=rollout_free_up,
            rm_free_up=rm_free_up,
            rollout_down_xfer=rollout_down_xfer,
            rm_down_xfer=rm_down_xfer,
            p2_gain_rejected=p2_gain_rejected,
            p2_branch_notes=p2_branch_notes,
        )
        if diagnostics:
            diagnostics.sort()
            merged = "|".join(diagnostics + detail)
            final_reason = f"no_action_{merged}"[:4096]
            self._policy_log_no_action(final_reason, diagnostics, detail)
            return [], final_reason
        merged_detail = " ;; ".join(detail)
        final_reason = f"no_action_{merged_detail}"[:4096]
        self._policy_log_no_action(final_reason, [], detail)
        return [], final_reason

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[str, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> ScalingDecision:
        if not self.enable:
            self._policy_log("decision", outcome="skipped", reason="policy_disabled")
            return ScalingDecision(actions=[], reason="policy_disabled", estimated_lambda=0.0, role_to_total_mu={})
        if not signals:
            self._policy_log("decision", outcome="skipped", reason="empty_signals")
            return ScalingDecision(actions=[], reason="empty_signals", estimated_lambda=0.0, role_to_total_mu={})
        if execution_in_progress:
            self._policy_log("decision", outcome="skipped", reason="decision_execution_in_progress")
            return ScalingDecision(
                actions=[],
                reason="decision_execution_in_progress",
                estimated_lambda=0.0,
                role_to_total_mu={},
            )

        now_ms = time.time() * 1000
        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            remain = self.cooldown_ms - (now_ms - self.last_action_time_ms)
            self._policy_log(
                "decision",
                cooldown_remaining_ms=remain,
                outcome="skipped",
                reason="cooldown",
            )
            return ScalingDecision(actions=[], reason="cooldown", estimated_lambda=0.0, role_to_total_mu={})

        grouped = self._group_by_role(signals)
        instance_mu, role_total_mu = self._build_mu_maps(signals)
        estimated_lambda = self._estimate_lambda(signals, role_total_mu)

        # Hard guarantee: if one role has zero awaken instances but router backlog exists,
        # force wake one instance for that role. Must run *before* the all-signals-stale guard:
        # asleep engines often have no fresh snapshot timestamps, which would otherwise skip this.
        backlog_map = router_backlog_by_role or {}
        for role_name, role_signals in grouped.items():
            awaken_cnt = sum(1 for s in role_signals if s.is_awaken)
            backlog_cnt = int(backlog_map.get(role_name, 0))
            if awaken_cnt > 0 or backlog_cnt <= 0:
                continue
            force_up = self._pick_scale_up_candidate(role_signals, instance_mu)
            if force_up is None:
                r = f"force_wake_needed_but_no_asleep_candidate_{role_name}_backlog_{backlog_cnt}"
                self._policy_log(
                    "decision",
                    backlog_cnt=backlog_cnt,
                    outcome="no_action",
                    reason=r,
                    role_name=role_name,
                )
                return ScalingDecision(
                    actions=[],
                    reason=r,
                    estimated_lambda=estimated_lambda,
                    role_to_total_mu=role_total_mu,
                )
            self.last_action_time_ms = now_ms
            fw_reason = f"force_wake_{role_name}_backlog_{backlog_cnt}"
            self._policy_log(
                "decision",
                action="scale_up",
                backlog_cnt=backlog_cnt,
                instance_id=force_up.instance_id,
                model_name=force_up.model_name,
                outcome="action",
                policy_branch="force_wake_backlog",
                reason=fw_reason,
                role_name=role_name,
            )
            return ScalingDecision(
                actions=[
                    ScalingAction(
                        action_type="scale_up",
                        role_name=force_up.role_name,
                        model_name=force_up.model_name,
                        preferred_instance_ids=[force_up.instance_id],
                        reason=f"force_wake_from_router_backlog_{role_name}_{backlog_cnt}",
                    )
                ],
                reason=fw_reason,
                estimated_lambda=estimated_lambda,
                role_to_total_mu=role_total_mu,
            )

        # Safety guard: skip aggressive scale decisions when snapshots are stale.
        stale_count = 0
        for signal in signals:
            snapshot = {"timestamp": signal.snapshot_timestamp}
            if self._is_snapshot_staled(snapshot):
                stale_count += 1
        if stale_count == len(signals):
            self._policy_log(
                "decision",
                outcome="skipped",
                reason="all_signals_stale",
                stale_count=stale_count,
            )
            return ScalingDecision(actions=[], reason="all_signals_stale", estimated_lambda=0.0, role_to_total_mu={})
        actions, reason = self._make_stepwise_decision(
            grouped,
            instance_mu,
            role_total_mu,
            trainer_waiting_hint=trainer_waiting_hint,
        )

        if actions:
            self.last_action_time_ms = now_ms
        self._policy_log(
            "decision_summary",
            actions_count=len(actions),
            estimated_lambda=estimated_lambda,
            reason=reason,
            role_to_total_mu=dict(role_total_mu),
        )
        return ScalingDecision(
            actions=actions,
            reason=reason,
            estimated_lambda=estimated_lambda,
            role_to_total_mu=role_total_mu,
        )

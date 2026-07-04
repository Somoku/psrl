"""Session hang/continue scheduler (ThunderAgent port).

Pure, side-effect-free capacity logic. Given a snapshot of per-instance KV-cache
capacity and the set of live TITO sessions (each pinned to one vLLM instance),
decide which sessions to **hang** (evict from an over-capacity instance) and which
hung sessions to **continue** (readmit once their pinned instance has room).

This mirrors ThunderAgent's ``_pause_until_safe`` / ``_greedy_resume`` and its
per-backend token capacity model, adapted to psrl_smg terminology:

- ThunderAgent "program"  -> psrl_smg TITO "session"
- ThunderAgent REASONING  -> "generate" (a trajectory is inferring on vLLM/SMG)
- ThunderAgent ACTING     -> "env"      (a trajectory is calling the environment)
- ThunderAgent pause/resume -> hang/continue

The module imports nothing from Ray/HTTP so it can be unit-tested in isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os

from .base import (
    SESSION_HUNG,
    STATUS_ENV,
    InstanceCapacity,
    SessionInfo,
    SessionScheduler,
    SessionSchedulingBase,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class ThunderAgentScheduler(SessionScheduler):
    """Decide session hang/continue from instance capacity + session footprints.

    Args:
        env_token_weight: Coefficient applied to env-status session tokens in
            ``capacity_used`` (ThunderAgent's ``tool_coefficient``). Lower values
            assume an env-status session's KV cache is likely evicted before it
            returns, so it counts for less.
        buffer_per_session: Decode headroom (tokens) reserved per running session.
    """

    def __init__(self, env_token_weight: float = 1.0, buffer_per_session: int = 100):
        self.env_token_weight = float(env_token_weight)
        self.buffer_per_session = int(buffer_per_session)

    def decide(
        self,
        instances: list[InstanceCapacity],
        sessions: list[SessionInfo],
    ) -> tuple[list[str], list[str]]:
        """Return ``(session_ids_to_hang, session_ids_to_continue)``.

        A session is only ever hung from / continued on the instance it is pinned
        to (sessions are sticky-routed), so both decisions are made per instance.
        """
        instance_by_id: dict = {inst.instance_id: inst for inst in instances}

        # Group sessions by their pinned instance. Sessions with no pinning yet
        # (instance_id is None) or pinned to an unknown instance are ignored.
        running_by_instance: dict = {}
        hung_by_instance: dict = {}
        for sess in sessions:
            if sess.instance_id is None or sess.instance_id not in instance_by_id:
                continue
            if sess.hang_state == SESSION_HUNG:
                hung_by_instance.setdefault(sess.instance_id, []).append(sess)
            else:
                running_by_instance.setdefault(sess.instance_id, []).append(sess)

        to_hang: list[str] = []
        to_continue: list[str] = []

        for inst in instances:
            running = running_by_instance.get(inst.instance_id, [])
            hung = hung_by_instance.get(inst.instance_id, [])

            # Sessions we plan to hang this tick free up their footprint.
            hung_now: set[str] = set()
            remaining = self._remaining_capacity(inst, running, hung_now)

            # --- Hang phase: shed running sessions until within capacity. ---
            # Priority: env-status first (off-GPU, cheapest to evict), then
            # generate-status; smallest tokens first within each group.
            if remaining < 0:
                hang_order = self._hang_order(running)
                for sess in hang_order:
                    if remaining >= 0:
                        break
                    to_hang.append(sess.session_id)
                    hung_now.add(sess.session_id)
                    remaining = self._remaining_capacity(inst, running, hung_now)

            # --- Continue phase: BFD readmit hung sessions that now fit. ---
            # Only sessions pinned to THIS instance are candidates. Smallest
            # first so we readmit as many as possible.
            if remaining > self.buffer_per_session and hung:
                for sess in sorted(hung, key=lambda s: s.total_tokens):
                    need = sess.total_tokens + self.buffer_per_session
                    if need <= remaining:
                        to_continue.append(sess.session_id)
                        remaining -= need

        if to_hang or to_continue:
            psrl_logger.info(
                "ThunderAgentScheduler decide: hang=%s continue=%s",
                to_hang,
                to_continue,
            )
        return to_hang, to_continue

    def _remaining_capacity(
        self,
        inst: InstanceCapacity,
        running: list[SessionInfo],
        hung_now: set[str],
    ) -> int:
        """Remaining KV-token capacity on ``inst`` given the running set.

        Sessions in ``hung_now`` are treated as already evicted (their footprint
        and buffer removed). Mirrors ThunderAgent's ``remaining_capacity``:

            capacity_used = generate_tokens + env_token_weight * env_tokens + buffer
            buffer        = buffer_per_session * running_session_count
            remaining     = total_kv_tokens - capacity_used

        The engine's measured ``used_tokens`` already reflects prefix-cache
        sharing, so we take ``min(attributed, measured)`` to avoid
        double-counting shared prefixes (ThunderAgent's ``shared_tokens`` role).
        """
        generate_tokens = 0
        env_tokens = 0
        count = 0
        for sess in running:
            if sess.session_id in hung_now:
                continue
            count += 1
            if sess.status == STATUS_ENV:
                env_tokens += sess.total_tokens
            else:
                generate_tokens += sess.total_tokens

        attributed = int(generate_tokens + self.env_token_weight * env_tokens)
        # Cross-check against the engine's real resident-token count.
        effective_used = min(attributed, inst.used_tokens) if inst.used_tokens > 0 else attributed
        buffer = count * self.buffer_per_session
        return inst.total_kv_tokens - (effective_used + buffer)

    @staticmethod
    def _hang_order(running: list[SessionInfo]) -> list[SessionInfo]:
        """Eviction order: env-status before generate-status, smallest tokens first."""
        env = sorted(
            (s for s in running if s.status == STATUS_ENV),
            key=lambda s: s.total_tokens,
        )
        generate = sorted(
            (s for s in running if s.status != STATUS_ENV),
            key=lambda s: s.total_tokens,
        )
        return env + generate


class ThunderAgentSessionMixin(SessionSchedulingBase):
    """Mixin providing the periodic thunder_agent session hang/continue loop.

    Inherits the generic session-router HTTP helpers from ``SessionSchedulingBase``.
    Expects ``self`` to have the following attributes (initialized by RolloutCoordinator):
      self._thunder_agent_cfg
      self.stop_thunder_agent
      self.instance_ids
      self.instance_to_total_kv_tokens
      self.tag_to_replica_ids
      self.server_handles
      self.instance_to_engine_status
      self.session_router_url
      self._session_client
    """

    async def _refresh_kv_token_capacities(self) -> None:
        """Fetch absolute KV-token capacity (estimate_max_model_len) for rollout
        instances that don't have it cached yet."""
        missing_replicas = {
            instance_id[0]
            for instance_id in self.instance_ids
            if instance_id not in self.instance_to_total_kv_tokens
            and instance_id[0] in self.tag_to_replica_ids["rollout"]
        }
        for replica_id in missing_replicas:
            max_len = int(await self.server_handles[replica_id].estimate_max_model_len.remote())
            for instance_id in self.instance_ids:
                if instance_id[0] == replica_id:
                    self.instance_to_total_kv_tokens[instance_id] = max_len

    def _build_instance_capacities(self):
        """Snapshot per-instance KV capacity for the scheduler (rollout instances only)."""
        capacities = []
        for instance_id, engine_status in self.instance_to_engine_status.items():
            if instance_id[0] not in self.tag_to_replica_ids["rollout"]:
                continue
            total = self.instance_to_total_kv_tokens.get(instance_id)
            if not total:
                # Capacity not resolved yet; skip until estimate_max_model_len lands.
                continue
            used = int(round(engine_status.get_kv_cache_utilization() * total))
            capacities.append(
                InstanceCapacity(instance_id=instance_id, total_kv_tokens=total, used_tokens=used)
            )
        return capacities

    @staticmethod
    def _parse_session_infos(payload):
        """Convert /control/sessions JSON into SessionInfo objects."""
        sessions = []
        for item in payload.get("sessions", []):
            # A session that is closing should not be scheduled.
            if item.get("closing"):
                continue
            base_worker_id = item.get("base_worker_id")
            target_dp_rank = item.get("target_dp_rank")
            instance_id = None
            if base_worker_id is not None and target_dp_rank is not None:
                instance_id = (str(base_worker_id), int(target_dp_rank))
            sessions.append(
                SessionInfo(
                    session_id=str(item["session_id"]),
                    instance_id=instance_id,
                    status=str(item.get("status", "env")),
                    hang_state=str(item.get("hang_state", "running")),
                    total_tokens=int(item.get("total_tokens", 0)),
                )
            )
        return sessions

    async def _thunder_agent_loop(self):
        """Periodically hang/continue sessions to keep instances within KV capacity."""
        cfg = self._thunder_agent_cfg
        self._thunder_scheduler = ThunderAgentScheduler(
            env_token_weight=float(cfg.get("env_token_weight", 1.0)),
            buffer_per_session=int(cfg.get("buffer_per_session", 100)),
        )
        interval = int(cfg.get("check_interval_in_ms", 1000)) / 1000.0
        psrl_logger.info("Starting thunder_agent (session hang/continue) loop, interval=%.3fs", interval)

        while not self.stop_thunder_agent:
            await asyncio.sleep(interval)
            if self.stop_thunder_agent:
                break
            await self._refresh_kv_token_capacities()
            payload = await self._session_get_json("/control/sessions")
            sessions = self._parse_session_infos(payload)
            if not sessions:
                continue
            instances = self._build_instance_capacities()
            to_hang, to_continue = self._thunder_scheduler.decide(instances, sessions)
            if to_hang:
                await self._session_post_json("/control/hang", [{"session_id": sid} for sid in to_hang])
            if to_continue:
                await self._session_post_json(
                    "/control/continue", [{"session_id": sid} for sid in to_continue]
                )
        psrl_logger.info("Stopped thunder_agent loop.")

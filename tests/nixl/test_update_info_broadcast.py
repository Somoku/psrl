"""The meta server's rebroadcast of updated client infos (item E3 envelope).

Both directions of a colocate switch end in this broadcast: the sender publishes new
addresses, the meta server fans them out, and every client waits for exactly one
tagged ``UPDATE_INFO`` before it will read from the new addresses.

Findings from a two-node run that hung in ``validate_publish``:

* The server sent a bare ``pickle.dumps(dict)`` while the reader routes on the tag, so
  ``decode`` returned ``None`` and every waiting client filed the payload as foreign.
  Nothing logged on either side, and the run sat until the handshake budget expired.
* The rank count per instance came from ``worker_to_ps_idx``, which is filled in only
  where PS storage workers exist. Under direct supply it is empty, so the fan-out named
  no destination for the rollout and validate instances while they still waited.

Both are silent, so both are pinned here against the server and the trainer.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.handshake_notif import NotifKind, decode

pytestmark = pytest.mark.cpu_test


class _StubAgent:
    """Records what the server hands to `send_notif` instead of sending it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []

    def send_notif(self, agent_name: str, payload: bytes) -> None:
        self.sent.append((agent_name, payload))


class _StubInfo:
    def serialize(self) -> bytes:
        return b"info-bytes"


def _server(update_client_names: list[str]) -> tuple:
    from psrl.utils.nixl.server import NIXLMetaServer

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.server_name = "NIXLMetaServer"
    server.agent = _StubAgent()
    server.connected_clients = {name: [name] for name in ("a", "b")}
    server.client_infos = {name: _StubInfo() for name in update_client_names}
    server._client_temp_mappings = {name: {} for name in update_client_names}
    return server, server.agent


class TestTheBroadcastIsTagged:
    def test_the_payload_carries_the_update_info_kind(self):
        """An untagged payload decodes to `None`, which the reader silently discards."""
        server, agent = _server(["NIXLGenClient_I8_R0"])
        server.broadcast_update_client_infos(["a", "b"], ["NIXLGenClient_I8_R0"])

        assert agent.sent, "nothing was broadcast"
        parsed = decode(agent.sent[0][1])

        assert parsed is not None
        assert parsed.kind is NotifKind.UPDATE_INFO

    def test_every_destination_gets_the_same_tagged_payload(self):
        """One destination silently skipped is one client that waits out the budget."""
        server, agent = _server(["c0", "c1"])
        server.broadcast_update_client_infos(["a", "b"] * 3, ["c0", "c1"])

        assert {name for name, _ in agent.sent} == {"a", "b"}
        assert all(decode(payload) is not None for _, payload in agent.sent)

    def test_entries_carry_info_and_temp_mapping_per_client(self):
        """The shape the reader unpacks; a missing key fails on the receiving side."""
        server, agent = _server(["c0"])
        server.broadcast_update_client_infos(["a"], ["c0"])

        entries = decode(agent.sent[0][1]).entries

        assert set(entries["c0"]) == {"info", "temp_mapping"}


class TestRankCountsDoNotComeFromThePsMap:
    def test_counts_come_from_the_worker_groups(self):
        """`worker_to_ps_idx` is empty under direct supply, where there are no PS workers."""

        class _Group:
            def __init__(self, world_size: int) -> None:
                self.world_size = world_size

        class _Trainer:
            rollout_wg_list = [_Group(1), _Group(2), _Group(4)]
            validate_wg_list = [_Group(8)]

            def _ranks_per_instance(self, role):  # pragma: no cover - replaced below
                raise AssertionError("the real implementation must be used")

        from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

        trainer = _Trainer()

        assert PSRL_RayPPOTrainer._ranks_per_instance(trainer, "rollout") == [1, 2, 4]
        assert PSRL_RayPPOTrainer._ranks_per_instance(trainer, "validate") == [8]

    def test_the_trainer_no_longer_derives_counts_from_the_ps_map(self):
        """Only the PS placement block may mention it; the docstring names it on purpose."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")
        at = source.index("def _ranks_per_instance")
        block = source[at : at + 1200]
        body = block.split('"""', 2)[2]

        assert "worker_to_ps_idx" not in body


class TestValidationMetricsSurviveTheDumpPayload:
    def test_a_dict_valued_series_does_not_reach_the_averager(self):
        """`reward_extra_info` is one dict per sample for the generation dump.

        verl turns every key it is given into a metric series, so handing it that key
        raises `TypeError: unsupported operand type(s) for /: 'dict' and 'int'` from
        `np.mean` -- which killed every validation that reached the metrics step.
        """
        trainer = _FakeTrainer()

        metrics = trainer._val_metrics_update(
            ["math_dapo", "math_dapo"],
            ["uid0", "uid1"],
            {
                "reward": [1.0, 0.0],
                "acc": [1.0, 0.0],
                "reward_extra_info": [{"acc": 1.0}, {"acc": 0.0}],
            },
            [1, 1],
        )

        assert metrics
        assert not any("reward_extra_info" in key for key in metrics)
        assert any("mean" in key for key in metrics)


class _FakeTrainer:
    """Just the method under test; `PSRL_RayPPOTrainer` needs a full Ray cluster."""

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

        return PSRL_RayPPOTrainer._val_metrics_update(
            self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )


def _source(relpath: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return (root / relpath).read_text()

"""Handshake notifications carry their kind, and the server routes on it (item E3).

The three handshake payloads were all `{client_name: value}` pickled, so nothing in the
bytes distinguished them. The server could only tell them apart by *which phase it was
in*, which works exactly as long as the handshake is lock-step -- and a client joining
after it is, by definition, not.

Two failure shapes follow from that, and both are silent. Neither raises, neither logs:

* a notification consumed by the wrong `wait_for_*` lands in the wrong registry, because
  one `{name: value}` dict looks like any other;
* a notification nobody is waiting for is dropped, because `get_new_notifs` is
  destructive and hands each one to a single caller.

So the tests below are mostly about *misrouting* rather than about round-tripping. A
format that survives a round trip but is accepted by the wrong reader is the bug this
replaced.
"""

from __future__ import annotations

import pickle

import pytest

from psrl.utils.nixl.handshake_notif import NotifKind, decode, encode

pytestmark = pytest.mark.cpu_test


class TestTheKindSurvivesTheWire:
    def test_a_notification_round_trips(self):
        raw = encode(NotifKind.SHARDING, {"train_0": {"w": "sharding"}})

        notif = decode(raw)

        assert notif.kind is NotifKind.SHARDING
        assert notif.entries == {"train_0": {"w": "sharding"}}

    @pytest.mark.parametrize("kind", list(NotifKind))
    def test_every_kind_decodes_back_to_itself(self, kind):
        """The whole point: the kind is carried, not inferred."""
        assert decode(encode(kind, {"c": 1})).kind is kind

    def test_several_clients_ride_in_one_notification(self):
        """A multi-client agent reports for all of its clients at once."""
        entries = {"train_0": "a", "train_1": "b"}

        assert decode(encode(NotifKind.CLIENT_INFO, entries)).entries == entries


class TestTheKindsAreNotConfusableWithEachOther:
    """The property the old format lacked, stated directly."""

    def test_identical_payloads_under_different_kinds_stay_distinct(self):
        """Same bytes of payload, different kind -- previously indistinguishable.

        This is the exact pair that broke: a sharding dict and a client-info dict with
        the same shape. Under the old format both decoded to the same object and the
        reader that happened to be running claimed it.
        """
        entries = {"train_0": "identical payload"}

        sharding = decode(encode(NotifKind.SHARDING, entries))
        info = decode(encode(NotifKind.CLIENT_INFO, entries))

        assert sharding.kind is not info.kind
        assert sharding.entries == info.entries, "the payloads really are identical"


class TestForeignTrafficIsLeftAlone:
    """Other subsystems publish on the same agent stream.

    `decode` returns None rather than raising for anything it does not recognise. A raise
    would turn an unrelated message into a handshake failure -- and the update-info
    reader did raise, so an untagged notification arriving there took the run down
    instead of being skipped.
    """

    def test_a_bare_dict_is_not_a_handshake_notification(self):
        """The old wire format must not be mistaken for the new one."""
        assert decode(pickle.dumps({"train_0": "sharding"})) is None

    def test_a_chunk_notification_is_not_claimed(self):
        """Peer diffusion sends its own payloads through the same stream."""
        assert decode(pickle.dumps(("chunk-ready", 7))) is None

    def test_unpicklable_bytes_return_none_rather_than_raising(self):
        assert decode(b"not a pickle at all") is None

    def test_a_tuple_of_the_right_length_but_wrong_magic_is_rejected(self):
        """Shape alone must not be enough -- that was the original mistake."""
        assert decode(pickle.dumps(("some.other.protocol", "sharding", {"c": 1}))) is None

    def test_an_unknown_kind_is_rejected_rather_than_guessed(self):
        """A peer sending a kind this build has never heard of must not be filed
        somewhere arbitrary."""
        from psrl.utils.nixl.handshake_notif import _MAGIC

        assert decode(pickle.dumps((_MAGIC, "kind_from_the_future", {"c": 1}))) is None

    def test_entries_must_be_a_mapping(self):
        from psrl.utils.nixl.handshake_notif import _MAGIC

        assert decode(pickle.dumps((_MAGIC, "sharding", ["not", "a", "mapping"]))) is None


class TestTheServerRoutesOnTheKind:
    """The drain files each notification by tag, and serves all kinds from one read.

    One drain because `get_new_notifs` is destructive. Three independent readers polling
    it means whichever runs first consumes the others' mail, which is what made an
    out-of-phase arrival vanish.
    """

    def _server(self, notifs: dict[str, list[bytes]]):
        """A stand-in holding only what `_drain_notifs` touches.

        The real method is called unbound against it, so this exercises the shipped
        implementation rather than a restatement of it.
        """
        from psrl.utils.nixl.server import NIXLMetaServer

        class _Agent:
            def __init__(self):
                self._pending = notifs

            def get_new_notifs(self):
                drained, self._pending = self._pending, {}
                return drained

        class _Server:
            def __init__(self):
                self.agent = _Agent()
                self.client_sharding_dicts = {}
                self.client_infos = {}
                self._client_temp_mappings = {}
                self.connected_clients = {}
                self._agents_reported = {kind: set() for kind in NotifKind}

            _add_client = NIXLMetaServer._add_client
            _drain_notifs = NIXLMetaServer._drain_notifs

        return _Server()

    def test_sharding_and_temp_mapping_land_in_their_own_registries(self):
        server = self._server(
            {
                "agent_0": [
                    encode(NotifKind.SHARDING, {"train_0": "the sharding"}),
                    encode(NotifKind.TEMP_MAPPING, {"train_0": "the mapping"}),
                ]
            }
        )

        server._drain_notifs()

        assert server.client_sharding_dicts == {"train_0": "the sharding"}
        assert server._client_temp_mappings == {"train_0": "the mapping"}

    def test_one_drain_serves_every_kind_at_once(self):
        """Two kinds in a single read -- three separate readers could not do this."""
        server = self._server(
            {
                "agent_0": [encode(NotifKind.SHARDING, {"train_0": "s"})],
                "agent_1": [encode(NotifKind.TEMP_MAPPING, {"train_1": "m"})],
            }
        )

        server._drain_notifs()

        assert server._agents_reported[NotifKind.SHARDING] == {"agent_0"}
        assert server._agents_reported[NotifKind.TEMP_MAPPING] == {"agent_1"}

    def test_an_out_of_phase_arrival_is_filed_not_discarded(self):
        """The elastic case: sharding arrives when no one is waiting for sharding.

        Under the old code this was consumed by whichever loop was running and written
        into that loop's registry.
        """
        server = self._server({"agent_9": [encode(NotifKind.SHARDING, {"late": "s"})]})

        server._drain_notifs()

        assert server.client_sharding_dicts == {"late": "s"}
        assert server.client_infos == {}, "it must not leak into another registry"
        assert "agent_9" in server._agents_reported[NotifKind.SHARDING]

    def test_foreign_payloads_do_not_stop_the_drain(self):
        """A notification from another subsystem must not cost the ones behind it."""
        server = self._server(
            {
                "agent_0": [
                    pickle.dumps({"untagged": "legacy"}),
                    b"garbage",
                    encode(NotifKind.SHARDING, {"train_0": "s"}),
                ]
            }
        )

        server._drain_notifs()

        assert server.client_sharding_dicts == {"train_0": "s"}

    def test_the_roster_records_the_agent_not_the_client(self):
        """One agent may host several clients and reports for all of them at once, so
        counting clients would overshoot `expected_agents`."""
        server = self._server(
            {"agent_0": [encode(NotifKind.SHARDING, {"c0": "s", "c1": "s", "c2": "s"})]}
        )

        server._drain_notifs()

        assert server._agents_reported[NotifKind.SHARDING] == {"agent_0"}
        assert len(server.client_sharding_dicts) == 3

    def test_an_update_carries_info_and_mapping_together(self):
        """Resharding re-sends both, so its payload has a different shape."""
        from psrl.utils.nixl.nixl_spec import NIXLClientInfo

        info = NIXLClientInfo(
            name="train_0",
            node_ip="127.0.0.1",
            node_gpu_id=0,
            type=None,
            tensor_infos={},
            meta=b"",
        )
        server = self._server(
            {
                "agent_0": [
                    encode(
                        NotifKind.UPDATE_INFO,
                        {"train_0": {"info": info.serialize(), "temp_mapping": "m"}},
                    )
                ]
            }
        )

        server._drain_notifs()

        assert server.client_infos["train_0"].name == "train_0"
        assert server._client_temp_mappings == {"train_0": "m"}


class TestTheWaitNoLongerFreezesTheRoster:
    """`wait_for_*` used to return early once a "handshake finished" flag was set.

    That flag is what made membership permanently whatever the first handshake saw: a
    later caller asking for more agents got the old roster back immediately, and the
    joining client waited forever for a server that had stopped listening.
    """

    def test_no_completion_flags_remain(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/utils/nixl/server.py"
        ).read_text()

        assert "_is_all_client_shardings_recved" not in source
        assert "_is_all_client_infos_recved" not in source
        assert "_is_all_temp_mappings_recved" not in source

    def test_the_waits_share_one_implementation(self):
        """Three copies of the same loop is how they drifted apart in the first place --
        only one of them logged its failure usefully."""
        import inspect

        from psrl.utils.nixl.server import NIXLMetaServer

        for name in (
            "wait_for_client_shardings",
            "wait_for_client_infos",
            "wait_for_client_temp_mappings",
            "wait_for_update_infos",
        ):
            body = inspect.getsource(getattr(NIXLMetaServer, name))
            assert "_wait_for(" in body
            assert "get_new_notifs" not in body

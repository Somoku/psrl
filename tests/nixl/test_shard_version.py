"""Version equality as a runtime assertion, not a content hash (§3.4).

The design lists "版本相等性" among four invariants that must be runtime assertions, and
says the metadata-only ones are constant-cost and should be on by default. These tests pin
that the check does what the invariant needs *and* stays cheap enough to leave on.

The property that matters most is the one a content hash gets wrong: the check must be
affordable at any model size. Hashing an 8B model measured ~61 s against a 1.4 s pull, so
it could only run sampled and off by default -- which means the fault lands on the keys
that were not sampled. A version stamp is O(1) per shard regardless of model or dtype.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.shard_version import (
    UNVERSIONED,
    ShardStaleness,
    check_pull_versions,
    describe,
)

pytestmark = pytest.mark.cpu_test


def _published(source: str, key: str, shard_versions: dict) -> dict:
    return {source: {key: shard_versions}}


class TestItCatchesTheFaultsVersionsCanCatch:
    def test_a_matching_pull_reports_nothing(self):
        published = _published("train_0", "w", {(0,): 7, (1,): 7})

        assert check_pull_versions(7, published) == []

    def test_a_stale_shard_is_reported(self):
        """The dominant real fault: a reader that pulled before the supplier republished."""
        published = _published("train_0", "w", {(0,): 7, (1,): 6})

        stale = check_pull_versions(7, published)

        assert len(stale) == 1
        assert stale[0].shard == (1,)
        assert stale[0].is_stale

    def test_a_torn_read_shows_as_several_shards(self):
        """Shards of one key at different versions is what a half-finished publish is."""
        published = _published("train_0", "w", {(0,): 7, (1,): 6, (2,): 6})

        stale = check_pull_versions(7, published)

        assert {item.shard for item in stale} == {(1,), (2,)}

    def test_an_unpublished_shard_is_distinguished_from_stale(self):
        """Registered-but-never-published is a different bug from one version behind."""
        published = _published("train_0", "w", {(0,): UNVERSIONED})

        stale = check_pull_versions(7, published)

        assert stale[0].actual is None
        assert not stale[0].is_stale
        assert not stale[0].is_ahead

    def test_a_reader_ahead_of_the_supplier_is_distinguished(self):
        """Ahead means the reader raced a publish; behind means it pulled too early.

        Separated because the remedies differ, and because a single "mismatch" bucket
        would make a racing reader look like a stale supplier.
        """
        published = _published("train_0", "w", {(0,): 9})

        stale = check_pull_versions(7, published)

        assert stale[0].is_ahead
        assert not stale[0].is_stale

    def test_version_zero_is_not_confused_with_unpublished(self):
        """At startup every shard is genuinely unpublished; v0 is a real version."""
        published = _published("train_0", "w", {(0,): 0})

        assert check_pull_versions(0, published) == []
        assert check_pull_versions(1, published)[0].actual == 0


class TestTheReportPreservesTheDiagnosis:
    def test_every_mismatch_is_reported_not_just_the_first(self):
        """The pattern is the diagnosis; stopping early discards it."""
        published = {
            "train_0": {"w": {(0,): 6, (1,): 6}},
            "train_1": {"w": {(0,): 7}},
        }

        assert len(check_pull_versions(7, published)) == 2

    def test_a_failing_source_is_visible_as_one_source_many_shards(self):
        published = {
            "train_0": {"a": {(0,): 6}, "b": {(0,): 6}},
            "train_1": {"a": {(0,): 7}, "b": {(0,): 7}},
        }

        stale = check_pull_versions(7, published)

        assert {item.source for item in stale} == {"train_0"}

    def test_a_layout_bug_is_visible_as_one_shard_many_sources(self):
        published = {
            "train_0": {"w": {(0,): 7, (1,): 6}},
            "train_1": {"w": {(0,): 7, (1,): 6}},
        }

        stale = check_pull_versions(7, published)

        assert {item.shard for item in stale} == {(1,)}
        assert {item.source for item in stale} == {"train_0", "train_1"}

    def test_the_summary_leads_with_counts(self):
        """So "all shards of one source" and "one shard of one key" need no counting by eye."""
        published = {"train_0": {"w": {(0,): 6, (1,): 6}}}

        summary = describe(check_pull_versions(7, published))

        assert "2 shard(s)" in summary
        assert "1 source(s)" in summary
        assert "2 behind" in summary

    def test_the_summary_is_bounded(self):
        """A layout bug yields one entry per shard; an unbounded message buries the shape."""
        published = {"train_0": {"w": {(i,): 6 for i in range(50)}}}

        summary = describe(check_pull_versions(7, published), limit=3)

        assert "and 47 more" in summary

    def test_an_empty_result_says_so(self):
        assert describe([]) == "no version mismatches"

    def test_the_message_names_key_shard_and_source(self):
        item = ShardStaleness(key="w", shard=(1,), source="train_2", expected=7, actual=6)

        rendered = str(item)

        assert "w" in rendered
        assert "(1,)" in rendered
        assert "train_2" in rendered
        assert "stale by 1" in rendered


class TestTheStampRidesTheExistingMetadata:
    def test_it_survives_serialization(self):
        import torch

        from psrl.utils.nixl.nixl_spec import NIXLShardMetaInfo

        meta = NIXLShardMetaInfo(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            shape=torch.Size([4, 4]),
            stride=(4, 1),
            is_contiguous=True,
            version=11,
        )

        assert NIXLShardMetaInfo.deserialize(meta.serialize()).version == 11

    def test_an_older_peer_reads_as_unpublished(self):
        """A peer predating the field must never look like a version match."""
        import torch

        from psrl.utils.nixl.nixl_spec import NIXLShardMetaInfo

        old_form = (str(torch.bfloat16), "cpu", (4, 4), (4, 1), True, False)

        assert NIXLShardMetaInfo.deserialize(old_form).version == UNVERSIONED

    def test_the_default_is_unpublished(self):
        """Registered is not published; defaulting to 0 would fake a match at startup."""
        import torch

        from psrl.utils.nixl.nixl_spec import NIXLShardMetaInfo

        meta = NIXLShardMetaInfo(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            shape=torch.Size([2]),
            stride=(1,),
            is_contiguous=True,
        )

        assert meta.version == UNVERSIONED


class TestItStaysCheapEnoughToLeaveOn:
    def test_the_check_does_not_touch_tensors(self):
        """The whole point: cost is independent of model size, dtype and quantisation.

        Asserted structurally rather than by timing -- the module must not import torch,
        because anything that reads tensor bytes reintroduces the cost that made the
        content-hash approach unusable by default.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/utils/nixl/shard_version.py"
        ).read_text()

        assert "import torch" not in source


class TestTheStampIsWrittenWhereReadersCanSeeIt:
    """Ordering, checked without a cluster -- it is the part that kept being wrong.

    Three cluster runs were spent discovering that this is an *ordering* property, not a
    logic one, and each cost ~30 minutes. All three failures are statements about where
    calls sit relative to each other, which source inspection settles in milliseconds:

      run 29/30  stamped after `nixl_protocol` -> readers cached `-1`
      run 32     stamped inside the protocol but asked the PSManager for the version,
                 which deadlocks against its own `wait_for_client_infos`

    So these are structural assertions. They do not prove the transport works; they prove
    the two mistakes that actually happened cannot silently come back.
    """

    def _worker_source(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[2]
            / "psrl/workers/train/engine_train_worker.py"
        ).read_text()

    def test_the_stamp_sits_between_registration_and_send(self):
        """Before step 4 there is no metadata to write into; after step 5 peers have
        already cached the copy that says "never published"."""
        source = self._worker_source()

        register_at = source.index("register_local_tensors")
        publish_at = source.index("publish_version(bootstrap_version)")
        send_at = source.index("send_local_info()")

        assert register_at < publish_at < send_at

    def test_the_worker_does_not_ask_the_manager_during_the_handshake(self):
        """This runs inside `nixl_protocol`; the manager is blocked in
        `wait_for_client_infos` waiting for this very rank, so a `ray.get` to it hangs
        until the handshake times out."""
        source = self._worker_source()
        start = source.index("def nixl_protocol")
        body = source[start : source.index("\n    def ", start + 10)]

        assert "get_ps_model_version" not in body

    def test_the_trainer_reads_the_version_before_dispatching_any_handshake(self):
        """Before the *meta server's* dispatch, not merely before the actor group's.

        `PSManager` is a Ray actor, so its calls serialise. Once `nixl_protocol` is
        running it sits in `wait_for_client_shardings`, and any later call on that
        handle queues behind it -- including this version read. The train ranks are in
        turn waiting on that read before they start their half of the handshake, so the
        handshake can never complete.

        Measured: zero clients connected in 27 minutes, with all six vLLM replicas up
        and healthy, which looked exactly like the slow-startup failure it was not.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/trainer/ppo/ray_trainer.py"
        ).read_text()

        read_at = source.index("bootstrap_version = (")
        server_dispatch_at = source.index("self.ps_manager_handle.nixl_protocol.remote()")
        worker_dispatch_at = source.index(
            '"nixl_protocol", actor_protocol_mode, bootstrap_version'
        )

        assert read_at < server_dispatch_at < worker_dispatch_at

    def test_only_direct_supply_stamps(self):
        """With a PS the weights come from it, so a train-rank stamp would claim
        authorship of bytes this rank did not publish."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/trainer/ppo/ray_trainer.py"
        ).read_text()
        at = source.index("bootstrap_version = (")

        assert "has_ps_data_plane" in source[at : at + 300]

    def test_both_overrides_take_the_argument(self):
        """A base class that does not accept it turns the extra argument into a TypeError
        inside a Ray actor, which surfaces as a handshake timeout rather than an error."""
        import inspect

        from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker
        from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

        for cls in (PSRL_BaseTrainWorker, PSRL_EngineTrainWorker):
            assert "bootstrap_version" in inspect.signature(cls.nixl_protocol).parameters


class TestTheHandshakeBudgetIsConfigurable:
    """Startup slowness must not be indistinguishable from a broken handshake.

    Two multi-node runs died in `wait_for_client_shardings` with "Timeout waiting for
    agents" while vLLM was still capturing CUDA graphs -- the code was fine and the
    cluster was merely slower that hour (18.6 min to reach the handshake on one run,
    23.5 min on the next). The message names agents and connectivity, which is why both
    runs were first investigated as transport faults.
    """

    def _source(self, relative: str) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / relative).read_text()

    def test_server_and_client_read_the_same_key(self):
        """They were 600s and 1200s: the server gave up first, so the clients were still
        waiting when it did and logged nothing useful."""
        for relative in ("psrl/utils/nixl/server.py", "psrl/utils/nixl/client.py"):
            assert 'getattr(nixl_config, "handshake_timeout_s"' in self._source(relative)

    def test_an_explicit_timeout_still_wins(self):
        """A caller that knows its own bound must not be overridden by the config."""
        source = self._source("psrl/utils/nixl/server.py")

        assert "self._handshake_timeout if timeout is None else timeout" in source

    def test_both_client_classes_define_the_attribute(self):
        """`NIXLMultiStorageClients` shares the wait methods; without the attribute they
        raise AttributeError instead of timing out."""
        source = self._source("psrl/utils/nixl/client.py")

        assert source.count("self._handshake_timeout = float(") == 2

    def test_the_config_documents_why_it_is_tunable(self):
        source = self._source("psrl/trainer/config/psrl/nixl.yaml")

        assert "handshake_timeout_s" in source
        assert "18.6" in source and "23.5" in source

    def test_no_wait_compares_against_an_unresolved_timeout(self):
        """Every `None`-defaulted wait must compare against `budget`, not `timeout`.

        The regression this exists for: the signatures were changed to
        `timeout: float | None = None` and a `budget` was resolved, but the comparison in
        some bodies still read `timeout`. That makes `time.time() - start > None` raise
        `TypeError: '>' not supported between instances of 'float' and 'NoneType'` the
        moment a caller omits the argument -- which every production caller does.

        It cost a multi-node run, and it surfaced on the *server* as "Timeout waiting for
        agents" because the clients died before connecting, which is the same symptom as
        the slow-startup problem this change was meant to fix.
        """
        import re

        for relative in ("psrl/utils/nixl/server.py", "psrl/utils/nixl/client.py"):
            source = self._source(relative)
            offenders = []
            for match in re.finditer(r"    (?:async )?def (\w+)\(", source):
                name = match.group(1)
                body = source[match.end() : source.find("\n    def ", match.end())]
                signature = source[match.start() : match.end() + 200]
                if "timeout: float | None = None" not in signature:
                    continue
                if "> timeout:" in body:
                    offenders.append(f"{relative}::{name}")

            assert not offenders, (
                f"these waits default `timeout` to None but still compare against it, "
                f"so omitting the argument raises TypeError: {offenders}"
            )

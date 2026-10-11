"""Verify that what a rollout received equals what the train rank published.

Direct supply removes an observation point. With a PS in the middle you could compare
"the version in the PS" against "the version in the rollout" and tell a transport fault
apart from a training fault; `_debug_log_ps_info` exists for exactly that. Reading the
train ranks directly deletes the middle term, so a wrong tensor on the rollout has no
cheap explanation (§3.7, "需补一个可选的 checksum 校验模式").

That matters more here than it would for a push transport, because an FP8/RDMA fault does
not raise. The receiving buffer is already allocated and already the right shape, so a
descriptor that lands short, lands in the wrong place, or lands stale produces a tensor
that is *valid* and *wrong* -- and training continues, slightly off, for as long as nobody
looks.

The checksum is deliberately cheap and deliberately not cryptographic: it runs on the GPU
over tensors that are already resident, and it only has to catch corruption, never resist
an adversary. What it must not do is cost anything when switched off, which is why the
callers gate on `enabled` before building anything.

Cross-rank comparison is the caller's job. This module answers "what does this tensor hash
to"; who compares which digests is a question about the comm plan, and lives with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChecksumReport:
    """One side's digest of one weight version."""

    #: Model version the digest was taken at. Comparing across versions is meaningless,
    #: so the version travels with the digest rather than being assumed by the caller.
    version: int
    #: Digest per weight key. Keyed rather than a single whole-model number so a mismatch
    #: names the tensor that diverged -- "the model differs" is not actionable, and with
    #: one number per model a single bad shard looks the same as a wholly wrong version.
    digests: dict[str, int]

    def compare(self, other: ChecksumReport) -> list[str]:
        """Return the keys that differ, or the empty list when they agree.

        A key present on one side only counts as differing: under direct supply the two
        sides derive their key sets independently, so a missing key is a real fault and
        not a reason to skip the comparison.
        """
        if self.version != other.version:
            raise ValueError(
                f"refusing to compare checksums across versions "
                f"{self.version} and {other.version}; a mismatch would say nothing "
                f"about the transport"
            )

        differing = [
            key
            for key in sorted(set(self.digests) | set(other.digests))
            if self.digests.get(key) != other.digests.get(key)
        ]
        return differing


def tensor_digest(tensor: torch.Tensor) -> int:
    """A cheap order-sensitive digest of a tensor's bytes.

    Order-sensitive on purpose: a plain sum would miss a permuted shard, which is one of
    the failure modes a re-shard across differing TP degrees can actually produce. The
    positional weights make "same values, wrong order" visible.

    Computed in float64 over a byte view, so it is dtype-agnostic -- FP8 and BF16 tensors
    have no float64 path of their own, and reinterpreting as `uint8` avoids needing one.
    A float sum of the values themselves would also lose low bits exactly where FP8 needs
    them most.
    """
    if tensor.numel() == 0:
        return 0

    flat = tensor.detach().reshape(-1)
    # Fold the length in, so an all-zero buffer does not hash like an absent one -- an
    # unfilled landing buffer is precisely the corruption this has to catch, and with a
    # purely multiplicative digest every zero tensor collides with the empty one.
    length_term = flat.numel() % _MODULUS
    if not flat.is_contiguous():
        flat = flat.contiguous()

    as_bytes = flat.view(torch.uint8).to(torch.float64)
    positions = torch.arange(
        1, as_bytes.numel() + 1, device=as_bytes.device, dtype=torch.float64
    )

    # Modulo keeps the running value inside float64's exact-integer range (2^53) for
    # tensors of any size; without it a multi-GB tensor silently loses precision and two
    # different tensors start colliding.
    weighted = torch.remainder(as_bytes * positions, _MODULUS)
    total = torch.remainder(weighted.sum(), _MODULUS).item()
    return int((total + length_term) % _MODULUS)


#: A prime below 2^53 / 255, so `byte * position` and the running sum both stay exactly
#: representable in float64.
_MODULUS = 2_147_483_647


def checksum_state_dict(
    state_dict: dict[str, torch.Tensor],
    version: int,
    *,
    keys: list[str] | None = None,
) -> ChecksumReport:
    """Digest every tensor in `state_dict`, or just `keys` when given.

    `keys` exists for the sampled mode: hashing a whole model every step is affordable
    only because the tensors are already on the GPU, and at DeepSeek-V3 scale even that
    is not free. Sampling a stable subset keeps the check continuous rather than
    all-or-nothing.
    """
    selected = state_dict.keys() if keys is None else keys
    return ChecksumReport(
        version=version,
        digests={key: tensor_digest(state_dict[key]) for key in selected if key in state_dict},
    )


def sample_keys(state_dict: dict[str, torch.Tensor], count: int) -> list[str]:
    """Pick `count` keys deterministically, spread across the model.

    Deterministic and spread, not random: every rank must choose the *same* keys without
    communicating, or the digests are not comparable. Striding over the sorted key list
    gives both, and spreading across the model matters because faults are not uniform --
    a layout bug tends to hit one layer family, so sampling the first N keys would look
    clean while a later block was wrong.
    """
    ordered = sorted(state_dict)
    if count <= 0 or not ordered:
        return []
    if count >= len(ordered):
        return ordered

    stride = len(ordered) / count
    return [ordered[int(index * stride)] for index in range(count)]

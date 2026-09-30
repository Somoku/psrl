"""
Represent sampler distributions as compact response-aligned sparse rows.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np


def capture_width(mode: str, max_tokens: int) -> int:
    """
    Reserve a sentinel entry to prove support completeness within the budget.
    """
    if mode not in ("disabled", "topk", "support") or not 1 <= max_tokens < 2**31 - 1:
        raise ValueError("Invalid sampler distribution mode or sampler_max_tokens budget.")
    return 0 if mode == "disabled" else max_tokens + (mode == "support")


def configure_capture(body: dict, mode: str, max_tokens: int) -> None:
    """
    Apply the sampler capture contract to a chat request.
    """
    body["logprobs"] = True
    body["top_logprobs"] = 0
    body["sampler_top_logprobs"] = capture_width(mode, max_tokens)
    if mode == "disabled":
        return
    if float(body.get("temperature", 1.0)) <= 0:
        raise ValueError("Sampler distribution capture requires positive temperature.")
    if mode == "topk" and (
        float(body.get("top_p", 1.0)) < 1 or int(body.get("top_k", -1)) > 0 or float(body.get("min_p", 0.0)) > 0
    ):
        raise ValueError("Truncated sampling requires sampler_distribution=support.")


@dataclass(frozen=True)
class SamplerDistribution:
    """
    Store token IDs and log probabilities with one offset per response boundary.

    Empty rows represent environment tokens. `complete` distinguishes a captured
    support from an approximate head without inferring completeness from mass.
    """

    offsets: np.ndarray
    token_ids: np.ndarray
    log_probs: np.ndarray
    complete: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def prefix(self, length: int) -> SamplerDistribution:
        """
        Truncate response rows without copying the entry buffers.
        """
        if not 0 <= length <= len(self):
            raise ValueError(f"Invalid sampler distribution prefix length {length!r}.")
        end = int(self.offsets[length])
        return SamplerDistribution(
            self.offsets[: length + 1], self.token_ids[:end], self.log_probs[:end], self.complete[:length]
        )

    @classmethod
    def empty(cls, rows: int) -> SamplerDistribution:
        """
        Construct rows for masked environment tokens.
        """
        return cls(
            np.zeros(rows + 1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.zeros(rows, dtype=np.bool_),
        )

    @classmethod
    def from_wire(cls, payload: dict, sampled: list) -> SamplerDistribution:
        """
        Decode and validate a TITO distribution against its sampled tokens.
        """
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        ids = np.frombuffer(base64.b64decode(payload["token_ids"], validate=True), dtype="<i4")
        log_probs = np.frombuffer(base64.b64decode(payload["log_probs"], validate=True), dtype="<f4")
        complete = np.asarray(payload["complete"], dtype=np.bool_)
        rows = len(sampled)
        if (
            offsets.ndim != 1
            or len(offsets) != rows + 1
            or complete.shape != (rows,)
            or offsets[0] != 0
            or offsets[-1] != len(ids)
            or len(ids) != len(log_probs)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("Sampler distribution offsets do not align with sampled tokens.")
        if np.any(ids < 0) or not np.isfinite(log_probs).all() or np.any(log_probs > 1e-6):
            raise ValueError("Sampler distribution contains invalid IDs or log probabilities.")
        for row, (sample_logprob, sample_id) in enumerate(sampled):
            if not np.isfinite(sample_logprob) or sample_logprob > 1e-6 or sample_id < 0:
                raise ValueError(f"Invalid sampled token or log probability at row {row}.")
            start, end = offsets[row : row + 2]
            row_ids, row_lp = ids[start:end], log_probs[start:end]
            if len(np.unique(row_ids)) != len(row_ids):
                raise ValueError(f"Duplicate sampler token IDs at row {row}.")
            mass = np.exp(row_lp.astype(np.float64)).sum()
            if mass > 1 + 2e-5 or (complete[row] and abs(mass - 1) > 2e-5):
                raise ValueError(f"Invalid sampler probability mass {mass!r} at row {row}.")
            selected = row_ids == sample_id
            if complete[row] and not selected.any():
                raise ValueError(f"Complete sampler support omits the sampled token at row {row}.")
            if selected.any() and not np.isclose(row_lp[selected][0], sample_logprob, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Sampler head and sampled log probability disagree at row {row}.")
        return cls(offsets, ids, log_probs, complete)

    @classmethod
    def concatenate(cls, segments: list[SamplerDistribution]) -> SamplerDistribution:
        """
        Join turns and environment rows with one allocation per entry buffer.
        """
        if not segments:
            return cls.empty(0)
        counts = np.concatenate([np.diff(segment.offsets) for segment in segments])
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(
            offsets,
            np.concatenate([segment.token_ids for segment in segments]),
            np.concatenate([segment.log_probs for segment in segments]),
            np.concatenate([segment.complete for segment in segments]),
        )

    def validate_capture(self, mode: str, max_tokens: int, response_mask: list[int]) -> None:
        """
        Reject incomplete support and missing active rows before training.
        """
        active = np.asarray(response_mask, dtype=np.bool_)
        if active.shape != (len(self),):
            raise ValueError("Sampler distribution and response mask lengths differ.")
        counts = np.diff(self.offsets)
        if np.any(counts[active] == 0) or np.any(counts[~active] != 0):
            raise ValueError("Sampler distribution rows do not match the response mask.")
        if mode == "support" and (not self.complete[active].all() or np.any(counts > max_tokens)):
            raise ValueError("Sampler support exceeds the capture budget. Increase sampler_max_tokens.")

    def as_tensor_fields(self, prompt_length: int) -> dict:
        """
        Place sparse rows at next-token prediction positions for packed training.
        """
        import torch

        if prompt_length < 1:
            raise ValueError("Sampler distributions require a nonempty prompt.")
        # Entries remain compact. Only row offsets and completeness cover the full sequence.
        offsets = np.concatenate((np.zeros(prompt_length - 1, dtype=np.int64), self.offsets, self.offsets[-1:]))
        complete = np.concatenate((np.zeros(prompt_length - 1, dtype=np.bool_), self.complete, [False]))
        return {
            "sampler_offsets": torch.from_numpy(offsets),
            "sampler_token_ids": torch.from_numpy(self.token_ids.copy()),
            "sampler_log_probs": torch.from_numpy(self.log_probs.copy()),
            "sampler_complete": torch.from_numpy(complete),
        }

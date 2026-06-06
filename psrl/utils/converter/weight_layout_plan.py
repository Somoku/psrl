"""
Weight Layout Plan Execution for PSRL

This module executes WeightLayoutPlan from vLLM models, converting vLLM runtime
parameters to HuggingFace checkpoint format.

Key classes:
- PlanExecutor: Executes a flattened WeightLayoutPlan on a model's parameters
- ConvertedWeight: Output fragment with HF name, tensor, and sharding metadata
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from vllm_patches.weight_layout import ResolvedWeightLayoutPlan

from psrl.utils.nixl.nixl_spec import NIXLSharding


@dataclass
class ConvertedWeight:
    """Output fragment from parameter conversion.
    
    Attributes:
        name: HuggingFace parameter name
        param: Tensor (view or copy)
        sharding: Optional NIXL sharding metadata
    """
    name: str
    param: torch.Tensor
    sharding: NIXLSharding | None = None


class PlanExecutor:
    """Executes a WeightLayoutPlan to convert vLLM parameters to HF format.
    
    Usage:
        plan = model.build_weight_layout().flatten()
        executor = PlanExecutor(plan, tp_rank=0)
        for param_name, param, module in model_params:
            for converted in executor.execute(param_name, param, module):
                output_state_dict[converted.name] = converted.param
                output_sharding[converted.name] = converted.sharding
    """
    
    def __init__(
        self,
        plan: ResolvedWeightLayoutPlan,
        tp_rank: int = 0,
        ep_rank: int = 0,
    ):
        """Initialize executor with a plan.
        
        Args:
            plan: Flattened WeightLayoutPlan from model.build_weight_layout().flatten()
            tp_rank: Tensor parallel rank
            ep_rank: Expert parallel rank
        """
        self.plan = plan
        self.tp_rank = tp_rank
        self.ep_rank = ep_rank
    
    def execute(
        self,
        param_name: str,
        param: torch.Tensor,
        module: nn.Module | None = None,
    ) -> Iterable[ConvertedWeight]:
        """Execute parameter conversion.

        Args:
            param_name: vLLM parameter name (full path, e.g. "model.layers.0.self_attn.qkv_proj.weight")
            param: Parameter tensor
            module: The module containing this parameter

        Yields:
            ConvertedWeight fragments
        """
        # Check if parameter should be excluded
        if self.plan.should_exclude(param_name):
            return

        # Find matching rules
        matching_rules = self.plan.matches_rules(param_name, module)

        if not matching_rules:
            # No matching rule - parameter is passthrough (HF format already)
            # Apply reverse name map if present
            hf_name = param_name
            if self.plan.name_map is not None:
                hf_name = self.plan.name_map.to_hf_or_identity(hf_name)
            yield ConvertedWeight(name=hf_name, param=param)
            return

        # Apply each matching rule (there should typically be only one)
        for flattened_rule in matching_rules:
            yield from self._apply_rule(
                param_name,
                param,
                module,
                flattened_rule,
            )

    def _apply_rule(
        self,
        full_name: str,
        param: torch.Tensor,
        module: nn.Module | None,
        flattened_rule,
    ) -> Iterable[ConvertedWeight]:
        """Apply a single flattened rule to a parameter.

        For SUFFIX match rules the transform produces *relative* fragment names
        (e.g. ``"q_proj.weight"``).  We need to reconstruct the full HF path by
        prepending the base that was stripped when matching.

        Args:
            full_name: Full vLLM parameter name (e.g.
                ``"model.layers.0.self_attn.qkv_proj.weight"``)
            param: Tensor to convert
            module: Module context
            flattened_rule: FlattenedRule (has ``.rule`` and ``.prefix``)

        Yields:
            ConvertedWeight fragments with fully-qualified HF names
        """
        from vllm_patches.weight_layout import MatchMode

        from psrl.utils.converter.weight_layout_transforms import TransformExecutor

        rule = flattened_rule.rule

        # ------------------------------------------------------------------
        # Compute the name_base used to reconstruct the full HF path.
        #
        # For SUFFIX matches the transform emits names relative to the
        # suffix pattern (e.g. "q_proj.weight"), so we need:
        #   name_base = full_name[: -len(pattern)]
        #
        # When the rule lives inside a mounted sub-plan the FlattenedRule
        # carries a prefix (e.g. "language_model").  The param_name passed
        # here is always the *full* name, so we just strip the pattern from
        # the end regardless of prefix.
        #
        # For EXACT / PREFIX / REGEX matches the transform is expected to
        # emit fully-qualified names already, so name_base stays None.
        # ------------------------------------------------------------------
        name_base: str | None = None
        if rule.match == MatchMode.SUFFIX:
            pattern = rule.vllm_pattern
            if full_name.endswith(pattern):
                name_base = full_name[: -len(pattern)]

        # Create transform executor
        transform_exec = TransformExecutor(
            tp_rank=self.tp_rank,
            ep_rank=self.ep_rank,
        )

        # Execute transform to get fragments
        fragments = transform_exec.execute(
            transform=rule.transform,
            param=param,
            module=module,
            full_name=full_name,
        )

        # Apply name base + global name map and yield
        for fragment in fragments:
            hf_name = fragment.name

            # Handle identity transform sentinel:
            # name=None means "keep original vLLM name" (no rename was specified)
            if hf_name is None:
                hf_name = full_name
            elif name_base is not None:
                # SUFFIX match: prepend base to reconstruct full path
                # e.g. "model.layers.0.self_attn." + "q_proj.weight"
                hf_name = name_base + hf_name
            # else: EXACT/PREFIX/REGEX — fragment.name is already the full path

            # Apply global name map if present
            if self.plan.name_map is not None:
                hf_name = self.plan.name_map.to_hf_or_identity(hf_name)

            yield ConvertedWeight(
                name=hf_name,
                param=fragment.param,
                sharding=fragment.sharding,
            )

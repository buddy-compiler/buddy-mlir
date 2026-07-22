# ===- template_materialization.py -------------------------------------------
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-------------------------------------------------------------------------

"""Compile-time plan for materializing one subgraph per unique template."""

from dataclasses import dataclass

from .operation import Op
from .region_analysis import GraphRegion, GraphValueRef, RegionInputKind
from .type import TensorMeta


@dataclass(frozen=True)
class TemplateUnit:
    template_id: int
    representative: GraphRegion
    instances: tuple[GraphRegion, ...]


@dataclass
class TemplateMaterializationPlan:
    templates: list[TemplateUnit]
    region_to_template_id: dict[GraphRegion, int]
    parameter_indices: dict[Op, int]


def graph_value_tensor_meta(value: GraphValueRef) -> TensorMeta:
    meta = value.op.tensor_meta
    shape = meta.shape if isinstance(meta, TensorMeta) else meta["shape"]
    dtype = meta.dtype if isinstance(meta, TensorMeta) else meta["dtype"]
    is_multi_result = (
        isinstance(shape, (list, tuple))
        and bool(shape)
        and isinstance(shape[0], (list, tuple))
    )
    if is_multi_result:
        try:
            shape = shape[value.result_index]
        except IndexError as error:
            raise ValueError(
                f"result index {value.result_index} is out of range for "
                f"{value.op.name!r}"
            ) from error
        if isinstance(dtype, (list, tuple)):
            try:
                dtype = dtype[value.result_index]
            except IndexError as error:
                raise ValueError(
                    f"dtype result index {value.result_index} is out of range "
                    f"for {value.op.name!r}"
                ) from error
    elif value.result_index != 0:
        raise ValueError(
            f"operation {value.op.name!r} has no metadata for result "
            f"{value.result_index}"
        )
    elif isinstance(dtype, (list, tuple)):
        if len(dtype) != 1:
            raise ValueError(
                f"operation {value.op.name!r} has ambiguous result dtype"
            )
        dtype = dtype[0]
    return TensorMeta(shape, dtype)


def _validate_v1_region_interface(region: GraphRegion) -> None:
    region_name = f"region {region.kind.name}"
    if any(
        input_ref.kind is RegionInputKind.CONSTANT
        for input_ref in region.interface.ordered_inputs
    ):
        raise ValueError(
            f"{region_name}: external TensorConstant inputs are not supported "
            "by template materialization V1"
        )
    if (
        any(
            input_ref.kind is RegionInputKind.STATE
            for input_ref in region.interface.ordered_inputs
        )
        or region.interface.state_inputs
        or region.interface.state_outputs
    ):
        raise ValueError(
            f"{region_name}: explicit RegionInputKind.STATE and "
            "state_inputs/state_outputs are not supported by template "
            "materialization V1"
        )


def _validate_template_unit(unit: TemplateUnit) -> None:
    if unit.representative not in unit.instances:
        raise ValueError(
            f"template {unit.template_id} representative is not an instance"
        )
    representative = unit.representative.interface
    representative_inputs = representative.ordered_inputs
    representative_outputs = representative.ordered_outputs
    for region in unit.instances:
        interface = region.interface
        if len(interface.ordered_inputs) != len(representative_inputs):
            raise ValueError(
                f"template {unit.template_id} input count mismatch"
            )
        for slot, (actual, expected) in enumerate(
            zip(
                interface.ordered_inputs,
                representative_inputs,
                strict=True,
            )
        ):
            if actual.kind is not expected.kind:
                raise ValueError(
                    f"template {unit.template_id} input {slot} kind mismatch"
                )
            actual_meta = graph_value_tensor_meta(actual.value)
            expected_meta = graph_value_tensor_meta(expected.value)
            if tuple(actual_meta.shape) != tuple(expected_meta.shape):
                raise ValueError(
                    f"template {unit.template_id} input {slot} shape mismatch"
                )
            if actual_meta.dtype != expected_meta.dtype:
                raise ValueError(
                    f"template {unit.template_id} input {slot} dtype mismatch"
                )
        if len(interface.ordered_outputs) != len(representative_outputs):
            raise ValueError(
                f"template {unit.template_id} output count mismatch"
            )
        for slot, (actual, expected) in enumerate(
            zip(
                interface.ordered_outputs,
                representative_outputs,
                strict=True,
            )
        ):
            actual_meta = graph_value_tensor_meta(actual)
            expected_meta = graph_value_tensor_meta(expected)
            if tuple(actual_meta.shape) != tuple(expected_meta.shape):
                raise ValueError(
                    f"template {unit.template_id} output {slot} shape mismatch"
                )
            if actual_meta.dtype != expected_meta.dtype:
                raise ValueError(
                    f"template {unit.template_id} output {slot} dtype mismatch"
                )


def build_template_materialization_plan(
    graph, structure_index, template_index
) -> TemplateMaterializationPlan:
    """Assign stable template ids while retaining original Region instances.

    Template materialization V1 treats KV/cache graph inputs as DATA; explicit
    STATE ABI is reserved for a later version.
    """
    regions = list(structure_index.regions)
    for region in regions:
        _validate_v1_region_interface(region)
    region_set = set(regions)
    group_for_region = {}
    for group in template_index.template_groups:
        if group.representative not in group.instances:
            raise ValueError("template representative is not in instances")
        for region in group.instances:
            if region not in region_set:
                raise ValueError("template instance is not in structure regions")
            if region in group_for_region:
                raise ValueError("region belongs to more than one template group")
            group_for_region[region] = group
    for region in template_index.non_reusable_regions:
        if region not in region_set:
            raise ValueError("non-reusable Region is not in structure regions")
        if region in group_for_region:
            raise ValueError(
                "non-reusable Region also belongs to a template group"
            )

    templates = []
    region_to_template_id = {}
    emitted_groups = set()
    for region in regions:
        group = group_for_region.get(region)
        if group is None:
            instances = (region,)
            representative = region
        else:
            group_key = id(group)
            if group_key in emitted_groups:
                continue
            emitted_groups.add(group_key)
            instances = tuple(item for item in regions if item in group.instances)
            representative = group.representative
        template_id = len(templates)
        unit = TemplateUnit(template_id, representative, instances)
        _validate_template_unit(unit)
        templates.append(unit)
        for instance in instances:
            if instance in region_to_template_id:
                raise ValueError("region maps to more than one template")
            region_to_template_id[instance] = template_id

    if set(region_to_template_id) != region_set:
        raise ValueError("not every Region maps to exactly one Template")

    parameter_indices = {
        param_op: index for index, param_op in enumerate(graph.params)
    }
    return TemplateMaterializationPlan(
        templates, region_to_template_id, parameter_indices
    )

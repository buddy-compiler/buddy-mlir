# ===- template_partitioned_graph_driver.py ----------------------------------
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-------------------------------------------------------------------------

"""Materialize unique Region templates and a statically scheduled wrapper."""

import copy

from buddy_mlir import ir

from .graph import Graph, GraphImporter, NodeType
from .operation import (
    CallOp,
    FuncOp,
    GetItemOp,
    OutputOp,
    PlaceholderOp,
    TensorConstantOp,
)
from .region_analysis import (
    GraphValueRef,
    RegionInputKind,
    RegionInputRef,
    _operand_dict_items,
    _resolve_operand_node_reference,
    iter_op_input_references,
)
from .template_materialization import (
    TemplateMaterializationPlan,
    graph_value_tensor_meta,
)


def _copy_operand_container(value):
    if isinstance(value, list):
        return [_copy_operand_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_operand_container(item) for item in value)
    if isinstance(value, dict):
        return {
            _copy_operand_container(key): _copy_operand_container(item)
            for key, item in value.items()
        }
    return value


class TemplatePartitionedGraphDriver:
    """Independent compile-time template materialization path.

    Template subgraphs represent unique function bodies. Region instances are
    represented by calls in the combined forward.
    """

    def __init__(self, graph, structure_index, plan) -> None:
        self._graph = graph
        self._structure_index = structure_index
        self._plan: TemplateMaterializationPlan = plan
        self._subgraphs = {}
        self._combined_graph = None

    def template_symbol(self, template_id: int) -> str:
        suffix = self._graph._func_name.removeprefix("forward_")
        separator = "" if suffix in ("prefill", "decode") else "_"
        return f"subgraph0_{suffix}{separator}{template_id}"

    @property
    def subgraphs(self):
        return [
            self._subgraphs[unit.template_id]
            for unit in self._plan.templates
            if unit.template_id in self._subgraphs
        ]

    @property
    def combined_graph(self):
        return self._combined_graph

    def _rewrite_operand(
        self,
        value,
        result_index,
        bindings,
        local_nodes,
    ):
        referenced = _resolve_operand_node_reference(
            value, self._graph.node_table
        )
        if referenced is not None:
            if referenced in local_nodes:
                return value, False
            ref = GraphValueRef(referenced, result_index)
            try:
                return bindings[ref], True
            except KeyError as error:
                raise ValueError(
                    f"external value {referenced.name!r}:{result_index} is "
                    "missing from ordered_inputs"
                ) from error
        if isinstance(value, list):
            rewritten = [
                self._rewrite_operand(
                    item, result_index, bindings, local_nodes
                )
                for item in value
            ]
            return [item for item, _ in rewritten], any(
                changed for _, changed in rewritten
            )
        if isinstance(value, tuple):
            rewritten = [
                self._rewrite_operand(
                    item, result_index, bindings, local_nodes
                )
                for item in value
            ]
            return tuple(item for item, _ in rewritten), any(
                changed for _, changed in rewritten
            )
        if isinstance(value, dict):
            rewritten = [
                (
                    self._rewrite_operand(
                        key, result_index, bindings, local_nodes
                    ),
                    self._rewrite_operand(
                        item, result_index, bindings, local_nodes
                    ),
                )
                for key, item in _operand_dict_items(value)
            ]
            return {
                key_value: item_value
                for ((key_value, _), (item_value, _)) in rewritten
            }, any(
                key_changed or item_changed
                for ((_, key_changed), (_, item_changed)) in rewritten
            )
        return value, False

    def _clone_region_node(self, op, bindings, local_nodes):
        clone = copy.copy(op)
        clone._arguments = []
        clone._args_index = []
        for index, value in enumerate(op.args):
            result_index = op._args_index[index] if index < len(op._args_index) else 0
            if isinstance(value, (list, tuple, dict)) and result_index != 0:
                raise ValueError(
                    "nested operands with non-zero result indices are not "
                    "supported by template materialization V1"
                )
            rewritten, changed = self._rewrite_operand(
                value, result_index, bindings, local_nodes
            )
            clone._arguments.append(rewritten)
            clone._args_index.append(0 if changed else result_index)
        rewritten_kwargs, _ = self._rewrite_operand(
            op.kwargs, 0, bindings, local_nodes
        )
        clone._keyword_arguments = rewritten_kwargs
        clone._parents = list(op.parents)
        clone._children = list(op._children)
        clone._tensor_meta = _copy_operand_container(op.tensor_meta)
        return clone

    def build_template_subgraphs(self):
        if self._subgraphs:
            return self.subgraphs
        for unit in self._plan.templates:
            region = unit.representative
            interface = region.interface
            values = [item.value for item in interface.ordered_inputs]
            if len(values) != len(set(values)):
                raise ValueError(
                    f"template {unit.template_id} ordered_inputs contains a "
                    "duplicate GraphValueRef"
                )
            subgraph = Graph(
                self._graph._ops_registry,
                self.template_symbol(unit.template_id),
                self._graph.device,
                verbose=self._graph._verbose,
                verbose_path=self._graph._verbose_path,
            )
            local_nodes = set(region.nodes)
            local_names = {op.name for op in region.nodes}
            bindings = {}
            for slot, input_ref in enumerate(interface.ordered_inputs):
                placeholder = PlaceholderOp()
                placeholder.name = f"__template_arg{slot}"
                if placeholder.name in local_names:
                    raise ValueError(
                        f"template placeholder name {placeholder.name!r} "
                        "conflicts with a Region node"
                    )
                placeholder.tensor_meta = graph_value_tensor_meta(
                    input_ref.value
                )
                subgraph.add_node(placeholder, NodeType.InputNode)
                bindings[input_ref.value] = placeholder.name

            for op in region.nodes:
                subgraph.add_node(
                    self._clone_region_node(op, bindings, local_nodes)
                )

            output = OutputOp()
            output.name = "output"
            for slot, value in enumerate(interface.ordered_outputs):
                if value.result_index == 0:
                    output.add_argument(value.op.name)
                    continue
                getitem = GetItemOp()
                getitem.name = f"__template_result{slot}"
                getitem.add_argument(value.op.name)
                getitem.add_argument(value.result_index)
                subgraph.add_node(getitem)
                output.add_argument(getitem.name)
            subgraph.add_node(output)
            self._subgraphs[unit.template_id] = subgraph
        return self.subgraphs

    def resolve_input(self, input_ref, main_graph, value_map):
        value = input_ref.value
        if input_ref.kind is RegionInputKind.PARAMETER:
            try:
                parameter_index = self._plan.parameter_indices[value.op]
            except KeyError as error:
                raise KeyError(
                    f"parameter {value.op.name!r} has no explicit index"
                ) from error
            parameter = main_graph.params[parameter_index]
            if parameter is not value.op or value.result_index != 0:
                raise ValueError(
                    f"parameter slot {parameter_index} does not match "
                    f"{value.op.name!r}"
                )
            resolved = (parameter.name, 0)
            value_map[value] = resolved
            return resolved
        if input_ref.kind is RegionInputKind.CONSTANT:
            raise ValueError(
                "external TensorConstant inputs are not supported by template "
                "materialization V1"
            )
        if value in value_map:
            return value_map[value]
        if value.op in self._graph.inputs and value.result_index == 0:
            resolved = (value.op.name, 0)
            value_map[value] = resolved
            return resolved
        raise KeyError(
            f"Region input {value.op.name!r}:{value.result_index} has not "
            "been produced"
        )

    def register_outputs(self, region, call_results, value_map) -> None:
        outputs = region.interface.ordered_outputs
        if len(call_results) != len(outputs):
            raise ValueError(
                f"Region {region.kind.name} call result count does not match "
                "ordered_outputs"
            )
        for value, result in zip(outputs, call_results, strict=True):
            if value in value_map:
                raise ValueError(
                    f"Region output {value.op.name!r}:{value.result_index} "
                    "was already registered"
                )
            value_map[value] = result

    def construct_template_combined_main_graph(
        self, do_param_pack=False, output_remap: list[int] | None = None
    ):
        main_graph = Graph(
            self._graph._ops_registry,
            self._graph._func_name,
            verbose=self._graph._verbose,
            verbose_path=self._graph._verbose_path,
        )
        for op in self._graph.params:
            main_graph.add_node(op, NodeType.FakeNode)
        for op in self._graph.inputs:
            main_graph.add_node(op, NodeType.InputNode)

        units = {unit.template_id: unit for unit in self._plan.templates}
        for unit in self._plan.templates:
            func_node = FuncOp()
            func_node.name = self.template_symbol(unit.template_id)
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for item in unit.representative.interface.ordered_inputs:
                func_node.add_argument(graph_value_tensor_meta(item.value))
            for value in unit.representative.interface.ordered_outputs:
                meta = graph_value_tensor_meta(value)
                func_node.tensor_meta["shape"].append(meta.shape)
                func_node.tensor_meta["dtype"].append(meta.dtype)
            main_graph.add_node(func_node)

        value_map = {
            GraphValueRef(op): (op.name, 0) for op in self._graph.inputs
        }
        for call_index, region in enumerate(self._structure_index.regions):
            try:
                template_id = self._plan.region_to_template_id[region]
                unit = units[template_id]
            except KeyError as error:
                raise KeyError("Region has no materialized template") from error
            call = CallOp()
            call.name = f"template_call{call_index}"
            call.call_func_name = self.template_symbol(template_id)
            call.tensor_meta = {"shape": [], "dtype": []}
            for input_ref in region.interface.ordered_inputs:
                operand_name, operand_index = self.resolve_input(
                    input_ref, main_graph, value_map
                )
                call.add_argument(operand_name, operand_index)
            expected_operands = len(
                unit.representative.interface.ordered_inputs
            )
            if len(call.args) != expected_operands:
                raise ValueError(
                    f"template {template_id} call operand count mismatch"
                )
            for value in region.interface.ordered_outputs:
                meta = graph_value_tensor_meta(value)
                call.tensor_meta["shape"].append(meta.shape)
                call.tensor_meta["dtype"].append(meta.dtype)
            call_results = [
                (call.name, index)
                for index in range(len(call.tensor_meta["shape"]))
            ]
            if len(call_results) != len(region.interface.ordered_outputs):
                raise ValueError(
                    f"template {template_id} call result count mismatch"
                )
            main_graph.add_node(call)
            self.register_outputs(region, call_results, value_map)

        final_outputs = []
        for original_output in self._graph.body:
            if not isinstance(original_output, OutputOp):
                continue
            final_outputs.extend(
                iter_op_input_references(
                    original_output, self._graph.node_table
                )
            )
        resolved_outputs = []
        for value in final_outputs:
            if value.op in self._plan.parameter_indices:
                kind = RegionInputKind.PARAMETER
            elif isinstance(value.op, TensorConstantOp):
                kind = RegionInputKind.CONSTANT
            else:
                kind = RegionInputKind.DATA
            resolved_outputs.append(
                self.resolve_input(
                    RegionInputRef(kind, value), main_graph, value_map
                )
            )
        if output_remap is not None:
            if len(output_remap) != len(resolved_outputs):
                raise ValueError(
                    "output_remap length must match the number of graph outputs"
                )
            if any(
                index < 0 or index >= len(resolved_outputs)
                for index in output_remap
            ):
                raise ValueError("output_remap contains an invalid output index")
            resolved_outputs = [
                resolved_outputs[index] for index in output_remap
            ]

        output = OutputOp()
        output.name = "output"
        getitem_index = 0
        for producer_name, producer_index in resolved_outputs:
            if producer_index == 0:
                output.add_argument(producer_name)
                continue
            getitem = GetItemOp()
            getitem.name = f"template_getitem{getitem_index}"
            getitem_index += 1
            getitem.add_argument(producer_name)
            getitem.add_argument(producer_index)
            main_graph.add_node(getitem)
            output.add_argument(getitem.name)
        main_graph.add_node(output)
        self._combined_graph = main_graph

        with ir.Location.unknown(ir.Context()):
            importer = GraphImporter(
                main_graph.body,
                main_graph.params_shapes,
                main_graph.inputs_shapes,
                main_graph._func_name,
                main_graph._ops_registry,
                do_param_pack,
                verbose=main_graph._verbose,
                verbose_path=main_graph._verbose_path,
            )
            return importer.import_main_graph()

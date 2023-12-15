# ===- graph_importer.py -------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the graph importer for Buddy Compiler's graph.
#
# ===---------------------------------------------------------------------------

import operator
from typing import Any, List, Optional
import functools

import mlir.dialects.func as func
import mlir.ir as ir
import torch


class FXGraphImporter:
    """
    Imports an FX graph and generates an MLIR module in high-level dialects.

    Attributes:
        _symbol_table (dict): A dictionary to keep track of the symbols.
        _gm (torch.fx.GraphModule): The FX graph module to be imported.
        _func_name (str): Name of the generated MLIR function.
        _inputs (List[torch.Tensor]): Input tensor(s) of the FX graph.
        _num_input_visited (int): Number of input nodes that have been visited.
        _module (mlir.ir.Module): The generated MLIR module.
        _ops_registry (dict): Registry for the candidate operations.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        params: List[torch.Tensor],
        inputs: List[torch.Tensor],
        do_param_pack: bool = False,
        func_name: str = "forward",
        ops_registry: Optional[dict] = None,
    ):
        """
        Initializes the FX Graph importer.

        Args:
            gm (torch.fx.GraphModule): The FX graph that will be imported.
            inputs (List[torch.Tensor]): Input tensor(s) of the FX graph.
            func_name (str): Name of the generated MLIR function.
            ops_registry (dict): Registry for the candidate operations.
        """
        if ops_registry is None:
            ops_registry = {}
        self._symbol_table = {}
        self._gm = gm
        self._func_name = func_name
        self._params = params
        self._inputs = inputs
        self._do_param_pack = do_param_pack
        self._param_packs = []
        self._num_input_visited = 0
        self._module = ir.Module.create()
        self._ops_registry = ops_registry
        self._current_param_pack_offset = None

    def _torch_dtype_to_mlir_dtype(self, dtype: torch.dtype) -> ir.Type:
        """
        Converts a torch dtype to the corresponding MLIR dtype.

        Args:
            dtype (torch.dtype): The torch data type.

        Returns:
            mlir.ir.Type: The corresponding MLIR data type.

        Raises:
            NotImplementedError: If the given dtype is not supported.
        """
        match dtype:
            case torch.int32:
                return ir.IntegerType.get_signless(32)
            case torch.int64:
                return ir.IntegerType.get_signless(64)
            case torch.float32:
                return ir.F32Type.get()
            case torch.bool:
                return ir.IntegerType.get_signless(1)
            case _:
                raise NotImplementedError(f"Unsupported dtype {dtype}")

    def _pack_params(self) -> None:
        dtypes = list(set([param.dtype for param in self._params]))
        dtypes.sort(key=str)
        self._current_param_pack_offset = {dtype: 0 for dtype in dtypes}
        for dtype in dtypes:
            params_of_dtype = [
                param for param in self._params if param.dtype == dtype
            ]
            param_total_size = 0
            for param in params_of_dtype:
                param_total_size += functools.reduce(
                    lambda x, y: x * y, list(param.shape)
                )
            mlir_dtype = self._torch_dtype_to_mlir_dtype(dtype)
            self._param_packs.append(
                ir.RankedTensorType.get([param_total_size], mlir_dtype)
            )

    def import_graph(self) -> ir.Module:
        """
        Imports FX graph and generates an MLIR module in high-level dialects.

        Returns:
            mlir.ir.Module: An MLIR module in high-level dialects.
        """
        with ir.InsertionPoint(self._module.body):
            arguments = []
            if self._do_param_pack:
                self._pack_params()
                arguments.extend(self._param_packs)
                inputs = self._inputs
            else:
                inputs = self._params + self._inputs
            for arg in inputs:
                shape_list = list(arg.shape)
                torch_dtype = arg.dtype
                mlir_dtype = self._torch_dtype_to_mlir_dtype(torch_dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                for node in self._gm.graph.nodes:
                    if not (
                        node.op in ["output", "placeholder", "call_function"]
                        or node.target is operator.getitem
                    ):
                        continue
                    if node.op == "output":
                        output_node_args = node.args[0]
                        returns = [
                            self._symbol_table.get((str(output_arg), 0))
                            for output_arg in output_node_args
                        ]
                        self._symbol_table[("output", 0)] = returns
                    elif node.op == "placeholder":
                        self._import_placeholder(node, args_list)
                    elif node.target is operator.getitem:
                        self._symbol_table[
                            (str(node.name), 0)
                        ] = self._symbol_table[
                            (str(node.args[0]), node.args[1])
                        ]
                    else:
                        self._import_op(node)

                return self._symbol_table.get(("output", 0))

        return self._module

    def _import_placeholder(
        self, node: torch.fx.Node, args_list: List[ir.BlockArgument]
    ):
        """
        Imports a placeholder node from the FX graph.

        Args:
            node (torch.fx.Node): The FX node representing the placeholder.
            args_list (List[mlir.ir.BlockArgument]): List of input tensors.
        """
        if self._num_input_visited < len(self._params) and self._do_param_pack:
            dtype = node.meta["tensor_meta"].dtype
            pack_of_dtype = None
            for pack in args_list:
                if ir.RankedTensorType(
                    pack.type
                ).element_type == self._torch_dtype_to_mlir_dtype(dtype):
                    pack_of_dtype = pack
                    break
            placeholder_name = self._ops_registry["param.extract"](
                node, self._current_param_pack_offset[dtype], pack_of_dtype
            ).result
            self._current_param_pack_offset[dtype] += functools.reduce(
                lambda x, y: x * y, list(node.meta["tensor_meta"].shape)
            )
        elif self._do_param_pack:
            if len(self._params) > 0:
                placeholder_name = args_list[
                    self._num_input_visited
                    - len(self._params)
                    + len(self._param_packs)
                ]
            else:
                placeholder_name = args_list[self._num_input_visited]
        else:
            placeholder_name = args_list[self._num_input_visited]

        self._symbol_table[(str(node.name), 0)] = placeholder_name
        self._num_input_visited += 1

    def _import_op(self, node: torch.fx.Node):
        """
        Imports an operation node from the FX graph.

        Args:
            node (torch.fx.Node): The FX node representing the operation.

        """
        op_name = node.target.__name__
        op_ret: ir.Operation | ir.Value | tuple | ir.OpResult = (
            self._ops_registry[op_name](node, self._symbol_table)
        )
        if isinstance(op_ret, tuple):
            for i, operation in enumerate(op_ret):
                self._symbol_table[(str(node.name), i)] = operation.result
        elif isinstance(op_ret, ir.OpResult):
            self._symbol_table[(str(node.name), 0)] = op_ret
        else:
            self._symbol_table[(str(node.name), 0)] = op_ret.result

    def get_output_nodes(self):
        """
        Get output nodes from the lowered mlir func.
        """
        return self._symbol_table.get(("output", 0))

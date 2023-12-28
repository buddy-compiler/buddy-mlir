# ===- graph.py ----------------------------------------------------------------
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
# This is the graph level of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

import mlir.ir as ir
import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt

import os
from typing import Any, List, Optional
import ctypes
from enum import Enum
import functools
import numpy as np
import torch

from .op_def import *


class Tensordtype(Enum):
    """
    Enum class for declare tensor dtype.
    """
    Int32 = "int32"
    Int64 = "int64"
    Float32 = "float32"
    Bool = "bool"


class TensorMeta:
    """
    Store tensor's shape and dtype, overlook tensor's raw data.
    """
    def __init__(self, shape, dtype) -> None:
        self.shape = shape
        self.dtype = dtype


def make_output_memref_descriptor(ranks, dtypes):
    """
    Make output memref descriptor for the given memref ranks and dtypes.
    """
    memref_descriptor = []
    for i, rank, dtype in zip(range(len(ranks)), ranks, dtypes):
        memref_descriptor.append(
            (str(i), rt.make_nd_memref_descriptor(rank, dtype))
        )

    class OutputDescriptor(ctypes.Structure):
        """Builds an output struct descriptor for the multi memref."""

        _fields_ = memref_descriptor

    return OutputDescriptor


class Graph:
    """
    Graph is graph level expression of the frontends of Buddy Compiler.
    Graph acts as a model compute graph for the Buddy Compiler frontends,
    which converts an Graph into an equivalent MLIR module.

    Attributes:
        body: The op sequence of graph.
        inputs: The model inputs.
        params: The model params.
        outputs: The mlir func's outputs.
        device: The hardware for graph runtime.
        imported_module: The imported MLIR module after compilation.
        ops_registry: The ops lower strategy for graph.
        func_name: The func name for mlir module.
        ctx: The context of mlir module.
        output_memref: Memref pointer in mlir func output.
    """

    def __init__(
        self,
        inputs: List[TensorMeta],
        fake_params: List[TensorMeta],
        ops_registry: dict,
        func_name: str,
    ) -> None:
        """
        Initializes the Graph.

        Args:
            fx_graph: The torch fx graph to be lowered.
            inputs: The torch fx graph's inputs.
            params_flat: The real params of the torch fx graph.
        """
        self._body = []
        self._inputs = inputs
        self._fake_params = fake_params
        self._outputs = None
        self.device = "cpu"
        self._imported_module = None
        self._ops_registry = ops_registry
        self._func_name = func_name
        self._ctx = ir.Context()
        self._output_memref = None
        self._output_descriptor = None
        self.ee_ = None

    def add_node(self, node: Op):
        self._body.append(node)

    def lower_to_top_level_ir(self, do_params_pack=False):
        """
        Lower graph to top level mlir dialects.
        """
        with ir.Location.unknown(self._ctx):
            fx_importer = GraphImporter(
                self._body,
                self._fake_params,
                self._inputs,
                do_params_pack,
                self._func_name,
                self._ops_registry,
            )
            self._imported_module = fx_importer.import_graph()
            outputs = fx_importer.get_output_nodes()
        self._output_memref = []
        output_ranks = []
        output_dtypes = []
        for out_node in outputs:
            out_type = ir.RankedTensorType(out_node.type)
            shape = list(out_type.shape)
            dtype = out_type.element_type
            match str(dtype):
                case "i1":
                    np_type = np.dtype(np.bool_)
                case "i32":
                    np_type = np.dtype(np.int32)
                case "i64":
                    np_type = np.dtype(np.int64)
                case "f32":
                    np_type = np.dtype(np.float32)
            self._output_memref.append(
                ctypes.pointer(
                    ctypes.pointer(
                        rt.make_nd_memref_descriptor(
                            len(shape), rt.as_ctype(np_type)
                        )()
                    )
                )
            )
            output_ranks.append(len(shape))
            output_dtypes.append(rt.as_ctype(np_type))
        self._output_descriptor = make_output_memref_descriptor(
            output_ranks, output_dtypes
        )

    def lower_to_llvm_ir(self):
        """
        Lower graph to llvm ir.
        """
        if self._imported_module is None:
            self.lower_to_top_level_ir()
        with ir.Location.unknown(self._ctx):
            pm = PassManager("builtin.module")
            pm.add("func.func(tosa-to-linalg-named)")
            pm.add("func.func(tosa-to-linalg)")
            pm.add("func.func(tosa-to-tensor)")
            pm.add("func.func(tosa-to-arith)")
            pm.run(self._imported_module.operation)
            pm.add("arith-expand")
            pm.add("eliminate-empty-tensors")
            pm.add("empty-tensor-to-alloc-tensor")
            pm.add("convert-elementwise-to-linalg")
            pm.add("func.func(linalg-bufferize)")
            pm.add("func.func(convert-linalg-to-affine-loops)")
            pm.add("affine-loop-fusion")
            pm.add("func.func(affine-parallelize)")
            pm.add("lower-affine")
            pm.add("convert-scf-to-openmp")
            pm.add("func-bufferize")
            pm.add("arith-bufferize")
            pm.add("func.func(tensor-bufferize)")
            pm.add("func.func(buffer-deallocation)")
            pm.add("func.func(finalizing-bufferize)")
            pm.add("expand-strided-metadata")
            pm.add("convert-vector-to-llvm")
            pm.add("memref-expand")
            pm.add("arith-expand")
            pm.add("convert-arith-to-llvm")
            pm.add("finalize-memref-to-llvm")
            pm.add("convert-scf-to-cf")
            pm.add("func.func(llvm-request-c-wrappers)")
            pm.add("convert-openmp-to-llvm")
            pm.add("convert-math-to-llvm")
            pm.add("convert-math-to-libm")
            pm.add("convert-func-to-llvm")
            pm.add("reconcile-unrealized-casts")
            pm.run(self._imported_module.operation)

    def compile(self):
        """
        Compile graph from torch fx graph to llvm ir.
        """
        self.lower_to_top_level_ir(True)
        print(self._imported_module, flush=True)
        self.lower_to_llvm_ir()

    def dynamo_run(self):
        """
        Construct mlir runtime function for dynamo forward
        """
        self.compile()
        path_prefix = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(path_prefix, "../../../../../llvm/build/lib")
        lib_path = os.path.abspath(lib_path)
        shared_libs = [
            str(lib_path) + "/libmlir_runner_utils.so",
            str(lib_path) + "/libmlir_c_runner_utils.so",
            str(lib_path) + "/libomp.so",
        ]
        self.ee_ = ExecutionEngine(
            self._imported_module, opt_level=3, shared_libs=shared_libs
        )

        def cast_c_ptr(outdata_ptr, memref_ptr):
            outdata_addr = ctypes.addressof(outdata_ptr.contents)
            out_ptr = ctypes.cast(outdata_addr, type(memref_ptr))
            return out_ptr

        def move_c_ptr(outdata_ptr, memref_ptr):
            elem_size = ctypes.sizeof(memref_ptr.contents)
            outdata_addr = ctypes.addressof(outdata_ptr.contents)
            out_ptr = ctypes.cast(outdata_addr + elem_size, type(memref_ptr))
            return out_ptr

        def exec_buddy_graph(*args):
            input_memref = [
                ctypes.pointer(
                    ctypes.pointer(
                        rt.get_ranked_memref_descriptor(tensor.numpy())
                    )
                )
                for tensor in args
            ]
            output_memref = self._output_descriptor()
            input_memref = [
                ctypes.pointer(ctypes.pointer(output_memref))
            ] + input_memref
            self.ee_.invoke(self._func_name, *input_memref)
            output_tensor = []
            outdata_ptr = input_memref[0][0]
            for output_ptr in self._output_memref:
                data_ptr = cast_c_ptr(outdata_ptr, output_ptr[0])
                output_tensor.append(rt.ranked_memref_to_numpy(data_ptr))
                outdata_ptr = move_c_ptr(outdata_ptr, output_ptr[0])
            return [torch.from_numpy(tensor) for tensor in output_tensor]

        return exec_buddy_graph


class GraphImporter:
    """
    Imports an buddy graph and generates an MLIR module in high-level dialects.

    Attributes:
        _symbol_table (dict): A dictionary to keep track of the symbols.
        _gm (List[Op]): The FX graph module to be imported.
        _func_name (str): Name of the generated MLIR function.
        _inputs (List[TensorMeta]): Input tensor(s) of the FX graph.
        _num_input_visited (int): Number of input nodes that have been visited.
        _module (mlir.ir.Module): The generated MLIR module.
        _ops_registry (dict): Registry for the candidate operations.
    """

    def __init__(
        self,
        body: List[Op],
        params: List[TensorMeta],
        inputs: List[TensorMeta],
        do_param_pack: bool,
        func_name: str,
        ops_registry: dict,
    ):
        """
        Initializes the buddy Graph importer.

        Args:
            gm (Graph): The buddy graph that will be imported.
            inputs (List[TensorMeta]): Input tensor(s) of the buddy graph.
            func_name (str): Name of the generated MLIR function.
            ops_registry (dict): Registry for the candidate operations.
        """
        if ops_registry is None:
            ops_registry = {}
        self._symbol_table = {}
        self._gm = body
        self._func_name = func_name
        self._params = params
        self._inputs = inputs
        self._do_param_pack = do_param_pack
        self._param_packs = []
        self._num_input_visited = 0
        self._module = ir.Module.create()
        self._ops_registry = ops_registry
        self._current_param_pack_offset = None

    def _str_to_mlir_dtype(self, dtype: str) -> ir.Type:
        """
        Converts a str to the corresponding MLIR dtype.

        Args:
            dtype (str): The tensor type.

        Returns:
            mlir.ir.Type: The corresponding MLIR data type.

        Raises:
            NotImplementedError: If the given dtype is not supported.
        """
        match dtype:
            case Tensordtype.Int32:
                return ir.IntegerType.get_signless(32)
            case Tensordtype.Int64:
                return ir.IntegerType.get_signless(64)
            case Tensordtype.Float32:
                return ir.F32Type.get()
            case Tensordtype.Bool:
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
            mlir_dtype = self._str_to_mlir_dtype(dtype)
            self._param_packs.append(
                ir.RankedTensorType.get([param_total_size], mlir_dtype)
            )

    def import_graph(self) -> ir.Module:
        """
        Imports buddy graph and generates an MLIR module in high-level dialects.

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
                dtype = arg.dtype
                mlir_dtype = self._str_to_mlir_dtype(dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                for node in self._gm:
                    # if not (
                    #     node.op in ["output", "placeholder", "call_function"]
                    #     or node.target is operator.getitem
                    # ):
                    #     continue
                    if isinstance(node, OutputOp):
                        output_node_args = node.args
                        returns = [
                            self._symbol_table.get((str(output_arg), 0))
                            for output_arg in output_node_args
                        ]
                        self._symbol_table[("output", 0)] = returns
                    elif isinstance(node, PlaceholderOp):
                        self._import_placeholder(node, args_list)
                    elif isinstance(node, GetItemOp):
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
        self, node: PlaceholderOp, args_list: List[ir.BlockArgument]
    ):
        """
        Imports a placeholder node from the FX graph.

        Args:
            node (PlaceholderOp): The FX node representing the placeholder.
            args_list (List[mlir.ir.BlockArgument]): List of input tensors.
        """
        if self._num_input_visited < len(self._params) and self._do_param_pack:
            dtype = node.tensor_meta["dtype"]
            pack_of_dtype = None
            for pack in args_list:
                if ir.RankedTensorType(
                    pack.type
                ).element_type == self._str_to_mlir_dtype(dtype):
                    pack_of_dtype = pack
                    break
            placeholder_name = self._ops_registry["param.extract"](
                node, self._current_param_pack_offset[dtype], pack_of_dtype
            ).result
            self._current_param_pack_offset[dtype] += functools.reduce(
                lambda x, y: x * y, list(node.tensor_meta["shape"])
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

    def _import_op(self, node: Op):
        """
        Imports an operation node from the buddy graph.

        Args:
            node (Op): The buddy node representing the operation.

        """
        op_name = node.__class__.__name__
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

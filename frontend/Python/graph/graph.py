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
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt

import os
import ctypes
import numpy as np
import torch

from .graph_importer import FXGraphImporter


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
        self, fx_graph, inputs, params_flat, ops_registry, func_name
    ) -> None:
        """
        Initializes the Graph.

        Args:
            fx_graph: The torch fx graph to be lowered.
            inputs: The torch fx graph's inputs.
            params_flat: The real params of the torch fx graph.
        """
        self._body = fx_graph
        self._inputs = inputs
        self._params = params_flat
        self._outputs = None
        self.device = "cpu"
        self._imported_module = None
        self._ops_registry = ops_registry
        self._func_name = func_name
        self._ctx = ir.Context()
        self._output_memref = None
        self._output_descriptor = None
        self.ee_ = None

    def lower_to_top_level_ir(self, do_params_pack=False):
        """
        Lower graph to top level mlir dialects.
        """
        with ir.Location.unknown(self._ctx):
            fx_importer = FXGraphImporter(
                self._body,
                self._params,
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
        self.lower_to_top_level_ir()
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

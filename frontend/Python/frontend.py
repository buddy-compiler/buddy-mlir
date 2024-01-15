# ===- frontend.py -------------------------------------------------------------
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
# This is the entry of the Buddy Compiler frontend.
# 
# TODO[Low]: When integrating more frameworks, `frontend.py` acts as a unified
# entry and driver, separating out compilers/importers for various platforms 
# (e.g. DynamoCompiler).
#
# ===---------------------------------------------------------------------------

from typing import Any, List, Optional
import operator
import os
import ctypes
import platform

import mlir.ir as ir
import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt
import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree

from .ops.linalg import ops_registry as linalg_ops_registry
from .ops.tosa import ops_registry as tosa_ops_registry
from .ops.math import ops_registry as math_ops_registry
from .graph import Graph, TensorDType, TensorMeta
from .graph.operation import *


class DynamoCompiler:
    """
    Dynamo Compiler is one of the frontends of Buddy Compiler.
    Dynamo Compiler acts as a custom compiler for the TorchDynamo framework,
    which converts an FX Graph into an equivalent Buddy Graph and MLIR module.

    Attributes:
        imported_graphs: The imported graphs.
        imported_params: The imported parameters from the model.
    """

    def __init__(
        self,
        func_name: str = "forward",
        primary_registry: Optional[dict] = None,
        aot_autograd_decomposition: Optional[dict] = None,
    ) -> None:
        # TODO: Update docstring.
        """
        Initializes the Dynamo Compiler.

        Args:
            func_name (str, optional): The function name to be used.
            primary_registry (dict, optional): The primary operations registry.
            aot_autograd_decomposition (Optional[dict], optional):
            The ahead-of-time autograd decomposition dictionary.
        """
        if primary_registry is None:
            primary_registry = {}
        self._func_name = func_name
        self._aot_autograd_decomposition = aot_autograd_decomposition
        self._imported_graphs = []
        self._ops_registry = {}
        self._imported_params = {}
        self._ops_registry.update(math_ops_registry)
        self._ops_registry.update(linalg_ops_registry)
        self._ops_registry.update(tosa_ops_registry)
        self._ops_registry.update(primary_registry)
        self._ops_map = {
            "output": OutputOp,
            "placeholder": PlaceholderOp,
            "arange.start": ArangeOp,
            "arange.default": ArangeOp,
            "unsqueeze.default": UnsqueezeOp,
            "view.default": ViewOp,
            "ones.default": OnesOp,
            "full.default": FullOp,
            "lt.Tensor": LessThanOp,
            "embedding.default": EmbeddingOp,
            "masked_fill.Scalar": MaskedFillOp,
            "slice.Tensor": SliceOp,
            "expand.default": ExpandOp,
            "_to_copy.default": ToCopyOp,
            "rsub.Scalar": RsubOp,
            "pow.Tensor_Scalar": PowOp,
            "mean.dim": MeanOp,
            "rsqrt.default": RsqrtOp,
            "mul.Tensor": MulOp,
            "t.default": TOp,
            "mm.default": MatmulOp,
            "transpose.int": TransposeOp,
            "index.Tensor": IndexOp,
            "neg.default": NegOp,
            "cat.default": CatOp,
            "squeeze.dim": SqueezeOp,
            "bmm.default": BatchMatmulOp,
            "div.Tensor": DivOp,
            "_softmax.default": SoftmaxOp,
            "clone.default": CloneOp,
            "silu.default": SiluOp,
            "add.Tensor": AddOp,
            "addmm.default": AddMMOp,
            "permute.default": PermuteOp,
            "convert_element_type.default": ConvertElementTypeOp,
            "sum.dim_IntList": SumDimOp,
            "tanh.default": TanhOp,
            "sub.Tensor": SubOp,
            "var_mean.correction": VarMeanOp,
            "amax.default": AmaxOp,
            "select.int": SelectOp,
            "exp.default": ExpOp,
            "erf.default": ErfOp,
            "getitem": GetItemOp,
        }

    @property
    def imported_graphs(self):
        """Returns the imported buddy graphs after compilation."""
        return self._imported_graphs

    @property
    def imported_params(self):
        """Returns the imported model params after compilation."""
        return self._imported_params

    def _torch_dtype_translate(self, dtype):
        match dtype:
            case "torch.int64":
                return TensorDType.Int64
            case "torch.int32":
                return TensorDType.Int32
            case "torch.float32":
                return TensorDType.Float32
            case "torch.bool":
                return TensorDType.Bool

    def _create_node(
        self,
        gm_node_name: str,
        node_name: str,
        node_input: Tuple,
        node_output_shape: list = [],
        node_output_dtype: TensorDType = None,
        node_kwargs: Optional[Dict] = None,
    ):
        # TODO: Add docstring.
        op_class = self._ops_map.get(gm_node_name)
        buddy_node = op_class()
        buddy_node._name = node_name
        if gm_node_name == "output":
            for input_arg in node_input[0]:
                buddy_node.add_argument(str(input_arg))
            return buddy_node
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        if node_kwargs is None:
            node_kwargs = {}
        buddy_node._keyword_arguments.update(node_kwargs)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

    def _compile_fx(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ) -> Any:
        # TODO: Update docstring.
        """
        Compiles the provided FX Graph to Buddy Graph and MLIR module.

        Args:
            gm (torch.fx.GraphModule): The GraphModule to be compiled.
            inputs (List[torch.Tensor]): The input tensors.

        Returns:
            Any: The result of the ahead-of-time compiled module.
        """

        def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):
            """Compile a FX graph in Aten/Prims IR to MLIR."""
            # TODO: source of the params_flat.
            nonlocal params_flat
            func_inputs = []
            for inp in _inputs[len(params_flat) :]:
                inp_shape = inp.shape
                inp_dtype = self._torch_dtype_translate(str(inp.dtype))
                func_inputs.append(TensorMeta(inp_shape, inp_dtype))
            fake_params = []
            for param in params_flat:
                param_dtype = self._torch_dtype_translate(str(param.dtype))
                fake_params.append(TensorMeta(param.shape, param_dtype))
            graph = Graph(
                func_inputs,
                fake_params,
                self._ops_registry,
                self._func_name,
            )
            for gm_node in _gm.graph.nodes:
                if gm_node.op == "placeholder":
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = self._create_node(
                        gm_node.op,
                        gm_node.name,
                        gm_node.args,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )

                elif gm_node.op == "output":
                    buddy_node = self._create_node(
                        gm_node.op,
                        gm_node.name,
                        gm_node.args,
                    )

                elif gm_node.target is operator.getitem:
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )

                else:
                    tensor_meta = gm_node.meta.get("tensor_meta")
                    val = gm_node.meta.get("val")
                    num_returns = len(gm_node.target._schema.returns)
                    if num_returns == 1:
                        node_dtype = self._torch_dtype_translate(
                            str(tensor_meta.dtype)
                        )
                        node_shape = tensor_meta.shape
                    elif num_returns > 1:
                        node_dtype = tuple(
                            [
                                self._torch_dtype_translate(val_item.dtype)
                                for val_item in val
                            ]
                        )
                        node_shape = tuple([val_item.shape for val_item in val])
                    else:
                        raise RuntimeError("Zero returns is not supported.")

                    buddy_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        node_shape,
                        node_dtype,
                        node_kwargs=gm_node.kwargs,
                    )

                graph.add_node(buddy_node)
            self._imported_graphs.append(graph)
            self._imported_params[graph] = params_flat
            return self.dynamo_run()

        params = {
            **dict(gm.named_parameters(remove_duplicate=False)),
            **dict(gm.named_buffers(remove_duplicate=False)),
        }
        params_flat, _ = pytree.tree_flatten(params)
        return aot_module_simplified(
            gm,
            inputs,
            fw_compiler=_compiler,
            decompositions=self._aot_autograd_decomposition,
        )

    def __call__(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ) -> Any:
        # TODO: Update docstring.
        """
        A callable method that wraps around the `_compile_fx` method.

        Args:
            gm (torch.fx.GraphModule): The GraphModule to be compiled.
            inputs (List[torch.Tensor]): The input tensors.

        Returns:
            Any: The result of the ahead-of-time compiled module.
        """
        return self._compile_fx(gm, inputs)

    def importer(self, model, *args, **kwargs) -> List[Graph]:
        """
        Imports the provided model as MLIR module and flat parameters.

        Args:
            model: The model to be imported.
            args: Arguments for the model.
            kwargs: Keyword arguments for the model.

        Returns:
            imported_graphs: The imported buddy graphs.
        """
        model_opt = dynamo.optimize(self._compile_fx)(model)
        model_opt(*args, **kwargs)
        return self._imported_graphs

    def dynamo_run(self):
        # TODO: Add docstring.
        def get_lib_extension():
            if platform.system() == "Linux":
                return ".so"
            elif platform.system() == "Darwin":
                return ".dylib"
            else:
                raise RuntimeError("Unsupported platform")

        # TODO：why imported graph is a list?
        graph = self._imported_graphs[0]
        graph.compile()
        # Collect dependency libraries.
        lib_extension = get_lib_extension()
        lib_names = ["libmlir_runner_utils", "libmlir_c_runner_utils", "libomp"]
        path_prefix = os.path.dirname(os.path.abspath(__file__))
        lib_base_path = os.path.join(path_prefix, "../../../../llvm/build/lib/")
        lib_base_path = os.path.abspath(lib_base_path)
        shared_libs = [
            os.path.join(lib_base_path, lib_name + lib_extension)
            for lib_name in lib_names
        ]
        # Define execution engine.
        ee = ExecutionEngine(
            graph._imported_module, opt_level=3, shared_libs=shared_libs
        )

        # TODO: Add comment for the memref and pointers.
        def cast_c_ptr(outdata_ptr, memref_ptr):
            outdata_addr = ctypes.addressof(outdata_ptr.contents)
            out_ptr = ctypes.cast(outdata_addr, type(memref_ptr))
            return out_ptr

        # TODO: Add comment for the memref and pointers.
        def move_c_ptr(outdata_ptr, memref_ptr):
            elem_size = ctypes.sizeof(memref_ptr.contents)
            outdata_addr = ctypes.addressof(outdata_ptr.contents)
            out_ptr = ctypes.cast(outdata_addr + elem_size, type(memref_ptr))
            return out_ptr

        # TODO: Add the call specifications for TorchDynamo.
        def exec_buddy_graph(*args):
            # TODO: Add comment for the memref and pointers.
            input_memref = [
                ctypes.pointer(
                    ctypes.pointer(
                        rt.get_ranked_memref_descriptor(tensor.numpy())
                    )
                )
                for tensor in args
            ]
            output_memref = [
                ctypes.pointer(ctypes.pointer(graph._output_descriptor()))
            ]
            args_memref = output_memref + input_memref
            # TODO: Add invoke function description.
            ee.invoke(graph._func_name, *args_memref)
            # TODO: Add output tensor description.
            output_tensor = []
            outdata_ptr = args_memref[0][0]
            for output_ptr in graph._output_memref:
                data_ptr = cast_c_ptr(outdata_ptr, output_ptr[0])
                output_tensor.append(rt.ranked_memref_to_numpy(data_ptr))
                outdata_ptr = move_c_ptr(outdata_ptr, output_ptr[0])
            return [torch.from_numpy(tensor) for tensor in output_tensor]

        return exec_buddy_graph

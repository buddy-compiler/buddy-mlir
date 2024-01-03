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
# ===---------------------------------------------------------------------------

from typing import Any, List, Optional
import operator

import mlir.ir as ir
import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree

from .ops.linalg import ops_registry as linalg_ops_registry
from .ops.tosa import ops_registry as tosa_ops_registry
from .graph import Graph, Tensordtype, TensorMeta
from .frontend_ops_map import torch_ops_map


class DynamoCompiler:
    """
    Dynamo Compiler is one of the frontends of Buddy Compiler.
    Dynamo Compiler acts as a custom compiler for the Torch Dynamo framework,
    which converts an FX Graph into an equivalent MLIR module.

    Attributes:
        imported_module: The imported MLIR module after compilation.
        imported_params: The imported parameters from the model.
    """

    def __init__(
        self,
        func_name: str = "forward",
        primary_registry: Optional[dict] = None,
        aot_autograd_decomposition: Optional[dict] = None,
    ) -> None:
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
        # self._ops_registry.update(math_ops_registry)
        # self._ops_registry.update(linalg_ops_registry)
        # self._ops_registry.update(primary_registry)
        self._ops_registry.update(linalg_ops_registry)
        self._ops_registry.update(tosa_ops_registry)

    @property
    def imported_graphs(self):
        """Returns the imported buddy graphs after compilation."""
        return self._imported_graphs

    @property
    def imported_params(self):
        """Returns the imported model params after compilation."""
        return self._imported_params

    def torch_dtype_to_str(self, dtype):
        match dtype:
            case "torch.int64":
                return Tensordtype.Int64
            case "torch.int32":
                return Tensordtype.Int32
            case "torch.float32":
                return Tensordtype.Float32
            case "torch.bool":
                return Tensordtype.Bool

    def _compile_fx(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ) -> Any:
        """
        Compiles the provided FX Graph to MLIR module.

        Args:
            gm (torch.fx.GraphModule): The GraphModule to be compiled.
            inputs (List[torch.Tensor]): The input tensors.

        Returns:
            Any: The result of the ahead-of-time compiled module.
        """

        def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):
            """Compile a FX graph in Aten/Prims IR to MLIR."""
            nonlocal params_flat
            func_inputs = []
            for inp in _inputs[len(params_flat) :]:
                inp_shape = inp.shape
                inp_dtype = self.torch_dtype_to_str(str(inp.dtype))
                func_inputs.append(TensorMeta(inp_shape, inp_dtype))
            fake_params = []
            for param in params_flat:
                param_dtype = self.torch_dtype_to_str(str(param.dtype))
                fake_params.append(TensorMeta(param.shape, param_dtype))
            graph = Graph(
                func_inputs,
                fake_params,
                self._ops_registry,
                self._func_name,
            )
            for node in _gm.graph.nodes:
                if node.op == "placeholder":
                    node_dtype = self.torch_dtype_to_str(
                        str(node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = torch_ops_map[node.op].fx_create_node(
                        node.name,
                        node.args,
                        node.users.keys(),
                        node.meta["tensor_meta"].shape,
                        node_dtype,
                    )
                elif node.op == "output":
                    buddy_node = torch_ops_map[node.op].fx_create_node(
                        node.name, node.args
                    )
                elif node.target is operator.getitem:
                    print(str(node.target))
                    node_dtype = self.torch_dtype_to_str(
                        str(node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = torch_ops_map[str(node.target)].fx_create_node(
                        node.name,
                        node.args,
                        node.users.keys(),
                        node.meta["tensor_meta"].shape,
                        node_dtype,
                    )
                else:
                    node_dtype = self.torch_dtype_to_str(
                        str(node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = torch_ops_map[
                        node.target.__name__
                    ].fx_create_node(
                        node.name,
                        node.args,
                        node.users.keys(),
                        node.meta["tensor_meta"].shape,
                        node_dtype,
                    )
                graph.add_node(buddy_node)
            self._imported_graphs.append(graph)
            self._imported_params[graph] = params_flat
            return graph.dynamo_run()

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

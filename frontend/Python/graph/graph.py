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

from typing import Any, List, Optional
from types import FunctionType
import ctypes
import functools
import numpy as np

import mlir.ir as ir
import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt

from .operation import *
from .type import *


def make_output_memref_descriptor(ranks, dtypes):
    """
    Make an output memref descriptor for the given memref ranks and dtypes.

    Parameters:
    - ranks: List[int]
        A list of integers representing the ranks of each memref.
    - dtypes: List[str]
        A list of strings representing the data types of each memref.

    Returns:
    ctypes.Structure
        An output memref descriptor struct.

    Example:
    ranks = [2, 3, 1]
    dtypes = [np.float32, np.int64, np.bool]
    descriptor = make_output_memref_descriptor(ranks, dtypes)
    # Use the descriptor in your code
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
    Graph is a graph-level expression for the Buddy Compiler frontends.
    It acts as a model compute graph, which converts a Graph into an equivalent
    MLIR module.

    Attributes:
    - _body: List[Op]
        The sequence of operation nodes in the graph.
    - _inputs: List[TensorMeta]
        The model inputs represented as TensorMeta objects.
    - _fake_params: List[TensorMeta]
        The fake parameters represented as TensorMeta objects.
    - device: str
        The hardware for graph runtime.
    - _imported_module: Union[None, ImportedModuleType]
        The imported MLIR module after compilation, if set.
    - _ops_registry: dict
        The ops lower strategy for the graph.
    - _func_name: str
        The function name for the MLIR module.
    - _ctx: ir.Context
        The context of the MLIR module.
    - _output_memref: Union[None, ctypes.POINTER]
        The memref pointer in the MLIR function output, if set.
    - _output_descriptor: Union[None, OutputDescriptorType]
        The output descriptor for the MLIR function, if set.
    - ee_: Union[None, ExecutionEngineType]
        The execution engine for the graph, if set.
    """

    def __init__(
        self,
        inputs: List[TensorMeta],
        fake_params: List[TensorMeta],
        ops_registry: dict,
        func_name: str,
        device: DeviceType = DeviceType.CPU,
        verbose=False,
    ) -> None:
        """
        Initializes the Graph.

        Args:
            inputs: List[TensorMeta]
                The model inputs represented as TensorMeta objects.
            fake_params: List[TensorMeta]
                The fake parameters represented as TensorMeta objects.
            ops_registry: dict
                The ops lower strategy for the graph.
            func_name: str
                The function name for the MLIR module.
        """
        self._body = []
        self._inputs = inputs
        self.node_table: Dict[str, Op] = {}
        self._fake_params = fake_params
        self.device = device
        self._imported_module = None
        self._verbose = verbose
        self._ops_registry = ops_registry
        self._func_name = func_name
        self._ctx = ir.Context()
        self._output_memref = None
        self._output_descriptor = None
        self.execution_engine = None
        self.op_groups: Dict[str, List[Op]] = {}
        self.group_map_device: Dict[str, DeviceType] = {}

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, new_body):
        self._body = new_body

    def add_node(self, node: Op):
        """
        Adds an operation node to the graph's body.

        Parameters:
        - node: Op
            The operation node to be added to the graph.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        op_node = Op()
        graph_instance.add_node(op_node)
        # The op_node is now part of the graph's body
        """
        self._body.append(node)
        self.node_table[node.name] = node

    def check_delete_node(self, node: Op) -> bool:
        """
        Determines if a node exists in the graph and has no child nodes.

        Args:
            node (Op): The operation node to check for deletion eligibility.

        Returns:
            bool: True if the node exists in the graph and has no children.
        """
        if not (node.name in self.node_table):
            raise KeyError("node{0} not in graph".format(node.name))

        if len(node._children) == 0:
            return True
        return False

    def delete_node(self, node: Op, parents: List[Op]):
        """
        Removes a node from the graph and updates its parent nodes accordingly.

        Args:
            node (Op): The operation node to be deleted from the graph.
            parents (List[Op]): A list of parent operation nodes that reference the node to be deleted.

        Returns:
            None
        """
        for i in parents:
            i._children.remove(node.name)
        node.args.clear()
        node.kwargs.clear()
        node._children.clear()
        self._body.remove(node)
        self.node_table.pop(node.name)

    def displace_node(self, node: Op, newnode: Op):
        """
        Replaces an existing node with a new node in the graph.

        Args:
            node (Op): The operation node to be replaced.
            newnode (Op): The new operation node that will replace the existing node.

        Returns:
            None
        """
        newnode._arguments = node.args
        newnode._keyword_arguments = node.kwargs
        newnode._tensor_meta = node.tensor_meta
        newnode._op_type = node._op_type

        for i in node._children:
            newnode.add_children(i)
        users = [self.node_table[i] for i in node._children]
        for user in users:
            if node.name in user._parents:
                user._parents[user._parents.index(node.name)] = newnode.name
            user.args[user.args.index(node.name)] = newnode.name
        node._children.clear()
        # deal with parents+args
        for i in node._parents:
            newnode.add_parent(i)
        parents = [self.node_table[i] for i in node._parents]
        for parent in parents:
            parent._children[parent._children.index(node.name)] = newnode.name
        node._parents.clear()
        # update node table
        self._body[self._body.index(node)] = newnode
        self.node_table.pop(node.name)
        self.node_table[newnode.name] = newnode

    def init_op_group(self):
        """
        Initializes operation groups within the graph.

        Returns:
        - None
        """
        for i, op in enumerate(self._body):
            if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
                continue
            group = [op]
            subgraph_name = "subgraph{}".format(i)
            self.group_map_device[subgraph_name] = DeviceType.CPU
            self.op_groups[subgraph_name] = group

    def fuse_ops(self, pattern_list: List[FunctionType]):
        """
        Fuse operations in the graph based on provided fusion patterns.

        Args:
        - pattern_list (List[FunctionType]): A list of functions representing
        fusion patterns.

        Returns:
        - None
        """
        # TODO: discuss two fuse strategy
        # 1. fuse ops adapt for DSA(hardware dependent)
        # 2. common fuse strategy(hardware independent)

        # Apply fusion patterns
        for pattern_func in pattern_list:
            pattern_func(self)

    def perform(self, func_list: List[FunctionType]):
        """
        Perform a series of transformations on the graph using the provided list
        of functions.

        Args:
        - func_list (List[FunctionType]): A list of functions representing
        transformations to be applied to the graph.

        Returns:
        - None
        """
        for transform_func in func_list:
            transform_func(self)

    def lower_to_top_level_ir(self):
        """
        Lowers the graph to top-level MLIR dialects.

        Parameters:
        - do_params_pack: bool, optional (default=False)
            Flag indicating whether to perform parameters packing to one memref.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        graph_instance.lower_to_top_level_ir(do_params_pack=True)
        # The graph is now lowered to top-level MLIR dialects
        """
        with ir.Location.unknown(self._ctx):
            fx_importer = GraphImporter(
                self._body,
                self._fake_params,
                self._inputs,
                self._func_name,
                self._ops_registry,
                False,
                self.device,
                verbose=self._verbose,
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
                case "i8":
                    np_type = np.dtype(np.int8)
                case "i32":
                    np_type = np.dtype(np.int32)
                case "i64":
                    np_type = np.dtype(np.int64)
                case "f16":
                    np_type = np.dtype(np.float16)
                case "f32":
                    np_type = np.dtype(np.float32)
                case _:
                    raise NotImplementedError(f"Unsupported dtype {dtype}")
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
            pm.add("one-shot-bufferize{bufferize-function-boundaries}")
            pm.add("func.func(convert-linalg-to-affine-loops)")
            pm.add("affine-loop-fusion")
            pm.add("func.func(affine-parallelize)")
            pm.add("convert-scf-to-openmp")
            pm.add("expand-strided-metadata")
            pm.add("lower-affine")
            pm.add("convert-vector-to-llvm")
            pm.add("memref-expand")
            pm.add("arith-expand")
            pm.add("convert-arith-to-llvm")
            pm.add("finalize-memref-to-llvm")
            pm.add("convert-scf-to-cf")
            pm.add("convert-cf-to-llvm")
            pm.add("func.func(llvm-request-c-wrappers)")
            pm.add("convert-openmp-to-llvm")
            pm.add("convert-math-to-llvm")
            pm.add("convert-math-to-libm")
            pm.add("convert-func-to-llvm")
            pm.add("reconcile-unrealized-casts")
            pm.run(self._imported_module.operation)

    def compile(self):
        """
        Compile graph from Buddy Graph to LLVM IR.
        """
        self.lower_to_top_level_ir()
        self.lower_to_llvm_ir()


class GraphImporter:
    """
    Imports an buddy graph and generates an MLIR module in high-level dialects.

    Attributes:
        _symbol_table (dict): A dictionary to keep track of the symbols.
        _body (List[Op]): The FX graph module to be imported.
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
        func_name: str,
        ops_registry: dict,
        do_param_pack: bool = False,
        device: DeviceType = DeviceType.CPU,
        verbose=False,
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
        self._body = body
        self._device = device
        self._func_name = func_name
        self._params = params
        self._inputs = inputs
        self._verbose = verbose
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
            case TensorDType.Int8:
                return ir.IntegerType.get_signless(8)
            case TensorDType.Int32:
                return ir.IntegerType.get_signless(32)
            case TensorDType.Int64:
                return ir.IntegerType.get_signless(64)
            case TensorDType.Float16:
                return ir.F16Type.get()
            case TensorDType.Float32:
                return ir.F32Type.get()
            case TensorDType.Bool:
                return ir.IntegerType.get_signless(1)
            case _:
                raise NotImplementedError(f"Unsupported dtype {dtype}")

    def _pack_params(self) -> None:
        """
        Packs parameters of the graph to one memref.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        graph_instance._pack_params()
        # The parameters of the graph are now packed to one memref.
        """
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
                    lambda x, y: x * y, list(param.shape), 1
                )
            mlir_dtype = self._str_to_mlir_dtype(dtype)
            self._param_packs.append(
                ir.MemRefType.get([param_total_size], mlir_dtype)
            )

    def import_graph(self) -> ir.Module:
        """
        Imports buddy graph and generates an MLIR module in high-level dialects.

        Returns:
            mlir.ir.Module: An MLIR module in high-level dialects.
        """
        assert self._do_param_pack == False
        with ir.InsertionPoint(self._module.body):
            arguments = []
            inputs = self._params + self._inputs
            for arg in inputs:
                shape_list = list(arg.shape)
                dtype = arg.dtype
                mlir_dtype = self._str_to_mlir_dtype(dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)
            extern_func = []
            for node in self._body:
                if isinstance(node, FuncOp):
                    extern_func.append(node)
                    self._import_op(node)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                func_op = self._module.body.operations[0]
                for node in self._body:
                    if node in extern_func:
                        continue
                    old_ops = [op for op in func_op.body.blocks[0].operations]
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
                        self._symbol_table[(str(node.name), 0)] = (
                            self._symbol_table[
                                (str(node.args[0]), node.args[1])
                            ]
                        )
                    else:
                        self._import_op(node)
                    new_ops = [op for op in func_op.body.blocks[0].operations]
                    if self._verbose:
                        print("=" * 20 + "Graph Node" + "=" * 20)
                        print("Node: " + node.name)
                        print("Type: " + str(node._op_type))
                        print("Arguments: " + str(node.args))
                        print("Parents: " + str(node._parents))
                        print("Children: " + str(node._children))
                        print("-" * 20 + "MLIR OPS" + "-" * 20)
                        for op in new_ops:
                            if op not in old_ops:
                                print(op)
                        print("")

                return self._symbol_table.get(("output", 0))

        return self._module

    def import_main_graph(self) -> ir.Module:
        """
        Imports buddy main graph to organize all subgraphs and generates an MLIR
        module in high-level dialects with memref.

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
                tensor_arg = ir.MemRefType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)
            extern_func = []
            for node in self._body:
                if isinstance(node, FuncOp):
                    extern_func.append(node)
                    self._import_op(node)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                for node in self._body:
                    if node in extern_func:
                        continue
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
                        self._symbol_table[(str(node.name), 0)] = (
                            self._symbol_table[
                                (str(node.args[0]), node.args[1])
                            ]
                        )
                    else:
                        self._import_op(node)

                return self._symbol_table.get(("output", 0))

        return self._module

    def _import_placeholder(
        self, node: PlaceholderOp, args_list: List[ir.BlockArgument]
    ):
        """
        Imports a placeholder node from the Buddy graph.

        Parameters:
        - node (PlaceholderOp): The PlaceholderOp node representing the
        placeholder.
        - args_list (List[mlir.ir.BlockArgument]): List of input memrefs.

        Returns:
        None
        """
        if self._num_input_visited < len(self._params) and self._do_param_pack:
            dtype = node.tensor_meta["dtype"]
            pack_of_dtype = None
            for pack in args_list:
                if ir.MemRefType(
                    pack.type
                ).element_type == self._str_to_mlir_dtype(dtype):
                    pack_of_dtype = pack
                    break
            placeholder_name = self._ops_registry["param.extract"](
                node, self._current_param_pack_offset[dtype], pack_of_dtype
            ).result
            self._current_param_pack_offset[dtype] += functools.reduce(
                lambda x, y: x * y, list(node.tensor_meta["shape"]), 1
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
        op_ret: ir.Operation | ir.Value | tuple | List | ir.OpResult = (
            self._ops_registry[op_name](node, self._symbol_table)
        )
        if isinstance(op_ret, tuple | List | ir.OpResultList):
            for i, operation in enumerate(op_ret):
                if isinstance(operation, ir.Operation) or isinstance(
                    operation, ir.OpView
                ):
                    self._symbol_table[(str(node.name), i)] = operation.result
                elif isinstance(operation, ir.OpResult):
                    self._symbol_table[(str(node.name), i)] = operation
                else:
                    raise NotImplementedError
        elif isinstance(op_ret, ir.OpResult):
            self._symbol_table[(str(node.name), 0)] = op_ret
        else:
            for i, result in enumerate(op_ret.results):
                self._symbol_table[(str(node.name), i)] = result

    def get_output_nodes(self):
        """
        Get output nodes from the lowered mlir func.
        """
        return self._symbol_table.get(("output", 0))

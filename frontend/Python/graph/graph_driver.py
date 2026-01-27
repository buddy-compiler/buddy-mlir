# ===- graph_driver.py ---------------------------------------------------------
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
# This is the graph driver to drive the input graph:
#     1. Split the input graph into subgraphs.
#     2. Construct a main graph to call subgraphs with right order.
#
# ===---------------------------------------------------------------------------

import os
import functools
import numpy as np
from mlir import ir
import torch
from collections import deque, defaultdict

from .graph import Graph, GraphImporter, TensorMeta
from .operation import *
from .type import *
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Type, Optional

@dataclass
class SplitStrategy:
    name: str = "default_no_split"
    parallel_num: int = 1 
    ops_count: List[int] = field(default_factory=list)
    stage_boundary_op: Optional[Type] = None
    stage_boundary_op_num: int = 0 
    
    paral_input_positions: Dict[Union[int, str], Any] = field(default_factory=dict)

    def get_paral_pos(self, subgraph_idx: int) -> List[int]:
        if self.parallel_num <= 1:
            return []
            
        if subgraph_idx in self.paral_input_positions:
            return self.paral_input_positions[subgraph_idx]
        
        default_configs = self.paral_input_positions.get("default", [])
        if default_configs and self.ops_count:
            block_idx = (subgraph_idx - 1) % len(self.ops_count)
            
            if block_idx < len(default_configs):
                return default_configs[block_idx]
                
        return [] 
# DECODE_STRATEGY = SplitStrategy(
#     name="decode",
#     ops_count=[6, 65, 2, 6, 14, 2],
#     stage_boundary_op=PowOp,
    # stage_boundary_op_num: int = 0 ,
#     paral_input_positions={
#         0: [-1, -1, -1, -1],
#         169: [-1, -1, -1],
#         "default": [
#             [-1, -1],
#             [-1, 0, 0, 0, 0, 0, 0, -1, -1, 1, -1, 1, -1, 1],
#             ...
#         ]
#     }
# )
# driver = GraphDriver(my_graph, DECODE_STRATEGY)
# # 方式 1：完全不传
# driver = GraphDriver(my_graph)


class GraphDriver:
    """
    Class responsible for managing and driving the execution of a computational
    graph.

    Attributes:
    - _graph (Graph): The computational graph associated with this driver.
    - _subgraphs (dict): A dictionary mapping subgraph names to their
    corresponding subgraphs.
    - _subgraphs_inputs (dict): A dictionary mapping subgraph names to their
    input placeholders.
    - _subgraphs_outputs (dict): A dictionary mapping subgraph names to their
    output op's result.
    """

    def __init__(self, graph: Graph, strategy: Optional[SplitStrategy] = None) -> None:
        """
        Initialize the GraphDriver object with a given computational graph.

        Args:
        - graph (Graph): The computational graph to be associated with this
        driver.

        Returns:
        - None
        """
        self._graph = graph
        self.strategy = strategy or SplitStrategy()
        self._parallelism = self.strategy.parallel_num
        self._subgraph_dependencies = {}
        self._subgraph_input_shape = defaultdict(dict)
        self._paral_op_shape: Dict[str, List[int]] = {}
        self.op_groups = self._graph.op_groups
        (
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = self.get_split_strategy()
        (
            self._subgraphs
        ) = self.build_subgraph_by_group()
        self.group_map_device = self._graph.group_map_device

        self._call_table = {}  

        self._maingraphs = {}
        self._modules = {}
        self._subgraph_param_info = defaultdict(dict)

    @property
    def subgraphs(self):
        return list(self._subgraphs.values())

    @property
    def maingraphs(self):
        return list(self._maingraphs.values())
    
    @property
    def modules(self):
        return list(self._modules.values())
    
    @property
    def subgraph_param_indices(self):
       return list(self._subgraph_param_indices.values())
    
    def _add_paral_op_shape(self, op_name, shape):
        if op_name not in self._paral_op_shape.keys():
            self._paral_op_shape[op_name] = shape

    def _normalize_binary_operator_shape(self, shp1, shp2):
        """Normalize the shape of two input tensors according to the broadcasting
        rule"""
        shp1 = list(shp1)
        shp2 = list(shp2)
        while len(shp1) < len(shp2):
            shp1.insert(0, 1)
        while len(shp2) < len(shp1):
            shp2.insert(0, 1)
        return shp1, shp2
    
    def _infer_new_shape(self, old_shape, new_shape):
        total_size = 1
        for dim_siz in old_shape:
            total_size *= dim_siz

        neg_one_cnt = 0
        rest_size = 1
        for dim_siz in new_shape:
            if dim_siz == -1:
                neg_one_cnt += 1
                continue
            rest_size *= dim_siz

        if neg_one_cnt != 0:
            if neg_one_cnt > 1 or total_size % rest_size != 0:
                raise ValueError("Can not infer the new shape!")
            infer_dim_size = total_size // rest_size
            for i, _ in enumerate(new_shape):
                if new_shape[i] == -1:
                    new_shape[i] = infer_dim_size
        return new_shape
    
    def get_pack_params_size(self, tensors_meta: list[TensorMeta]) -> int:
        param_total_size = 0
        for tensor_meta in tensors_meta:
            param_total_size += functools.reduce(
                lambda x, y: x * y, list(tensor_meta.shape), 1
            )
        return param_total_size

    def get_split_strategy(self):
      """
      Group ops based on the computational graph in terms of subgraphs.
      
      Analyse the inputs and outputs of each subgraph.

      Update the shape information of the nodes in each subgraph 
      associated with the weight matrix to be split.

      Returns:
      - None
      """
        
      if self._parallelism < 1:
            raise ValueError("Parallelism must be greater than or equal to 1")
        
        
      self.op_groups = {}
      self.group_map_device = {}
      self._subgraphs_inputs = {}
      self._subgraphs_outputs = {}
      self._paral_op_shape = {}
        
      self._perform_vertical_split()
        
      self._analyze_subgraph_dependencies()
        
      if self._parallelism > 1:
            self._apply_horizontal_parallelism()

      return self._subgraphs_inputs, self._subgraphs_outputs
    
    def _apply_horizontal_parallelism(self):
        
        for i, subgraph_name in enumerate(self.op_groups.keys()):
            paral_pos = self.strategy.get_paral_pos(i) 
                
            input_count = 0
            for node in self._subgraphs_inputs[subgraph_name]:
                
                original_shape = list(node.tensor_meta["shape"])
                
                if input_count >= len(paral_pos):
                    break
                
                split_dim = paral_pos[input_count]
                input_count += 1
                
                
                if split_dim != -1 and split_dim < len(original_shape):
                    original_shape[split_dim] = original_shape[split_dim] // self._parallelism
                    self._add_paral_op_shape(node.name, original_shape)
                self._subgraph_input_shape[subgraph_name][node.name] = original_shape
            
                
        for subgraph_name in self.op_groups.keys():
            current_ops = self.op_groups[subgraph_name]
            
            for node in current_ops:
                # 1. PermuteOp
                if isinstance(node, PermuteOp):
                    if node.args[0] in self._paral_op_shape:
                        old_shape = self._paral_op_shape[node.args[0]]
                        permute_indices = node.args[1]
                        
                        try:
                            new_shape = [old_shape[index] for index in permute_indices]
                            self._add_paral_op_shape(node.name, new_shape)
                        except IndexError:
                            print(f"\n[ERROR] PermuteOp Shape Mismatch!")
                            print(f"  Node: {node.name}")
                            print(f"  Input Node: {node.args[0]}")
                            print(f"  Input Shape (old_shape): {old_shape}")
                            print(f"  Permute Indices: {permute_indices}")
                            print(f"  Reason: Indices require rank {max(permute_indices)+1}, but input has rank {len(old_shape)}.")
                            raise  

                # 2. MatmulOp
                elif isinstance(node, MatmulOp):
                    if (node.args[0] in self._paral_op_shape) or (node.args[1] in self._paral_op_shape):
                        input1_shape = self._get_shape_from_cache_or_node(node.args[0])
                        input2_shape = self._get_shape_from_cache_or_node(node.args[1])

                        new_shape = list(input1_shape)
                        new_shape[-1] = input2_shape[-1]
                        self._add_paral_op_shape(node.name, new_shape)

                # 3. AddMMOp
                elif isinstance(node, AddMMOp):
                    # args: bias, input, weight
                    if (node.args[1] in self._paral_op_shape) or (node.args[2] in self._paral_op_shape):
                        input_shape = self._get_shape_from_cache_or_node(node.args[1]) # [M, K]
                        weight_shape = self._get_shape_from_cache_or_node(node.args[2]) # [K, N]
                        
                        if input_shape and weight_shape:
                            M = input_shape[-2] if len(input_shape) >= 2 else input_shape[0]
                            N = weight_shape[-1]
                            new_shape = [M, N]
                            self._add_paral_op_shape(node.name, new_shape)

                # 4. Binary Ops (Add, Sub, Mul, Div)
                elif isinstance(node, (AddOp, SubOp, MulOp, DivOp)):
                    arg0_name = node.args[0]
                    arg1_name = node.args[1]
                    
                    is_arg0_split = isinstance(arg0_name, str) and (arg0_name in self._paral_op_shape)
                    is_arg1_split = isinstance(arg1_name, str) and (arg1_name in self._paral_op_shape)

                    if is_arg0_split or is_arg1_split:
                        shape0 = self._get_shape_from_cache_or_node(arg0_name)
                        shape1 = self._get_shape_from_cache_or_node(arg1_name)
                        
                        if shape0 and shape1:
                            norm_s1, norm_s2 = self._normalize_binary_operator_shape(shape0, shape1)
                            new_shape = []
                            for d1, d2 in zip(norm_s1, norm_s2):
                                new_shape.append(max(d1, d2))
                            self._add_paral_op_shape(node.name, new_shape)

                elif isinstance(node, ViewOp):
                    parent = node.args[0]
                    if parent in self._paral_op_shape:
                        old_shape = self._paral_op_shape[parent]
                        target_shape_args = list(node.args[1])

                        current_total = 1
                        for x in old_shape: current_total *= x
                        
                        target_total = 1
                        for x in target_shape_args: target_total *= x
                        
                        new_shape = target_shape_args.copy()

                        if current_total != target_total:
                            old_len = len(old_shape)
                            new_len = len(new_shape)
                            
                            tmp_old = [d for d in old_shape if d != 1]
                            tmp_new = [d for d in new_shape if d != 1]

                            if len(tmp_old) == len(tmp_new):
                                if old_len < new_len:
                                    for i in range(old_len):
                                        new_shape[i+1] = old_shape[i]
                                elif old_len == new_len:
                                    for i in range(new_len):
                                        new_shape[i] = old_shape[i]
                                else:
                                    for i in range(new_len):
                                        new_shape[i] = old_shape[i+1]
                            else:
                                if old_len < new_len:
                                    if new_shape[-1] != 0:
                                        new_shape[-2] = old_shape[-1] // new_shape[-1]
                                else:
                                    new_shape = self._infer_new_shape(old_shape, new_shape)
                        
                        self._add_paral_op_shape(node.name, new_shape)

                # 6. CatOp
                elif isinstance(node, CatOp):
                    tensors = node.args[0]
                    for t in tensors:
                        t_name = str(t)
                        if t_name in self._paral_op_shape:
                            self._add_paral_op_shape(node.name, self._paral_op_shape[t_name])
                            break

                # 7. IndexPutOp
                elif isinstance(node, IndexPutOp):
                    target_arg = str(node.args[0])
                    target_shape = self._get_shape_from_cache_or_node(target_arg)
                    if target_shape:
                        node.tensor_meta["shape"] = target_shape
                        self._add_paral_op_shape(node.name, target_shape)
                
                elif isinstance(node, ReshapeOp):
                    parent = node.args[0]
                    if parent in self._paral_op_shape:
                        old_shape = list(self._paral_op_shape[parent])  # e.g. [1,1,6,128]
                    else:
                        # self._add_paral_op_shape(node.name, self._paral_op_shape[arg])
                        continue
                    
                    target_shape_args = list(node.args[1])

                    # print("  Parent:", parent)
                    # print("  Parent Shape:", old_shape)
                    # print("  Target Shape:", target_shape_args)

                    def prod(shape):
                        p = 1
                        for d in shape:
                            if d == -1:
                                continue
                            p *= d
                        return p

                    old_total = prod(old_shape)
                    new_shape = target_shape_args.copy()

                    if -1 in new_shape:
                        known = 1
                        neg_idx = None
                        for i, d in enumerate(new_shape):
                            if d == -1:
                                neg_idx = i
                            else:
                                known *= d
                        
                        if known != 0 and neg_idx is not None:
                            new_shape[neg_idx] = old_total // known

                    if len(old_shape) == len(new_shape):
                        non1_slots = [i for i, d in enumerate(new_shape) if d != 1]
                        
                        old_non1_vals = [d for d in old_shape if d != 1]

                        if len(non1_slots) == len(old_non1_vals) and len(non1_slots) > 0:
                            
                            locked = {}
                            remaining_old = old_non1_vals.copy()

                            for idx in non1_slots:
                                want = new_shape[idx]
                                
                                if isinstance(want, int) and want > 1 and want in remaining_old:
                                    locked[idx] = want
                                    remaining_old.remove(want)

                            
                            filled = new_shape.copy()
                            rem_iter = iter(remaining_old)
                            for idx in non1_slots:
                                if idx in locked:
                                    filled[idx] = locked[idx]
                                else:
                                    filled[idx] = next(rem_iter)

                            
                            if prod(filled) == prod(old_shape):
                                new_shape = filled

                    
                    old_candidates = [d for d in old_shape if d != 1]
                    
                    old_non_trailing = old_candidates[:-1] if len(old_candidates) > 1 else old_candidates

                    for i, d in enumerate(new_shape):
                        if d > 1 and d not in old_shape:
                            if len(old_non_trailing) == 1:
                                new_shape[i] = old_non_trailing[0]

                    
                    if prod(new_shape) != prod(old_shape):
                        # print(f"[WARN] ReshapeOp {node.name} inferred shape mismatch: old={old_shape}, new={new_shape}. Keep old.")
                        new_shape = old_shape

                    # print("  Inferred Shape:", new_shape)
                    self._add_paral_op_shape(node.name, new_shape)


                # 8. ExpandOp
                elif isinstance(node, ExpandOp) and node != self.op_groups[subgraph_name][-1]:
                    op_arg = str(node.args[0])              
                    if op_arg in self._paral_op_shape:       
                        new_shape = self._paral_op_shape[op_arg]   

                        old_new_size = node.args[1]

                        new_group_dim = new_shape[1]

                        new_new_size = old_new_size.copy()
                        new_new_size[1] = new_group_dim      
                        
                        node.args[1] = new_new_size
                        self._add_paral_op_shape(node.name, new_new_size)
                        

                else:
                    for arg in node.args:
                        if isinstance(arg, str) and arg in self._paral_op_shape:
                            self._add_paral_op_shape(node.name, self._paral_op_shape[arg])
                            break
        
    
    def _get_shape_from_cache_or_node(self, arg_name):
        
        if isinstance(arg_name, (int, float)):
            return [] # 标量
        
        arg_name = str(arg_name)
        if arg_name in self._paral_op_shape:
            return self._paral_op_shape[arg_name]
        elif arg_name in self._graph.node_table:
            # 原始形状
            node = self._graph.node_table[arg_name]
            # 处理多输出情况
            shape = node.tensor_meta["shape"]
            if shape and isinstance(shape[0], (list, tuple)):
                return list(shape[0])
            return list(shape)
        return None
       
    
    def _perform_vertical_split(self):
        
        ops_count = self.strategy.ops_count
        max_strategy_op_count = self.strategy.stage_boundary_op_num
        
        submodel_count = 0
        strategy_op_count = 0
        tsf_count = 0
        
        def new_subgraph():
            nonlocal submodel_count, tsf_count
            key = list(self._graph.op_groups.keys())[0]
            name = f"{key}{submodel_count}"       
            self.op_groups[name] = []
            self.group_map_device[name] = DeviceType.CPU
            tsf_count = 0
            return name
        
        current_subgraph = None
        for op in self._graph.body:
                    # 跳过占位符和输出算子
            if isinstance(op, (PlaceholderOp, OutputOp)):
                continue
            
            if current_subgraph is None:
                current_subgraph = new_subgraph()
                self.op_groups[current_subgraph].append(op)
                continue
        
            if (self.strategy.stage_boundary_op is not None and 
                isinstance(op, self.strategy.stage_boundary_op)):
                strategy_op_count += 1
                submodel_count += 1
                current_subgraph = new_subgraph()
                tsf_count = 1
                self.op_groups[current_subgraph].append(op)
                continue
            
            if 0 < strategy_op_count  < max_strategy_op_count  and ops_count:
                target = ops_count[(submodel_count - 1) % len(ops_count)]
            
                if tsf_count == target:
                    submodel_count += 1
                    current_subgraph = new_subgraph()
                    tsf_count = 1
                    self.op_groups[current_subgraph].append(op)
                    continue
                else:
                    tsf_count += 1

            self.op_groups[current_subgraph].append(op)


    def _analyze_subgraph_dependencies(self):
        
        total_graph_outputs = []
        for node in self._graph.body:
            if isinstance(node, OutputOp):
                total_graph_outputs.extend([arg for arg in node.args])
        

        for name, ops in self.op_groups.items():
            self._subgraphs_inputs[name] = []
            self._subgraphs_outputs[name] = []
            
            op_set_in_subgraph = set(ops)
            
            for op in ops:
                deps = self._get_op_all_dependencies(op)
                
                for parent_name in deps:
                    if parent_name not in self._graph.node_table:
                        continue
                    
                    parent_op = self._graph.node_table[parent_name]
                    if parent_op not in op_set_in_subgraph:
                        if parent_op not in self._subgraphs_inputs[name]:
                            self._subgraphs_inputs[name].append(parent_op)

        all_inputs_of_all_subgraphs = []
        for in_list in self._subgraphs_inputs.values():
            all_inputs_of_all_subgraphs.extend(in_list)
        all_inputs_set = set(all_inputs_of_all_subgraphs)

        for name, ops in self.op_groups.items():
            for op in ops:
                if op in all_inputs_set or op.name in total_graph_outputs:
                    if op not in self._subgraphs_outputs[name]:
                        self._subgraphs_outputs[name].append(op)
            
            self._subgraph_dependencies[name] = set()
        node_to_index = {node: i for i, node in enumerate(self._graph.body)}    
        for name in self._subgraphs_inputs:
            self._subgraphs_inputs[name].sort(key=lambda node: node_to_index.get(node, -1))
            
    def _get_op_all_dependencies(self, op) -> List[str]:
        deps = []
        # 处理标准 parents
        for p in op._parents:
            if isinstance(p, str): deps.append(p)
            elif hasattr(p, "name"): deps.append(p.name)
        
        # 处理 args 中的列表 (如 IndexPutOp)
        for arg in op.args:
            if isinstance(arg, list):
                for item in arg:
                    if item is not None:
                        name = item if isinstance(item, str) else getattr(item, 'name', None)
                        if name and name in self._graph.node_table:
                            deps.append(name)
        return list(set(deps))
            

    def build_subgraph_by_group(self):
        """
        Builds subgraphs from a given graph based on groups.

        Args:
        - graph (Graph): The graph from which subgraphs are constructed.

        Returns:
        - tuple: A tuple containing dictionaries of subgraphs, subgraph inputs,
        and subgraph outputs.
        """

        subgraphs = {}

        # Construct each subgraph
        for subgraph_name in self.op_groups.keys():
            subgraph_input = []
            subgraph_body_list = []
            subgraph_device = self.group_map_device[subgraph_name]

            # Construct input placeholder nodes
            for node in self._subgraphs_inputs[subgraph_name]:
                
                if subgraph_name in self._subgraph_input_shape and \
                   node.name in self._subgraph_input_shape[subgraph_name]:
                    # print("A")
                    node_shape = self._subgraph_input_shape[subgraph_name][node.name]
                    # print(f"[DEBUG] Using modified shape for {node.name} in {subgraph_name}: {node_shape}")
                elif node.name in self._paral_op_shape.keys():
                    node_shape = self._paral_op_shape[node.name]
                else:
                    node_shape = node.tensor_meta["shape"]
                # node_dtype = node.tensor_meta["dtype"]
                

                if node_shape and isinstance(node_shape[0], (list, tuple)):
                    node_shape = list(node_shape[0]) 
                else:
                    node_shape = list(node_shape)

                raw_dtype = node.tensor_meta["dtype"]
                if isinstance(raw_dtype, (list, tuple)):
                    node_dtype = raw_dtype[0]
                else:
                    node_dtype = raw_dtype
                input_tensor_meta = TensorMeta(node_shape, node_dtype)
                subgraph_input.append(input_tensor_meta)
                placeholder_node = PlaceholderOp()
                placeholder_node.name = node.name
                placeholder_node.tensor_meta = input_tensor_meta
                for op in self.op_groups[subgraph_name]:
                    # if node.name in node._parents:
                    if node.name in (op._parents if isinstance(op._parents, (list, tuple)) else []):
                        placeholder_node.add_children(op.name)
                subgraph_body_list.append(placeholder_node)

            # Add operations to subgraph body
            for op in self.op_groups[subgraph_name]:
                if isinstance(op, (ViewOp, ReshapeOp)) and self._parallelism > 1:
                    if op.args[0] in self._paral_op_shape.keys():
                        op._newshape = self._paral_op_shape[op.name]
                subgraph_body_list.append(op)
                
            # Construct output node
            output_node = OutputOp()
            output_node.name = "output"
            for output in self._subgraphs_outputs[subgraph_name]:
                output_node.add_argument(output.name)
                output_node.add_parent(output.name)
            subgraph_body_list.append(output_node)

            # Create subgraph and add it to the dictionary
            subgraph = Graph(
                subgraph_input,
                [],
                self._graph._ops_registry,
                subgraph_name,
                subgraph_device,
                verbose=self._graph._verbose,
            )
            subgraph.body = subgraph_body_list
            for op in subgraph_body_list:
                subgraph.node_table[op.name] = op
            subgraphs[subgraph_name] = subgraph

        return subgraphs
    
    def topological_sort_subgraph(self):
        """
        Performs topological sorting on the subgraphs based on their dependencies.
        Args:
        - graph (Graph): The graph from which subgraphs are constructed.
        Returns:
        - list: A list of subgraph names in topological order if the graph is acyclic; otherwise, None.
        """
        # Calculate in degree of each subgraph
        in_degree = {
            subgraph_name: 0 for subgraph_name in list(self._subgraphs.keys())
        }
        for src, dests in self._subgraph_dependencies.items():
            for dest in dests:
                in_degree[dest] += 1
        # Topological sorting
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in self._subgraph_dependencies[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # TODO: If the custom subgraph partitioning is illegal, further partition the subgraph to make it valid.
        return (
            topo_order
            if len(topo_order) == len(list(self._subgraphs.keys()))
            else None
        )

    def construct_main_graph(self, do_param_pack=False):
        """
        Constructs the main computational graph by incorporating subgraphs' call
        and placeholder operations.

        Args:
        - do_param_pack (bool): Flag indicating whether parameter packing should
        be performed. Defaults to False.

        Returns:
        - Graph: The main computational graph constructed.

        Note: The actual call sequence and topology analysis are pending
        implementation.

        """
        # Analysis topology order to sort subgraph call.
        topo_order = self.topological_sort_subgraph()
        if topo_order == None:
            print("Error : Graph Partitioning is illegal!")
            return None
        # Adding FuncOp nodes for each subgraph
        inputs0 = self._graph._inputs
        split_group = []
        param_size_group = []
        for i, subgraph_name in enumerate(self._subgraphs.keys()):
            main_graph_name = f"{self._graph._func_name}{i}"
            current_param_info = {} 
            
            main_graph = Graph(
                [],
                [],
                self._graph._ops_registry,
                main_graph_name,
                self._graph._verbose,
              )

            func_node = FuncOp()
            func_node.name = subgraph_name
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs[subgraph_name]._inputs:
                func_node.add_argument(inp)
            
            outputs = self._subgraphs[subgraph_name]._outputs
            if outputs is not None and self._parallelism > 1:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    func_node.tensor_meta["shape"].append(torch.Size(output_shape))
            else:
                for output in self._subgraphs_outputs[subgraph_name]:
                    func_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
            
            for output in self._subgraphs_outputs[subgraph_name]:
                func_node.tensor_meta["dtype"].append(
                    self._graph.node_table[output.name].tensor_meta["dtype"]
                )
            main_graph.add_node(func_node)
            
            # Adding placeholder operations from the original graph
            ph_count : int = 0 
            
            issplit = False
            current_param_info["params"] = []
            current_param_info["total_partitions"] = 1
            split_group.append(1)
            current_subgraph_input_names = set(n.name for n in self._subgraphs_inputs[subgraph_name])


            maingraph_input = []
            for node in self._graph.body:
                if isinstance(node, PlaceholderOp):
                    # if node in self._subgraphs_inputs[subgraph_name]:
                    if node.name in current_subgraph_input_names:
                        if len(self._graph._fake_params) > ph_count:
                            main_graph._fake_params.append(self._graph._fake_params[ph_count])
                            if node.name in self._paral_op_shape.keys():
                                node._newshape = self._paral_op_shape[node.name]
                                main_graph._fake_params[-1]['shape'] = torch.Size(node._newshape)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": node._newshape}
                                )
                                issplit = True
                            else: 
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": []}
                                )
                        else:
                            if node.name in self._paral_op_shape:
                                node_shape = self._paral_op_shape[node.name]
                            else:
                                node_shape = node.tensor_meta["shape"]
                            node_dtype = node.tensor_meta["dtype"]
                            input_tensor_meta = TensorMeta(node_shape, node_dtype)
                            maingraph_input.append(input_tensor_meta)
                        main_graph.add_node(node)
                    ph_count += 1
            
            param_size_group.append(self.get_pack_params_size(main_graph._fake_params))

            if issplit: 
                current_param_info["total_partitions"] = self._parallelism
                split_group[-1] = self._parallelism
            # self._subgraph_param_info[subgraph_name] = current_param_info
            key = list(self._graph.op_groups.keys())[0]
            name = f"{key}{i}"    
            self._subgraph_param_info[name] = current_param_info

            for node in self._subgraphs_inputs[subgraph_name]:
                if (node.name not in main_graph.node_table.keys()):
                    if subgraph_name in self._subgraph_input_shape and \
                       node.name in self._subgraph_input_shape[subgraph_name]:
                        node_shape = self._subgraph_input_shape[subgraph_name][node.name]
                    else:
                        node_shape = node.tensor_meta["shape"]
                    
                    node_dtype = node.tensor_meta["dtype"]
                    input_tensor_meta = TensorMeta(node_shape, node_dtype)
                    maingraph_input.append(input_tensor_meta)
                    
                    placeholder_node = PlaceholderOp()
                    placeholder_node.name = node.name
                    placeholder_node.tensor_meta = input_tensor_meta
                    main_graph.add_node(placeholder_node)
            
            # Adding CallOp to invoke the single subgraph
            call_node = CallOp()
            call_node.name = "call{}".format(i)
            call_node.call_func_name = subgraph_name
            call_node.tensor_meta = {"shape": [], "dtype": []}
            for node in self._subgraphs_inputs[subgraph_name]:
                if node.name in self._graph.node_table:
                    call_node.add_argument(node.name)
                    continue
                for key, value in self._subgraphs_outputs.items():
                    if node in value:
                        call_node.add_argument(
                            arg=self._call_table[key].name,
                            arg_index=value.index(node.name),
                        )
                        break
            outputs = self._subgraphs[subgraph_name]._outputs
            if outputs is not None and self._parallelism > 1:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    call_node.tensor_meta["shape"].append(torch.Size(output_shape))
            else:
                for output in self._subgraphs_outputs[subgraph_name]:
                    call_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
            
            for output in self._subgraphs_outputs[subgraph_name]:
                call_node.tensor_meta["dtype"].append(
                    self._graph.node_table[output.name].tensor_meta["dtype"]
                )
            
            self._call_table[subgraph_name] = call_node
            main_graph.add_node(call_node)

            # Adding GetItemOps to retrieve individual output tensors
            output_node = OutputOp()
            for m, output in enumerate(self._subgraphs_outputs[subgraph_name]):
                getitem_node = GetItemOp()
                getitem_node.add_argument(call_node.name)
                getitem_node.add_argument(m)
                getitem_node.name = "getitem{}".format(m)
                output_node.add_argument(getitem_node.name)
                main_graph.add_node(getitem_node)
            # Marking the final output of the main graph
            output_node.name = "output"
            main_graph.add_node(output_node)
            self._maingraphs[main_graph_name] = main_graph

            # Importing the main graph
            with ir.Location.unknown(ir.Context()):
                main_importer = GraphImporter(
                    main_graph.body,
                    main_graph._fake_params,
                    maingraph_input,
                    main_graph._func_name,
                    main_graph._ops_registry,
                    do_param_pack,
                )
                self._modules[main_graph_name] = main_importer.import_main_graph()
        
        print(f"split_group: {split_group}")
        print(f"param_size_group: {param_size_group}")
        


    # def construct_main_graph(self, do_param_pack=False):
    #     """
    #     Constructs the main computational graph by incorporating subgraphs' call
    #     and placeholder operations.

    #     Args:
    #     - do_param_pack (bool): Flag indicating whether parameter packing should
    #     be performed. Defaults to False.

    #     Returns:
    #     - Graph: The main computational graph constructed.

    #     Note: The actual call sequence and topology analysis are pending
    #     implementation.

    #     """
    #     main_graph = Graph(
    #         self._graph._inputs,
    #         self._graph._fake_params,
    #         self._graph._ops_registry,
    #         self._graph._func_name,
    #         self._graph._verbose,
    #     )

    #     # Adding FuncOp nodes for each subgraph
    #     for subgraph_name in self._subgraphs.keys():
    #         func_node = FuncOp()
    #         func_node.name = subgraph_name
    #         func_node.tensor_meta = {"shape": [], "dtype": []}
    #         for inp in self._subgraphs[subgraph_name]._inputs:
    #             func_node.add_argument(inp)
    #         for output in self._subgraphs_outputs[subgraph_name]:
    #             # func_node.tensor_meta["shape"].append(
    #             #     self._graph.node_table[output].tensor_meta["shape"]
    #             # )
    #             # func_node.tensor_meta["dtype"].append(
    #             #     self._graph.node_table[output].tensor_meta["dtype"]
    #             # )
    #             func_node.tensor_meta["shape"].append(output.tensor_meta["shape"])
    #             func_node.tensor_meta["dtype"].append(output.tensor_meta["dtype"])
    #         main_graph.add_node(func_node)

    #     # Adding placeholder operations from the original graph
    #     for op in self._graph.body:
    #         if isinstance(op, PlaceholderOp):
    #             main_graph.add_node(op)
    #     # Analysis topology order to sort subgraph call.
    #     topo_order = self.topological_sort_subgraph()
    #     if topo_order == None:
    #         print("Error : Graph Partitioning is illegal!")
    #         return None
    #     # Adding CallOp to invoke the single subgraph
    #     for i, subgraph_name in enumerate(topo_order):
    #         call_node = CallOp()
    #         call_node.name = "call{}".format(i)
    #         call_node.call_func_name = subgraph_name
    #         call_node.tensor_meta = {"shape": [], "dtype": []}
    #         for inp in self._subgraphs_inputs[subgraph_name]:
    #             inp_name = inp.name if hasattr(inp, 'name') else inp
    #             if inp_name in main_graph.node_table:
    #                 call_node.add_argument(inp_name)
    #                 continue
    #             found_dependency = False
    #             for key, value in self._subgraphs_outputs.items():
    #                 if inp in value:
    #                     call_node.add_argument(
    #                         arg=self._call_table[key].name,
    #                         arg_index=value.index(inp),
    #                     )
    #                     found_dependency = True
    #                     break
    #             if not found_dependency:
    #                 print(f"[Warning] Input '{inp_name}' for subgraph '{subgraph_name}' not found in main graph or upstream outputs!")    
    #         for output in self._subgraphs_outputs[subgraph_name]:
    #             # call_node.tensor_meta["shape"].append(
    #             #     self._graph.node_table[output].tensor_meta["shape"]
    #             # )
    #             # call_node.tensor_meta["dtype"].append(
    #             #     self._graph.node_table[output].tensor_meta["dtype"]
    #             # )
    #             call_node.tensor_meta["shape"].append(output.tensor_meta["shape"])
    #             call_node.tensor_meta["dtype"].append(output.tensor_meta["dtype"])
    #         self._call_table[subgraph_name] = call_node
    #         main_graph.add_node(call_node)
    #     # Adding GetItemOps to retrieve individual output tensors
    #     output_node = OutputOp()
    #     for i, output in enumerate(self._subgraphs_outputs[topo_order[-1]]):
    #         getitem_node = GetItemOp()
    #         getitem_node.add_argument(call_node.name)
    #         getitem_node.add_argument(i)
    #         getitem_node.name = "getitem{}".format(i)
    #         output_node.add_argument(getitem_node.name)
    #         main_graph.add_node(getitem_node)
    #     # Marking the final output of the main graph
    #     output_node.name = "output"
    #     main_graph.add_node(output_node)
    #     # Importing the main graph
    #     with ir.Location.unknown(ir.Context()):
    #         main_importer = GraphImporter(
    #             main_graph.body,
    #             main_graph._fake_params,
    #             main_graph._inputs,
    #             main_graph._func_name,
    #             main_graph._ops_registry,
    #             do_param_pack,
    #         )
    #         return main_importer.import_main_graph()
    def construct_sub_params(self, params, subgraph_entry, output_dir):
        """
        Process parameters and generate multiple weight files based on the subgraph configuration.
        
        Parameters:
        params: All separated parameters, obtained via params = dynamo_compiler.imported_params[graph]
        subgraph: A dictionary containing the ‘params’ (parameter configuration list) and ‘total_partitions’ keys,
                    where each parameter configuration includes:
                    - “index”: Index within params
                    - “split_degree”: Split shape
        output_dir: Output directory where arg0.data, arg1.data, ... files will be generated
        """
        subgraph_name, subgraph = subgraph_entry
        total_partitions = subgraph["total_partitions"]

        partition_data = [[] for _ in range(total_partitions)]
        
        for param_info in subgraph["params"]:
            idx = param_info["index"]
            split_degree = param_info["split_degree"]
            
            tensor = params[idx]
            
            np_tensor = tensor.detach().cpu().numpy()
            orig_shape = np_tensor.shape
            
            if not split_degree:
                flat = np_tensor.reshape(-1)
                for part in range(total_partitions):
                    partition_data[part].append(flat)
            
            else:
                slice_shape = tuple(split_degree)
                if len(orig_shape) != len(slice_shape):
                    raise ValueError(
                        f"参数索引 {idx} 的原始形状 {orig_shape} 与 split degree {slice_shape} 维度不匹配"
                    )
                axis = None
                for dim in range(len(orig_shape)):
                    if slice_shape[dim] * total_partitions == orig_shape[dim] and \
                    all(slice_shape[d] == orig_shape[d] for d in range(len(orig_shape)) if d != dim):
                        axis = dim
                        break
                if axis is None:
                    raise ValueError(
                        f"参数索引 {idx} 的 split degree {slice_shape} 无法与原始形状 {orig_shape} 匹配 (分区数={total_partitions})"
                    )
                for part in range(total_partitions):
                    start = part * slice_shape[axis]
                    end = (part + 1) * slice_shape[axis]
                    slicer = [slice(None)] * len(orig_shape)
                    slicer[axis] = slice(start, end)
                    sliced = np_tensor[tuple(slicer)]
                    partition_data[part].append(sliced.reshape(-1))

        for part in range(total_partitions):
            if partition_data[part]:
                concat_arr = np.concatenate(partition_data[part])
            else:
                concat_arr = np.array([])
            filename = os.path.join(output_dir, f"{subgraph_name}_arg{part}.data")
            concat_arr.tofile(filename)


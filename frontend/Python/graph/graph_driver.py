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
    parallel_num: int = 1  # 默认不并行
    ops_count: List[int] = field(default_factory=list)
    stage_boundary_op: Optional[Type] = None
    stage_boundary_op_num: int = 0 
    
    # 使用 field(default_factory=dict) 确保每个实例都有独立的字典
    paral_input_positions: Dict[Union[int, str], Any] = field(default_factory=dict)

    def get_paral_pos(self, subgraph_idx: int) -> List[int]:
        """安全地获取切分位置，如果parallel_num为1，始终返回空列表"""
        if self.parallel_num <= 1:
            return []
            
        # 1. 特定索引匹配
        if subgraph_idx in self.paral_input_positions:
            return self.paral_input_positions[subgraph_idx]
        
        # 2. 循环块匹配
        default_configs = self.paral_input_positions.get("default", [])
        if default_configs and self.ops_count:
            block_idx = (subgraph_idx - 1) % len(self.ops_count)
            # 防止索引越界
            if block_idx < len(default_configs):
                return default_configs[block_idx]
                
        return [] # 默认不切分
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
        # 对是否需要分割两种情况进行处理
      if self._parallelism < 1:
            raise ValueError("Parallelism must be greater than or equal to 1")
        
        
      self.op_groups = {}
      self.group_map_device = {}
      self._subgraphs_inputs = {}
      self._subgraphs_outputs = {}
      self._paral_op_shape = {}
        # ==========================================
        # 1: 纵向切分 (Vertical Splitting)
        # ==========================================
      self._perform_vertical_split()
        
        # ==========================================
        # 2: 完善子图输入输出信息 (Dependency Analysis)
        # ==========================================
        # 即使不横向切分，这里也会建立完整的子图-算子映射
      self._analyze_subgraph_dependencies()

        # ==========================================
        # 3: 横向切分 (Horizontal Splitting) 更新形状
        # ==========================================
        
      if self._parallelism > 1:
            self._apply_horizontal_parallelism()

      return self._subgraphs_inputs, self._subgraphs_outputs
    
    def _apply_horizontal_parallelism(self):
        """处理横着切的逻辑：根据 paral_input_positions 切分输入"""
        # === Phase 1: 处理输入 (Input Seeding) ===
        # 根据策略，修改每个子图入口处张量的形状
        for i, subgraph_name in enumerate(self.op_groups.keys()):
            paral_pos = self.strategy.get_paral_pos(i) 
                
            input_count = 0
            for node in self._subgraphs_inputs[subgraph_name]:
                # 获取原始形状
                original_shape = list(node.tensor_meta["shape"])
                # 防止配置越界
                if input_count >= len(paral_pos):
                    break
                
                split_dim = paral_pos[input_count]
                input_count += 1
                
                # 如果 split_dim != -1，说明需要在该维度切分
                if split_dim != -1 and split_dim < len(original_shape):
                    # 执行切分：维度大小除以并行度
                    original_shape[split_dim] = original_shape[split_dim] // self._parallelism
                    # 记录新形状
                    self._add_paral_op_shape(node.name, original_shape)
                self._subgraph_input_shape[subgraph_name][node.name] = original_shape
            
                
        # === Phase 2: 形状传播 (Shape Propagation) ===
        # 遍历所有算子，如果其输入被切分了，推导其输出形状
        for subgraph_name in self.op_groups.keys():
            # 最后一个算子通常是 OutputOp 或不需要推导，略过
            current_ops = self.op_groups[subgraph_name]
            
            for node in current_ops:
                # 既然是 shape inference，我们只关心那些"根据输入变输出"的算子
                # 这里的逻辑完全复用你提供的代码，针对不同算子做特殊处理


                # 1. PermuteOp
                if isinstance(node, PermuteOp):
                    if node.args[0] in self._paral_op_shape:
                        old_shape = self._paral_op_shape[node.args[0]]
                        permute_indices = node.args[1]
                        
                        try:
                            # 尝试推导
                            new_shape = [old_shape[index] for index in permute_indices]
                            self._add_paral_op_shape(node.name, new_shape)
                        except IndexError:
                            print(f"\n[ERROR] PermuteOp Shape Mismatch!")
                            print(f"  Node: {node.name}")
                            print(f"  Input Node: {node.args[0]}")
                            print(f"  Input Shape (old_shape): {old_shape}")
                            print(f"  Permute Indices: {permute_indices}")
                            print(f"  Reason: Indices require rank {max(permute_indices)+1}, but input has rank {len(old_shape)}.")
                            raise  # 重新抛出异常终止程序
                # if isinstance(node, PermuteOp):
                #     if node.args[0] in self._paral_op_shape:
                #         old_shape = self._paral_op_shape[node.args[0]]
                #         # 根据 permute 参数重排 shape
                #         new_shape = [old_shape[index] for index in node.args[1]]
                #         self._add_paral_op_shape(node.name, new_shape)

                # 2. MatmulOp
                elif isinstance(node, MatmulOp):
                    # 只要有一个输入变了，就尝试推导
                    if (node.args[0] in self._paral_op_shape) or (node.args[1] in self._paral_op_shape):
                        input1_shape = self._get_shape_from_cache_or_node(node.args[0])
                        input2_shape = self._get_shape_from_cache_or_node(node.args[1])
                        
                        # Matmul 规则：[..., M, K] x [..., K, N] -> [..., M, N]
                        # 简单起见，取 input1 的形状，把最后一维改成 input2 的最后一维
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
                    # 检查任意一个输入是否被切分
                    arg0_name = node.args[0]
                    arg1_name = node.args[1]
                    
                    is_arg0_split = isinstance(arg0_name, str) and (arg0_name in self._paral_op_shape)
                    is_arg1_split = isinstance(arg1_name, str) and (arg1_name in self._paral_op_shape)

                    if is_arg0_split or is_arg1_split:
                        shape0 = self._get_shape_from_cache_or_node(arg0_name)
                        shape1 = self._get_shape_from_cache_or_node(arg1_name)
                        
                        if shape0 and shape1:
                            # 处理广播，取最大维度
                            norm_s1, norm_s2 = self._normalize_binary_operator_shape(shape0, shape1)
                            new_shape = []
                            for d1, d2 in zip(norm_s1, norm_s2):
                                new_shape.append(max(d1, d2))
                            self._add_paral_op_shape(node.name, new_shape)

                # 5. ViewOp (Reshape) - 复杂逻辑
                elif isinstance(node, ViewOp):
                    parent = node.args[0]
                    if parent in self._paral_op_shape:
                        old_shape = self._paral_op_shape[parent] # 这里的 old_shape 是指 parent 已经被切分后的形状
                        target_shape_args = list(node.args[1]) # 这是代码里写的原始 reshape 目标
                        
                        # 计算元素总数对比
                        current_total = 1
                        for x in old_shape: current_total *= x
                        
                        target_total = 1
                        for x in target_shape_args: target_total *= x
                        
                        new_shape = target_shape_args.copy()

                        # 如果元素总数不一致，说明 Reshape 的目标也需要修改以适应切分
                        if current_total != target_total:
                            # 逻辑复用你提供的 head处理逻辑
                            old_len = len(old_shape)
                            new_len = len(new_shape)
                            
                            tmp_old = [d for d in old_shape if d != 1]
                            tmp_new = [d for d in new_shape if d != 1]

                            if len(tmp_old) == len(tmp_new):
                                if old_len < new_len:
                                    # 维度增加 (Unsqueeze类操作)，继承数值
                                    for i in range(old_len):
                                        new_shape[i+1] = old_shape[i]
                                elif old_len == new_len:
                                    # 维度不变 (1:1 Mapping): 直接对应赋值
                                    for i in range(new_len):
                                        new_shape[i] = old_shape[i]
                                else:
                                    # 维度减少 (Squeeze类操作)
                                    for i in range(new_len):
                                        new_shape[i] = old_shape[i+1]
                            else:
                                # 维度拆分或合并 (例如 [B, S, H*D] <-> [B, S, H, D])
                                if old_len < new_len:
                                    # 拆分: 最后一维除一下
                                    # 假设 new_shape[-1] 是 D，new_shape[-2] 是 H
                                    # old_shape[-1] 是 H_new * D
                                    # 我们假设 D 不变，H 被切分了
                                    if new_shape[-1] != 0:
                                        new_shape[-2] = old_shape[-1] // new_shape[-1]
                                else:
                                    # 合并: 调用推导辅助函数
                                    new_shape = self._infer_new_shape(old_shape, new_shape)
                        
                        self._add_paral_op_shape(node.name, new_shape)

                # 6. CatOp
                elif isinstance(node, CatOp):
                    # 如果拼接的输入中有被切分的，且切分维不是拼接维，则输出也被切分
                    # 简化处理：只要有一个输入在表中，就取那个形状（假设沿着非切分轴 cat）
                    tensors = node.args[0]
                    for t in tensors:
                        t_name = str(t)
                        if t_name in self._paral_op_shape:
                            self._add_paral_op_shape(node.name, self._paral_op_shape[t_name])
                            break

                # 7. IndexPutOp
                elif isinstance(node, IndexPutOp):
                    # 输出形状等于第一个参数(Target)的形状
                    target_arg = str(node.args[0])
                    target_shape = self._get_shape_from_cache_or_node(target_arg)
                    if target_shape:
                        # 这里的关键是修改 tensor_meta，因为 IndexPut 比较特殊
                        node.tensor_meta["shape"] = target_shape
                        self._add_paral_op_shape(node.name, target_shape)

                # 8. ExpandOp
                elif isinstance(node, ExpandOp) and node != self.op_groups[subgraph_name][-1]:
                    op_arg = str(node.args[0])               # expand 的输入 tensor
                    if op_arg in self._paral_op_shape:       # 分割后的 shape 存在
                        new_shape = self._paral_op_shape[op_arg]   # e.g. [1,1,1024,128]

                        # ===== 原始 expand 目标 =====
                        # node.args[1] 原来是 [1,2,6,1024,128]
                        old_new_size = node.args[1]

                        # ===== 更新规则 =====
                        # expanded shape = [1, new_group, head_per_group, seq, dim]
                        # new_group 就是分割后的 shape 中第二维
                        new_group_dim = new_shape[1]

                        # 维度映射：
                        #  old_new_size = [B,  G_old, H,  S,   D]
                        #  new_new_size = [B,  G_new, H,  S,   D]
                        new_new_size = old_new_size.copy()
                        new_new_size[1] = new_group_dim      # 将 2 → 1

                        # ===== 写回算子 =====
                        node.args[1] = new_new_size
                        self._add_paral_op_shape(node.name, new_new_size)
                        # if(node.name=="expand_1"):
                        # print(f"[TP] Updated ExpandOp {node.name} target shape: {new_new_size}")
                
                # 9. 默认处理 (直接透传)
                else:
                    # 尝试从输入中找到一个已被切分的，继承其形状 (Elementwise 默认行为)
                    for arg in node.args:
                        if isinstance(arg, str) and arg in self._paral_op_shape:
                            self._add_paral_op_shape(node.name, self._paral_op_shape[arg])
                            break
        
    
    def _get_shape_from_cache_or_node(self, arg_name):
        """辅助函数：优先从 _paral_op_shape 获取，否则从原图获取，处理常量情况"""
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
        """处理竖着切的逻辑：根据 boundary_op 和 ops_count 分组"""
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
        """分析子图间的依赖，填充 _subgraphs_inputs 和 _subgraphs_outputs"""
        # 1. 识别全图输出节点名
        total_graph_outputs = []
        for node in self._graph.body:
            if isinstance(node, OutputOp):
                total_graph_outputs.extend([arg for arg in node.args])
        
        
        # 2. 识别每个子图的输入 (跨子图引用)
        for name, ops in self.op_groups.items():
            self._subgraphs_inputs[name] = []
            self._subgraphs_outputs[name] = []
            
            op_set_in_subgraph = set(ops)
            
            for op in ops:
                # 获取该算子的所有依赖名
                deps = self._get_op_all_dependencies(op)
                
                for parent_name in deps:
                    if parent_name not in self._graph.node_table:
                        continue
                    
                    parent_op = self._graph.node_table[parent_name]
                    # 如果父节点不在当前子图中，则是当前子图的外部输入
                    if parent_op not in op_set_in_subgraph:
                        if parent_op not in self._subgraphs_inputs[name]:
                            self._subgraphs_inputs[name].append(parent_op)

        # 3. 识别每个子图的输出 (被其他子图引用，或是全图输出)
        all_inputs_of_all_subgraphs = []
        for in_list in self._subgraphs_inputs.values():
            all_inputs_of_all_subgraphs.extend(in_list)
        all_inputs_set = set(all_inputs_of_all_subgraphs)

        for name, ops in self.op_groups.items():
            for op in ops:
                # 如果这个算子被其他子图当做输入，或者是全图的最终输出
                if op in all_inputs_set or op.name in total_graph_outputs:
                    if op not in self._subgraphs_outputs[name]:
                        self._subgraphs_outputs[name].append(op)
            
            # 初始化子图依赖表
            self._subgraph_dependencies[name] = set()
        node_to_index = {node: i for i, node in enumerate(self._graph.body)}    
        for name in self._subgraphs_inputs:
            self._subgraphs_inputs[name].sort(key=lambda node: node_to_index.get(node, -1))
            
    def _get_op_all_dependencies(self, op) -> List[str]:
        """获取算子的所有父节点名称（包含 args 列表中的隐藏依赖）"""
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
                
                # === 修改 2: 处理多输出导致的嵌套 shape (修复 TypeError: get) ===
                # 如果 node_shape 是 [[10, 10], [10, 10]] 这种嵌套结构
                if node_shape and isinstance(node_shape[0], (list, tuple)):
                    node_shape = list(node_shape[0]) # 取第一个输出的形状
                else:
                    node_shape = list(node_shape)
                
                # === 修改 3: 处理多输出导致的 tuple dtype  ===
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
                # 遍历当前子图的操作，切分与权重文件相关的操作
                # 与权重文件相关的操作指参数中包含权重矩阵或参数根据权重矩阵计算获得的操作

                # ReshapeOp会改变shape,需要更新shape参数列表
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
        # 为每个子图创建一个FuncOp节点，并将这些节点添加到主图中。
        # Adding FuncOp nodes for each subgraph
        inputs0 = self._graph._inputs
        split_group = []
        param_size_group = []
        # for i, subgraph_name in [(2, list(self._subgraphs.keys())[2])]:
        for i, subgraph_name in enumerate(self._subgraphs.keys()):
            main_graph_name = f"{self._graph._func_name}{i}"
            current_param_info = {} # 存储参数索引和分割方式
            if self._parallelism > 1:  
              main_graph = Graph(
                  [],
                  [],
                  self._graph._ops_registry,
                  main_graph_name,
                  self._graph._verbose,
              )
            else:
              main_graph = Graph(
                self._graph._inputs,
                self._graph._fake_params,
                self._graph._ops_registry,
                self._graph._func_name,
                self._graph._verbose,
              )
            # 为每个子图创建一个FuncOp节点，并将这些节点添加到对应主图中。
            # FuncOp节点代表每个子图，用于主图对子图的调用
            func_node = FuncOp()
            func_node.name = subgraph_name
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs[subgraph_name]._inputs:
                func_node.add_argument(inp)
            
            outputs = self._subgraphs[subgraph_name]._outputs
            if outputs is None or self._parallelism == 1:
                for output in self._subgraphs_outputs[subgraph_name]:
                    func_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
                    func_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            else:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    func_node.tensor_meta["shape"].append(torch.Size(output_shape))
                for output in self._subgraphs_outputs[subgraph_name]:
                    func_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            main_graph.add_node(func_node)
            
            # Adding placeholder operations from the original graph
            ph_count : int = 0 #原图 PlaceholderOp 的序号
            # 记录子图中是否有权重矩阵被分割
            issplit = False
            current_param_info["params"] = []
            current_param_info["total_partitions"] = 1
            split_group.append(1)
            current_subgraph_input_names = set(n.name for n in self._subgraphs_inputs[subgraph_name])
            if(i==1):
                print("HERE!!!!!!!!!!!!")
                print(current_subgraph_input_names)
                print("HERE!!!!!!!!!!!!")
            #遍历原图所有 PlaceholderOp（按出现顺序 ph_count）， 找到哪些是当前子图需要的，然后复制成主图里的参数 placeholder。
            # 处理从原本子图中就能得到的输入 
            #新增是子图输入但不从权重文件中读取的部分（decode子图从prefill拿到的kv cache）
            maingraph_input = list(inputs0) # 初始=self._graph._inputs
            for node in self._graph.body:
                if isinstance(node, PlaceholderOp):
                    # if node in self._subgraphs_inputs[subgraph_name]:
                    if node.name in current_subgraph_input_names:
                        if(i==1):
                            print(f"1"+node.name)
                            print(f"len(self._graph._fake_params)")
                            print(+len(self._graph._fake_params))
                            print(f"ph_count")
                            print(ph_count)
                        if(len(self._graph._fake_params) > (ph_count) and self._parallelism > 1):
                            if(i==1):
                                print(f"2"+node.name)
                            main_graph._fake_params.append(self._graph._fake_params[ph_count])
                            if node.name in self._paral_op_shape.keys():
                                if(i==1):
                                    print(f"3"+node.name)
                                node._newshape = self._paral_op_shape[node.name]
                                main_graph._fake_params[-1]['shape'] = torch.Size(node._newshape)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": node._newshape}
                                )
                                issplit = True
                            else: 
                                if(i==1):
                                    print(f"4"+node.name)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": []}
                                )
                        elif(self._parallelism > 1):
                            # current_param_info["params"].append(
                            #     {"index": ph_count, "split_degree": []}
                            # )
                            # if i > 0:
                            #     node_shape = node.tensor_meta["shape"]
                            #     node_dtype = node.tensor_meta["dtype"]
                            #     input_tensor_meta = TensorMeta(node_shape, node_dtype)
                            #     maingraph_input.append(input_tensor_meta)
                            if node.name in self._paral_op_shape:
                                node_shape = self._paral_op_shape[node.name]
                            else:
                                node_shape = node.tensor_meta["shape"]
                            node_dtype = node.tensor_meta["dtype"]
                            input_tensor_meta = TensorMeta(node_shape, node_dtype)
                            maingraph_input.append(input_tensor_meta)
                        if(i==1):
                            print(f"5"+node.name)
                        main_graph.add_node(node)


                    ph_count += 1
            param_size_group.append(self.get_pack_params_size(main_graph._fake_params))
            # print("drtryughbjgyuojikb vcfgyuiojlk")
            # # print(main_graph)
            
            # for i, op in enumerate(main_graph._body):
            # # #     if op.name == "clone_4" or op.name == "expand_6" or op.name == "view_14":
            # # #         print(f"[DEBUG] {op.name}: inferred shape = {op.tensor_meta}")

            # #     # print("=" * 20 + "Graph Node" + "=" * 20)
            #     print(f"{i}  ")
            #     # print(op)
            #     print("Node: " + op.name)
            #     print("Type: " + str(op._op_type))
            #     print("Arguments: " + str(op.args))
            #     print("Parents: " + str(op._parents))
            #     print("Children: " + str(op._children))
            # print("drtryughbjgyuojikb vcfgyuiojlk")
            

            if issplit: 
                current_param_info["total_partitions"] = self._parallelism
                split_group[-1] = self._parallelism
            self._subgraph_param_info[subgraph_name] = current_param_info

            #处理来自其他子图的输入节点
            # maingraph_input = list(inputs0) # 初始=self._graph._inputs
            # Identify inputs for each subgraph
            if self._parallelism > 1:
                for node in self._subgraphs_inputs[subgraph_name]:
                    
                    # --- 调试开关：直接使用外部的 i ---
                    debug_mode = (i == 1)

                    if debug_mode:
                        print(f"\n[DEBUG] i={i} | Start Processing Node: {node.name}")
                        print(f"[DEBUG] Step 1: Checking if node is in main_graph.node_table...")

                    if (node.name not in main_graph.node_table.keys()):
                        
                        if debug_mode: print(f"[DEBUG]   -> Result: Node NOT found. Preparing to add.")
                        
                        # if debug_mode: print(self._paral_op_shape)
                # if subgraph_name in self._subgraph_input_shape and \
                #    node.name in self._subgraph_input_shape[subgraph_name]:
                #     # print("A")
                #     node_shape = self._subgraph_input_shape[subgraph_name][node.name]
                        # if debug_mode: print(self._subgraph_input_shape)                        
                        # if node.name in self._paral_op_shape.keys():
                        #     node_shape = self._paral_op_shape[node.name]
                        if subgraph_name in self._subgraph_input_shape and \
                           node.name in self._subgraph_input_shape[subgraph_name]:
                            node_shape = self._subgraph_input_shape[subgraph_name][node.name]
                            if debug_mode: print(f"[DEBUG] Step 2: Found in _paral_op_shape. Using parallel shape: {node_shape}")
                            # issplit = True
                        else:
                            node_shape = node.tensor_meta["shape"]
                            if debug_mode: print(f"[DEBUG] Step 2: Not in _paral_op_shape. Using original meta shape: {node_shape}")
                        
                        node_dtype = node.tensor_meta["dtype"]
                        
                        if debug_mode: print(f"[DEBUG] Step 3: Creating TensorMeta with Shape={node_shape}, Dtype={node_dtype}")

                        input_tensor_meta = TensorMeta(node_shape, node_dtype)
                        maingraph_input.append(input_tensor_meta)
                        
                        placeholder_node = PlaceholderOp()
                        placeholder_node.name = node.name
                        placeholder_node.tensor_meta = input_tensor_meta
                        
                        if debug_mode: print(f"[DEBUG] Step 4: Adding PlaceholderOp '{node.name}' to main_graph.")
                        main_graph.add_node(placeholder_node)
                    
                    else:
                        if debug_mode: print(f"[DEBUG]   -> Result: Node ALREADY exists in main_graph. Skipping.")
            # if self._parallelism > 1:
            #   for node in self._subgraphs_inputs[subgraph_name]:
            #     if (node.name not in main_graph.node_table.keys()):
            #       if node.name in self._paral_op_shape.keys():
            #         node_shape = self._paral_op_shape[node.name]
            #         # issplit = True
            #       else:
            #         node_shape = node.tensor_meta["shape"]
            #       node_dtype = node.tensor_meta["dtype"]
            #       input_tensor_meta = TensorMeta(node_shape, node_dtype)
            #       maingraph_input.append(input_tensor_meta)
            #       placeholder_node = PlaceholderOp()
            #       placeholder_node.name = node.name
            #       placeholder_node.tensor_meta = input_tensor_meta
            #       main_graph.add_node(placeholder_node)
            
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
            if outputs is None or self._parallelism == 1:
                for output in self._subgraphs_outputs[subgraph_name]:
                    call_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
                    call_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            else:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    call_node.tensor_meta["shape"].append(torch.Size(output_shape))
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
                if self._parallelism == 1:
                    return main_importer.import_main_graph()
                # print("=== Debug main_graph before import ===")
                # print("main_graph_name:", main_graph_name)
                # print("len(maingraph_input):", len(maingraph_input))
                # print("maingraph_input shapes/dtypes:", [ (m.shape, m.dtype) for m in maingraph_input ])
                # print("len(main_graph._fake_params):", len(main_graph._fake_params))
                # print("fake_params shapes:", [p['shape'] if isinstance(p, dict) and 'shape' in p else None for p in main_graph._fake_params])
                # print("nodes in main_graph node_table:", list(main_graph.node_table.keys()))
                # print("main_graph.body node count:", len(main_graph.body))
                self._modules[main_graph_name] = main_importer.import_main_graph()
                inputs0 = [] 
        
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
        处理参数并根据 subgraph 的配置生成多个权重文件。
        
        参数:
        params: 分离出的全部参数，由 params = dynamo_compiler.imported_params[graph]获得
        subgraph: 包含 'params'（参数配置列表） 和 'total_partitions' 键的字典，
                    其中每个参数配置包括:
                    - "index": 在 params 中的索引
                    - "split_degree": 分片形状
        output_dir: 输出目录，将在此目录中生成 arg0.data, arg1.data, ... 文件
        """
        subgraph_name, subgraph = subgraph_entry
        total_partitions = subgraph["total_partitions"]

        # 为每个分区建立列表，存放各个参数（切分后的部分）的 flattened 数组
        partition_data = [[] for _ in range(total_partitions)]
        
        # 按 subgraph["params"] 中的顺序处理每个参数
        for param_info in subgraph["params"]:
            idx = param_info["index"]
            split_degree = param_info["split_degree"]
            
            # 从参数列表中获取 tensor
            tensor = params[idx]
            
            # 将 tensor 转为 NumPy 数组
            np_tensor = tensor.detach().cpu().numpy()
            orig_shape = np_tensor.shape
            
            if not split_degree:
                # 不切分，完整 tensor 复制到每个权重矩阵
                flat = np_tensor.reshape(-1)
                for part in range(total_partitions):
                    partition_data[part].append(flat)
            
            else:
                # split_degree 给出每个切片的形状
                slice_shape = tuple(split_degree)
                if len(orig_shape) != len(slice_shape):
                    raise ValueError(
                        f"参数索引 {idx} 的原始形状 {orig_shape} 与 split degree {slice_shape} 维度不匹配"
                    )
                # 确定切分轴：slice_shape[axis] * total_partitions == orig_shape[axis]
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
                # 按轴切分
                for part in range(total_partitions):
                    start = part * slice_shape[axis]
                    end = (part + 1) * slice_shape[axis]
                    slicer = [slice(None)] * len(orig_shape)
                    slicer[axis] = slice(start, end)
                    sliced = np_tensor[tuple(slicer)]
                    partition_data[part].append(sliced.reshape(-1))
        
        # 为每个分区将所有切分后的参数拼接，并写入输出文件
        for part in range(total_partitions):
            # 若当前分区没有数据，也生成一个空文件
            if partition_data[part]:
                concat_arr = np.concatenate(partition_data[part])
            else:
                concat_arr = np.array([])
            filename = os.path.join(output_dir, f"{subgraph_name}_arg{part}.data")
            concat_arr.tofile(filename)
            
            # # 输出调试信息
            # print(f"保存分区 {part} 权重到 {filename}")
            # print(f"总元素数: {concat_arr.size}")
            # print(f"内存占用: {concat_arr.nbytes/1024**2:.2f} MB\n")

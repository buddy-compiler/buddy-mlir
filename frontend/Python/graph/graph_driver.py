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
    stage_boundary_op_num: int = 0 ,
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
        self.op_groups = self._graph.op_groups
        (
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = self.get_split_strategy()
        (
            self._subgraphs
        ) = self.build_subgraph_by_group()
        self.group_map_device = self._graph.group_map_device

        self._paral_op_shape: Dict[str, List[int]] = {}
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
        # 步骤 1: 纵向切分 (Vertical Splitting)
        # ==========================================
      self._perform_vertical_split()
        
        # ==========================================
        # 步骤 2: 完善子图输入输出信息 (Dependency Analysis)
        # ==========================================
        # 即使不横向切分，这里也会建立完整的子图-算子映射
      self._analyze_subgraph_dependencies()

        # ==========================================
        # 步骤 3: 横向切分 (Horizontal Splitting)
        # ==========================================
        
      if self._parallelism > 1:
            self._apply_horizontal_parallelism()

      return self._subgraphs_inputs, self._subgraphs_outputs
    
    def _perform_vertical_split(self):
        """处理竖着切的逻辑：根据 boundary_op 和 ops_count 分组"""
        ops_count = self.strategy.ops_count
        max_strategy_op_count = self.strategy.stage_boundary_op_num
        
        submodel_count = 0
        strategy_op_count = 0
        tsf_count = 0
        
        def new_subgraph():
            nonlocal submodel_count, tsf_count
            name = f"subgraph{submodel_count}"
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
                if node.name in self._paral_op_shape.keys():
                    node_shape = self._paral_op_shape[node.name]
                else:
                    node_shape = node.tensor_meta["shape"]
                # node_dtype = node.tensor_meta["dtype"]
                # === 修改 1: 获取原始 shape ===
                if node.name in self._paral_op_shape.keys():
                    raw_shape = self._paral_op_shape[node.name]
                else:
                    raw_shape = node.tensor_meta["shape"]
                
                # === 修改 2: 处理多输出导致的嵌套 shape (修复 TypeError: get) ===
                # 如果 raw_shape 是 [[10, 10], [10, 10]] 这种嵌套结构
                if raw_shape and isinstance(raw_shape[0], (list, tuple)):
                    node_shape = list(raw_shape[0]) # 取第一个输出的形状
                else:
                    node_shape = list(raw_shape)
                
                # === 修改 3: 处理多输出导致的 tuple dtype (保持你之前的修复) ===
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
                if isinstance(op, ViewOp) and self._parallelism > 1:
                    if op.args[0] in self._paral_op_shape.keys():
                        op._newshape = self._paral_op_shape[op.name]
                subgraph_body_list.append(op)
                # if op.name == "unsqueeze_7":
                #     print("Node: " + op.name)
                #     print("Type: " + str(op._op_type))
                #     print("Arguments: " + str(op.args))
                #     print("Parents: " + str(op._parents))
                #     print("Children: " + str(op._children))
                
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
        main_graph = Graph(
            self._graph._inputs,
            self._graph._fake_params,
            self._graph._ops_registry,
            self._graph._func_name,
            self._graph._verbose,
        )

        # Adding FuncOp nodes for each subgraph
        for subgraph_name in self._subgraphs.keys():
            func_node = FuncOp()
            func_node.name = subgraph_name
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs[subgraph_name]._inputs:
                func_node.add_argument(inp)
            for output in self._subgraphs_outputs[subgraph_name]:
                # func_node.tensor_meta["shape"].append(
                #     self._graph.node_table[output].tensor_meta["shape"]
                # )
                # func_node.tensor_meta["dtype"].append(
                #     self._graph.node_table[output].tensor_meta["dtype"]
                # )
                func_node.tensor_meta["shape"].append(output.tensor_meta["shape"])
                func_node.tensor_meta["dtype"].append(output.tensor_meta["dtype"])
            main_graph.add_node(func_node)

        # Adding placeholder operations from the original graph
        for op in self._graph.body:
            if isinstance(op, PlaceholderOp):
                main_graph.add_node(op)
        # Analysis topology order to sort subgraph call.
        topo_order = self.topological_sort_subgraph()
        if topo_order == None:
            print("Error : Graph Partitioning is illegal!")
            return None
        # Adding CallOp to invoke the single subgraph
        for i, subgraph_name in enumerate(topo_order):
            call_node = CallOp()
            call_node.name = "call{}".format(i)
            call_node.call_func_name = subgraph_name
            call_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs_inputs[subgraph_name]:
                inp_name = inp.name if hasattr(inp, 'name') else inp
                if inp_name in main_graph.node_table:
                    call_node.add_argument(inp_name)
                    continue
                found_dependency = False
                for key, value in self._subgraphs_outputs.items():
                    if inp in value:
                        call_node.add_argument(
                            arg=self._call_table[key].name,
                            arg_index=value.index(inp),
                        )
                        found_dependency = True
                        break
                if not found_dependency:
                    print(f"[Warning] Input '{inp_name}' for subgraph '{subgraph_name}' not found in main graph or upstream outputs!")    
            for output in self._subgraphs_outputs[subgraph_name]:
                # call_node.tensor_meta["shape"].append(
                #     self._graph.node_table[output].tensor_meta["shape"]
                # )
                # call_node.tensor_meta["dtype"].append(
                #     self._graph.node_table[output].tensor_meta["dtype"]
                # )
                call_node.tensor_meta["shape"].append(output.tensor_meta["shape"])
                call_node.tensor_meta["dtype"].append(output.tensor_meta["dtype"])
            self._call_table[subgraph_name] = call_node
            main_graph.add_node(call_node)
        # Adding GetItemOps to retrieve individual output tensors
        output_node = OutputOp()
        for i, output in enumerate(self._subgraphs_outputs[topo_order[-1]]):
            getitem_node = GetItemOp()
            getitem_node.add_argument(call_node.name)
            getitem_node.add_argument(i)
            getitem_node.name = "getitem{}".format(i)
            output_node.add_argument(getitem_node.name)
            main_graph.add_node(getitem_node)
        # Marking the final output of the main graph
        output_node.name = "output"
        main_graph.add_node(output_node)
        # Importing the main graph
        with ir.Location.unknown(ir.Context()):
            main_importer = GraphImporter(
                main_graph.body,
                main_graph._fake_params,
                main_graph._inputs,
                main_graph._func_name,
                main_graph._ops_registry,
                do_param_pack,
            )
            return main_importer.import_main_graph()
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

import copy
import itertools
import operator
from typing import Callable, Dict, List, Optional, Set, Any, Tuple

import torch
import torch._dynamo as torchdynamo

from torch.fx import Node
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree
from torch._inductor.decomposition import decompositions as inductor_decomp

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from torch.ao.quantization.quantize_pt2e import (
    QuantizationSpec,
    Quantizer,
    QuantizationAnnotation,
    SharedQuantizationSpec,
    prepare_pt2e,
    convert_pt2e
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
    OperatorConfig,
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_bias_qspec,
    get_weight_qspec,
)
# import torchvision
from enum import Enum

class TensorDType(Enum):
    """
    Enum class for declaring tensor data types.

    Members:
    - Int32: str
        Represents the 32-bit integer data type.
    - Int64: str
        Represents the 64-bit integer data type.
    - Float32: str
        Represents the 32-bit floating-point data type.
    - Bool: str
        Represents the boolean data type.
    """

    Int8 = "int8"
    Int32 = "int32"
    Int64 = "int64"
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    Bool = "bool"


class TensorMeta:
    """
    Store tensor metadata, including shape and data type, while overlooking raw 
    data.

    Attributes:
    - shape: tuple
        Represents the shape of the tensor.
    - dtype: str
        Represents the data type of the tensor.

    Methods:
    - __init__(shape: tuple, dtype: str) -> None:
        Initializes a new instance of the TensorMeta class with the specified 
        shape and data type.

    Example:
    meta = TensorMeta(shape=(3, 4), dtype='float32')
    # Access metadata attributes: meta.shape, meta.dtype
    """

    def __init__(self, shape, dtype) -> None:
        """
        Initialize a new instance of the TensorMeta class.

        Parameters:
        - shape: tuple
            Represents the shape of the tensor.
        - dtype: str
            Represents the data type of the tensor.
        """
        self.shape = shape
        self.dtype = dtype

def _torch_dtype_translate(dtype):
    if dtype == "torch.int64":
            return TensorDType.Int64
    elif dtype == "torch.int32":
            return TensorDType.Int32
    elif dtype == "torch.float16":
            return TensorDType.Float16
    elif dtype == "torch.float32":
            return TensorDType.Float32
    elif dtype == "torch.float64":
            return TensorDType.Float64
    elif dtype == "torch.bool":
            return TensorDType.Bool
    elif dtype == "torch.int8":
            return TensorDType.Int8
    else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

class OpType(Enum):
    """
    Enum class for declaring operation types.

    Members:
    - BroadcastType: int
        Represents a broadcast operation.
    - ElementwiseType: int
        Represents an elementwise operation.
    - ReshapeType: int
        Represents a reshape operation.
    - ReduceType: int
        Represents a reduction operation.
    - ConcatType: int
        Represents a concatenation operation.
    - PlaceholderType: int
        Represents a placeholder operation.
    - GetItemType: int
        Represents an operation to retrieve an item.

    Note: The underlying values are integers for these operation types.
    """

    BroadcastType = 0
    ElementwiseType = 1
    ReshapeType = 2
    SliceLikeType = 3
    ReduceType = 4
    ConcatType = 5
    PlaceholderType = 6
    GetItemType = 7
    Unfusable = 8


class Op:
    """
    Base class for all operations in a computational graph.

    Attributes:
    - _name: str
        The unique name of the operation node.
    - _arguments: list
        The input arguments of the operation node.
    - _keyword_arguments: dict
        The keyword arguments of the operation node.
    - _tensor_meta: dict
        The metadata of the output tensor, including shape and data type.
    - _op_type: OpType
        The type of the operation node, as defined in the OpType enum.
    """

    def __init__(self) -> None:
        """
        Initialize a new instance of the Op class.
        """
        self._name = None
        self._gm_node_name = None
        self._arguments = []
        self._keyword_arguments = {}
        self._tensor_meta: Dict = {}
        self._op_type: OpType = None
        self._children: List[str] = []
        self._parents: List[str] = []

    def add_argument(self, arg):
        """
        Add an input argument to the operation node.

        Parameters:
        - arg: Any
            The input argument to be added.
        """
        self._arguments.append(arg)

    def add_parent(self, parent: str):
        """
        Add an parent node's name to the operation node.

        Parameters:
        - parent: str
            The parent node's name to be added.
        """
        self._parents.append(parent)

    def add_children(self, child):
        """
        Add an user node's name to the operation node.

        Parameters:
        - user: str
            The user node's name to be added.
        """
        self._children.append(child)

    @property
    def args(self):
        return self._arguments

    @property
    def kwargs(self):
        return self._keyword_arguments

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def tensor_meta(self):
        return self._tensor_meta

    @tensor_meta.setter
    def tensor_meta(self, new_tensor_meta):
        self._tensor_meta = new_tensor_meta

def _create_node(
    gm_node_name: str,
    node_name: str,
    node_input: Tuple,
    node_users: List[str],
    node_output_shape: list = [],
    node_output_dtype: TensorDType = None,
    node_kwargs: Optional[Dict] = None,
):
    """
    Create buddy op node from torch aten op.

    Args:
        gm_node_name: The op node class map to buddy op by _ops_map.
        node_name: The op node name to be used.
        node_input: The args input to op node.
        node_output_shape: The list of the op node's output shape.
        node_output_dtype: The TensorDType enum type of the op node's output
        data type.
        node_kwargs: The restful attributes for op node.
    """
    buddy_node = Op()
    buddy_node._gm_node_name = gm_node_name
    buddy_node._name = node_name
    if gm_node_name == "output":
        for input_arg in node_input[0]:
            buddy_node.add_argument(str(input_arg))
        return buddy_node
    for input_arg in node_input:
        if isinstance(input_arg, torch.fx.Node):
            buddy_node.add_argument(str(input_arg))
            buddy_node.add_parent(str(input_arg))
        elif isinstance(input_arg, torch.dtype):
            buddy_node.add_argument(_torch_dtype_translate(str(input_arg)))
        else:
            buddy_node.add_argument(input_arg)
    for user in node_users:
        buddy_node.add_children(user)
    if node_kwargs is None:
        node_kwargs = {}
    buddy_node._keyword_arguments.update(node_kwargs)
    buddy_node._tensor_meta["shape"] = node_output_shape
    buddy_node._tensor_meta["dtype"] = node_output_dtype
    return buddy_node

def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True

def _is_annotated(nodes: List[Node]):
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated

class BackendQuantizer(Quantizer):

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]

    def set_global(self, quantization_config: QuantizationConfig):
        """set global QuantizationConfig used for the backend.
        QuantizationConfig is defined in torch/ao/quantization/_pt2e/quantizer/quantizer.py.
        """
        self.global_config = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """annotate nodes in the graph with observer or fake quant constructors
        to convey the desired way of quantization.
        """
        global_config = self.global_config
        self.annotate_symmetric_config(model, global_config)

        return model

    def annotate_symmetric_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ) -> torch.fx.GraphModule:
        self._annotate_linear(model, config)
        self._annotate_conv2d(model, config)
        self._annotate_maxpool2d(model, config)
        return model

    def _annotate_conv2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue

            input_qspec_map = {}
            input_act = conv_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = get_weight_qspec(quantization_config)

            bias = conv_node.args[2]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)

            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        act_qspec = get_input_act_qspec(quantization_config)
        weight_qspec = get_weight_qspec(quantization_config)
        bias_qspec = get_bias_qspec(quantization_config)
        for module_or_fn_type, partitions in module_partitions.items():
            if module_or_fn_type == torch.nn.Linear:
                for p in partitions:
                    act_node = p.input_nodes[0]
                    output_node = p.output_nodes[0]
                    weight_node = None
                    bias_node = None
                    for node in p.params:
                        weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                        if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                            weight_node = node
                        if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                            bias_node = node
                    if weight_node is None:
                        raise ValueError("No weight found in Linear pattern")
                    # find use of act node within the matched pattern
                    act_use_node = None
                    for node in p.nodes:
                        if node in act_node.users:  # type: ignore[union-attr]
                            act_use_node = node
                            break
                    if act_use_node is None:
                        raise ValueError(
                            "Could not find an user of act node within matched pattern."
                        )
                    if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                        _annotate_input_qspec_map(
                            act_use_node,
                            act_node,
                            act_qspec,
                        )
                    if bias_node and _is_annotated([bias_node]) is False:
                        _annotate_output_qspec(bias_node, bias_qspec)
                    if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                        _annotate_output_qspec(weight_node, weight_qspec)
                    if _is_annotated([output_node]) is False:
                        _annotate_output_qspec(output_node, act_qspec)
                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_maxpool2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d]
        )
        maxpool_partitions = list(itertools.chain(*module_partitions.values()))
        for maxpool_partition in maxpool_partitions:
            output_node = maxpool_partition.output_nodes[0]
            maxpool_node = None
            for n in maxpool_partition.nodes:
                if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                    maxpool_node = n
            if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
                continue

            input_act = maxpool_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                _annotated=True,
            )
            output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=SharedQuantizationSpec((input_act, maxpool_node)),
                _annotated=True,
            )

    def validate(self, model: torch.fx.GraphModule) -> None:
        """validate if the annotated graph is supported by the backend"""
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []

def get_symmetric_quantization_config():
    # act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = \
    #     HistogramObserver
    # act_quantization_spec = QuantizationSpec(
    #     dtype=torch.int8,
    #     quant_min=-128,
    #     quant_max=127,
    #     qscheme=torch.per_tensor_affine,
    #     is_dynamic=False,
    #     observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
    # )
    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    act_quantization_spec = QuantizationSpec(
        dtype=torch.float,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr
    )
    act_quantization_spec = None

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PerChannelMinMaxObserver
    # weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        # qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args),
    )

    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float,
        observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    return quantization_config


def _compile_fx(
    gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
) -> Any:
    """
    Compiles the provided FX Graph to Buddy Graph.

    Args:
        gm (torch.fx.GraphModule): The GraphModule to be compiled.
        inputs (List[torch.Tensor]): The input tensors.

    Returns:
        dynamo_run: The function of the ahead-of-time compiled module,
        return for torchdynamo's call.
    """

    params = {
        **dict(gm.named_parameters(remove_duplicate=False)),
        **dict(gm.named_buffers(remove_duplicate=False)),
    }
    params_flat, _ = pytree.tree_flatten(params)

    # if self._verbose:
    #     print("Graph in tabular form:")
    #     gm.graph.print_tabular()

    def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):
        """Compile a FX graph in Aten/Prims IR to MLIR."""
        nonlocal params_flat
        func_inputs = []
        for inp in _inputs[len(params_flat) :]:
            inp_shape = inp.shape
            inp_dtype = _torch_dtype_translate(str(inp.dtype))
            # func_inputs.append(TensorMeta(inp_shape, inp_dtype))
        fake_params = []
        for param in params_flat:
            pass
            param_dtype = _torch_dtype_translate(str(param.dtype))
        #     fake_params.append(TensorMeta(param.shape, param_dtype))
        # graph = Graph(
        #     func_inputs,
        #     fake_params,
        #     self._ops_registry,
        #     self._func_name,
        # )
        for gm_node in _gm.graph.nodes:
            node_users = []
            for user in gm_node.users.keys():
                node_users.append(str(user))
            if gm_node.op == "placeholder":
                pass
                node_dtype = _torch_dtype_translate(
                    str(gm_node.meta["tensor_meta"].dtype)
                )
                buddy_node = _create_node(
                    gm_node.op,
                    gm_node.name,
                    gm_node.args,
                    node_users,
                    gm_node.meta["tensor_meta"].shape,
                    node_dtype,
                )

            elif gm_node.op == "output":
                buddy_node = _create_node(
                    gm_node.op,
                    gm_node.name,
                    gm_node.args,
                    node_users
                )

            elif gm_node.target is operator.getitem:
                node_dtype = _torch_dtype_translate(
                    str(gm_node.meta["tensor_meta"].dtype)
                )
                buddy_node = _create_node(
                    str(gm_node.target.__name__),
                    gm_node.name,
                    gm_node.args,
                    node_users,
                    gm_node.meta["tensor_meta"].shape,
                    node_dtype,
                )

            else:
                tensor_meta = gm_node.meta.get("tensor_meta")
                val = gm_node.meta.get("val")
                num_returns = len(gm_node.target._schema.returns)
                if num_returns == 1:
                    node_dtype = _torch_dtype_translate(
                        str(tensor_meta.dtype)
                    )
                    node_shape = tensor_meta.shape
                elif num_returns > 1:
                    node_dtype = tuple(
                        [
                            _torch_dtype_translate(str(val_item.dtype))
                            for val_item in val
                        ]
                    )
                    node_shape = tuple([val_item.shape for val_item in val])
                else:
                    raise RuntimeError("Zero returns is not supported.")

                buddy_node = _create_node(
                    str(gm_node.target.__name__),
                    gm_node.name,
                    gm_node.args,
                    node_users,
                    node_shape,
                    node_dtype,
                    node_kwargs=gm_node.kwargs,
                )

            # graph.add_node(buddy_node)
            print(buddy_node.name, buddy_node._gm_node_name)
            print(buddy_node._tensor_meta)
            print(buddy_node._parents)
            print(buddy_node._children)
            print('')
        # transform_list = [maxpool2d_simplify]
        # graph.perform(transform_list)
        # self._imported_graphs.append(graph)
        # self._imported_params[graph] = params_flat
        return _gm.forward

    return aot_module_simplified(
        gm,
        inputs,
        fw_compiler=_compiler,
        decompositions=inductor_decomp,
    )

def importer(model, *args, **kwargs) -> None:
        """
        Imports the provided model as MLIR module and flat parameters.

        Args:
            model: The model to be imported.
            args: Arguments for the model.
            kwargs: Keyword arguments for the model.

        """
        model_opt = torchdynamo.optimize(_compile_fx)(model)
        model_opt(*args, **kwargs)

if __name__ == "__main__":
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 10, bias=False)

        def forward(self, x):
            return self.linear(x)
    # example_inputs = (torch.randn(1, 3, 224, 224),)
    # m = torchvision.models.resnet18().eval()
    float_model = M()
    example_inputs = (torch.randn(8, 5),)
    # prg = torch.export.export(float_model, args=example_inputs)
    # m = prg.module()
    # m_copy = copy.deepcopy(m)
    # program capture
    original_gm, guards = torchdynamo.export(
        float_model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )
    print('original model is:')
    original_gm.print_readable()
    quantizer = BackendQuantizer()
    operator_config = get_symmetric_quantization_config()
    quantizer.set_global(operator_config)
    # Note: ``prepare_pt2e_quantizer`` will be updated to ``prepare_pt2e`` soon
    prepared_gm = prepare_pt2e(original_gm, quantizer)
    after_prepare_result = prepared_gm(*example_inputs)
    converted_gm = convert_pt2e(prepared_gm)
    print("converted module is:")
    converted_gm.print_readable()
    
    print('\n\n=======================================')
    print("original graph is:")
    importer(float_model, *example_inputs)
    # _compile_fx(original_gm, example_inputs)
    
    print('\n\n=======================================')
    print("quantized graph is:")
    converted_gm.recompile()
    importer(converted_gm, *example_inputs)
    # _compile_fx(converted_gm, example_inputs)
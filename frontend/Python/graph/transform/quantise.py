from .. import Graph
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list
from ..type import TensorDType

def quantise_graph(graph: Graph, target_dtype: TensorDType = TensorDType.Int8):
    """
    Quantise the weights of the 
    
    :param graph: Description
    :type graph: Graph
    """

    for input_idx in graph._inputs:

        node = graph._body[input_idx]
        print(f"{node._op_type}, {node._name}, {node._tensor_meta}\n")

    print("...")

    # Quantise all the params.
    for param_idx in graph._fake_params:
        node = graph._body[param_idx]
        node._tensor_meta["dtype"] = target_dtype
        
        print(f"{node._op_type}, {node._name}, {node._tensor_meta}\n")



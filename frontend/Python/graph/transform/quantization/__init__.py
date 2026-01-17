from ..quantise import quantise_graph
from .weight_only_channel_wise import WeightOnlyQuantization

def weight_only_channel_wise(
    graph
):
    quantise_graph(
        graph=graph,
        quantization_method=WeightOnlyQuantization(),
    )

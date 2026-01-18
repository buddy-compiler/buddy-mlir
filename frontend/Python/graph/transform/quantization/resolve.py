from ... import Graph, NodeType
from ...operation import *
from ... import DeviceType
from torch.fx.immutable_collections import immutable_list
from ...type import TensorDType

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import torch


def dequantize_unused_quantization(node: Op):
    if not isinstance(node, PlaceholderOp):
        return

    # need to check if it had any usages.

def resolve_graph(graph: Graph):
    """
    This pass takes the output of the quantization pass,
    and resolves any "empty quantizations", and other artifacts.
    """


from .graph.op_def import *
torch_ops_map = {
    "output": OutputOp,
    "placeholder": PlaceholderOp,
    "mm.default": MatmulOp
}
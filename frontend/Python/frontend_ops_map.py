# ===- frontend_ops_map.py -----------------------------------------------------
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
# This is the ops map between frontends and Buddy Graph IR.
#
# ===---------------------------------------------------------------------------
from .graph.op_def import *

torch_ops_map = {
    "output": OutputOp,
    "placeholder": PlaceholderOp,
    "arange.start": ArangeOp,
    "arange.default": ArangeOp,
    "unsqueeze.default": UnsqueezeOp,
    "view.default": ViewOp,
    "ones.default": OnesOp,
    "full.default": FullOp,
    "lt.Tensor": LessthanOp,
    "embedding.default": EmbeddingOp,
    "masked_fill.Scalar": MaskedFillOp,
    "slice.Tensor": SliceOp,
    "expand.default": ExpandOp,
    "_to_copy.default": ToCopyOp,
    "rsub.Scalar": RsubOp,
    "pow.Tensor_Scalar": PowOp,
    "mean.dim": MeanOp,
    "rsqrt.default": RsqrtOp,
    "mul.Tensor": MulOp,
    "t.default": TransposeOp,
    "mm.default": MatmulOp,
    "transpose.int": TransposeSpecificDimOp,
    "index.Tensor": IndexOp,
    "neg.default": NegOp,
    "cat.default": CatOp,
    "squeeze.dim": SqueezeOp,
    "bmm.default": BatchMatmulOp,
    "div.Tensor": DivOp,
    "_softmax.default": SoftmaxOp,
    "clone.default": CloneOp,
    "silu.default": SiluOp,
    "add.Tensor": AddOp,
    "addmm.default": AddMMOp,
    "permute.default": PermuteOp,
    "convert_element_type.default": ConvertElementTypeOp,
    "sum.dim_IntList": SumDimOp,
    "tanh.default": TanhOp,
    "sub.Tensor": SubOp,
    "var_mean.correction": VarMeanOp,
    "amax.default": AmaxOp,
    "select.int": SelectOp,
    "exp.default": ExpOp,
    "erf.default": ErfOp,
    "getitem": GetItemOp,
}

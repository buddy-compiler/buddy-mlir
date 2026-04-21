# ===- __init__.py -------------------------------------------------------------
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
# Init the packages in transform directory.
#
# ===---------------------------------------------------------------------------

from .eliminate_matmul_transpose_reshape import (
    eliminate_matmul_transpose_reshape,
)
from .eliminate_weight_transpose import eliminate_transpose
from .fuse_ops import (
    apply_classic_fusion,
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from .onednn_replace import (
    replace_matmul_with_onednn,
    replace_matmul_with_onednn_selective,
)
from .quantization import (
    weight_only_channel_wise,
)
from .rand_replace import RUNTIME_RNG_TRANSFORMS
from .runtime_matmul_replace import (
    replace_matmul_with_buddy_runtime,
)
from .useless_op_eliminate import maxpool2d_simplify

__all__ = [
    "apply_classic_fusion",
    "eliminate_matmul_transpose_reshape",
    "eliminate_transpose",
    "flash_attention_prefill",
    "gqa_attention_fusion",
    "maxpool2d_simplify",
    "replace_matmul_with_buddy_runtime",
    "replace_matmul_with_onednn",
    "replace_matmul_with_onednn_selective",
    "RUNTIME_RNG_TRANSFORMS",
    "simply_fuse",
    "weight_only_channel_wise",
]

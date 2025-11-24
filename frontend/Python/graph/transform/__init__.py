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

from .fuse_ops import simply_fuse, apply_classic_fusion
from .useless_op_eliminate import maxpool2d_simplify
from .eliminate_weight_transpose import eliminate_transpose
from .eliminate_matmul_transpose_reshape import (
    eliminate_matmul_transpose_reshape,
)
from .onednn_replace import (
    replace_matmul_with_onednn,
    replace_matmul_with_onednn_selective,
)

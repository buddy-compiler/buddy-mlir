# ===------------------------ vir.py -------------------------------------------
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
# The Dynamic Vector IR (VIR) dialect: its ops (generated from VIROps.td),
# the DynamicVectorType and ScalingFactorAttr classes (the _buddyDialectsVIR
# extension) and a SetVLOp that builds its region's block.
#
# ===---------------------------------------------------------------------------

from .._mlir_libs._buddyDialectsVIR import *
from ._vir_ops_gen import *
from ._vir_ops_gen import _Dialect

try:
    from ..ir import *
    from ._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from collections.abc import Sequence


@_ods_cext.register_operation(_Dialect, replace=True)
class SetVLOp(SetVLOp):
    """vir.set_vl with the block of its region: build the ops of the region
    with `InsertionPoint(op.body)`, ending with `vector.yield` of the
    results (none without results)."""

    def __init__(
        self,
        vl,
        results: Sequence[Type] | None = None,
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(list(results or []), vl, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self) -> Block:
        """The block of the region."""
        return self.regions[0].blocks[0]

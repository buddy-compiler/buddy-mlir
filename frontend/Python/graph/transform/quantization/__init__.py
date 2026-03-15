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
# Quantization pass infrastructure for Buddy graph IR.
#
# ===---------------------------------------------------------------------------

from .passes import (
    weight_only_channel_wise,
    weight_only_int4_f16_channel_wise,
    w8a8_channel_wise,
)

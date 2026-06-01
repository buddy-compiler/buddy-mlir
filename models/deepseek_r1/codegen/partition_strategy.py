#!/usr/bin/env python3
# ===- partition_strategy.py - DeepSeek R1 layer split strategy -----------===//
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
# ===----------------------------------------------------------------------===//

import os

from buddy.compiler.graph import SplitStrategy
from buddy.compiler.graph.operation import PowOp


def layer_split_strategy(kind: str) -> SplitStrategy:
    """Return the DeepSeek R1 layer split strategy.

    This mirrors the existing BuddyTensorParallel DeepSeek importer, but is used
    here only for vertical layer decomposition. parallel_num remains 1 so
    weights and tensor shapes are not horizontally sharded.
    """
    if kind == "prefill":
        return SplitStrategy(
            name="deepseek_r1_prefill_layers",
            parallel_num=1,
            ops_count=[6, 50, 2, 6, 11, 2],
            stage_boundary_op=PowOp,
            stage_boundary_op_num=57,
        )
    if kind == "decode":
        decode_split_mode = os.environ.get("BUDDY_DSR1_DECODE_SPLIT", "pow")
        if decode_split_mode == "fine":
            return SplitStrategy(
                name="deepseek_r1_decode_layers_fine",
                parallel_num=1,
                ops_count=[6, 44, 2, 6, 11, 2],
                stage_boundary_op=PowOp,
                stage_boundary_op_num=57,
            )
        return SplitStrategy(
            name="deepseek_r1_decode_pow_boundaries",
            parallel_num=1,
            ops_count=[],
            stage_boundary_op=PowOp,
            stage_boundary_op_num=57,
        )
    raise ValueError(f"unknown DeepSeek R1 layer split kind: {kind}")

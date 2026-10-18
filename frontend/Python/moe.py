# ===- moe.py ------------------------------------------------------------------
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
# Mixture-of-experts layers as one graph node.
#
# HF transformers computes the experts of a MoE block (Qwen3-MoE,
# Qwen3-VL-MoE) in a loop over the experts that the tokens select: the loop
# depends on the data and cannot be traced into a static graph. The custom
# op buddy::moe_experts computes the same thing and is traced as one node
# (MoEExpertsOp), which a backend replaces with a kernel (k3_w4.py).
#
#   out[t] = sum_j w[t, j] * down[e] (silu(g) * u),  e = idx[t, j],
#            [g; u] = gate_up[e] x[t]
#
# with x [T, H], idx [T, K] (int64), w [T, K], gate_up [E, 2I, H] (the gate
# rows first) and down [E, H, I], the layout of the HF expert parameters.
#
# use_buddy_moe_experts(model) makes the experts of a loaded HF model call it.
#
# ===---------------------------------------------------------------------------

import torch


@torch.library.custom_op("buddy::moe_experts", mutates_args=())
def moe_experts(
    x: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    inter = down.shape[2]
    out = torch.zeros_like(x)
    for e in torch.unique(idx).tolist():
        rows, slots = torch.where(idx == e)
        gu = x[rows] @ gate_up[e].t()
        h = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
        y = (h @ down[e].t()) * w[rows, slots, None]
        out.index_add_(0, rows, y.to(out.dtype))
    return out


@moe_experts.register_fake
def _(x, idx, w, gate_up, down):
    return torch.empty_like(x)


def _experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return torch.ops.buddy.moe_experts(
        hidden_states,
        top_k_index,
        top_k_weights.to(hidden_states.dtype),
        self.gate_up_proj,
        self.down_proj,
    )


def use_buddy_moe_experts(model) -> None:
    """Make the experts of `model` (HF, loaded) call buddy::moe_experts.
    The implementation is registered with transformers' ExpertsInterface as
    "buddy"; transformers checks the name when a model is created, so it is
    set on the loaded model's config."""
    from transformers.integrations.moe import ExpertsInterface

    ExpertsInterface.register("buddy", _experts_forward)
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_experts_implementation"):
            config._experts_implementation = "buddy"

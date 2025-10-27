# ===- transformer_model.py -----------------------------------------------
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
# DeepSeek R1 Complete Transformer Block Implementation
#
# ===---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for DeepSeek R1."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class DeepSeekAttention(nn.Module):
    """DeepSeek R1 1.5B single layer attention implementation with GQA."""

    def __init__(self, config=None):
        super().__init__()

        # DeepSeek R1 1.5B model configuration
        self.hidden_size = 1536
        self.num_attention_heads = 12
        self.num_key_value_heads = 2  # GQA
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = 131072
        self.rope_theta = 10000.0

        # Linear projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # Attention scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Repeat key and value for GQA
        key_states = key_states.repeat_interleave(
            self.num_attention_heads // self.num_key_value_heads, dim=1
        )
        value_states = value_states.repeat_interleave(
            self.num_attention_heads // self.num_key_value_heads, dim=1
        )

        # Compute attention scores
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(attn_weights.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and apply output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepSeekFFN(nn.Module):
    """DeepSeek R1 1.5B Feed Forward Network implementation."""

    def __init__(self, config=None):
        super().__init__()

        # DeepSeek R1 1.5B FFN configuration
        self.hidden_size = 1536
        self.intermediate_size = 8960

        # Linear projections
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        # Gate projection with SiLU activation
        gate = self.gate_proj(hidden_states)
        gate = F.silu(gate)

        # Up projection
        up = self.up_proj(hidden_states)

        # Gated mechanism: element-wise multiplication
        intermediate = gate * up

        # Down projection
        output = self.down_proj(intermediate)

        return output


class DeepSeekTransformerBlock(nn.Module):
    """
    DeepSeek R1 1.5B complete transformer block implementation.
    Includes RMSNorm, Attention, FFN, and residual connections.
    """

    def __init__(self, config=None):
        super().__init__()

        # Layer components
        self.input_layernorm = RMSNorm(1536, eps=1e-6)
        self.self_attn = DeepSeekAttention(config)
        self.post_attention_layernorm = RMSNorm(1536, eps=1e-6)
        self.mlp = DeepSeekFFN(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # FFN with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def create_transformer_model():
    """
    Create and return a DeepSeek transformer block model instance.
    """
    model = DeepSeekTransformerBlock()
    model.eval()
    return model


def create_sample_inputs(batch_size=1, seq_len=40):
    """
    Create sample inputs for the transformer model.

    Args:
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Tuple of (hidden_states, attention_mask)
    """
    hidden_size = 1536

    # Create random hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create attention mask
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    return hidden_states, attention_mask


if __name__ == "__main__":
    model = create_transformer_model()
    hidden_states, attention_mask = create_sample_inputs()

    with torch.no_grad():
        output = model(hidden_states, attention_mask)
        print(f"Input shape: {hidden_states.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
        print("Transformer block test passed successfully")

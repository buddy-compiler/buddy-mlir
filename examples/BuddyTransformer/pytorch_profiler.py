# ===- pytorch_profiler.py -----------------------------------------------
#
# PyTorch-level Operator Profiling for Transformer Model
#
# ===---------------------------------------------------------------------------

import torch
import time
import json
from collections import defaultdict
from contextlib import contextmanager
from transformer_model import create_transformer_model, create_sample_inputs

class OperatorProfiler:
    """PyTorch operator-level profiler using hooks."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_op = None
        self.start_time = None
        self.hooks = []
        
    def _pre_hook(self, module, input):
        """Hook called before module execution."""
        self.current_op = module.__class__.__name__
        self.start_time = time.perf_counter()
        
    def _post_hook(self, module, input, output):
        """Hook called after module execution."""
        if self.start_time is not None:
            end_time = time.perf_counter()
            duration_ms = (end_time - self.start_time) * 1000
            self.timings[self.current_op].append(duration_ms)
            
    def register_hooks(self, model):
        """Register hooks on all modules in the model."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(self._post_hook)
                pre_hook = module.register_forward_pre_hook(self._pre_hook)
                self.hooks.extend([hook, pre_hook])
                
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_report(self):
        """Generate profiling report."""
        report = {}
        for op_name, times in self.timings.items():
            if times:
                report[op_name] = {
                    'avg_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'total_calls': len(times),
                    'total_time_ms': sum(times)
                }
        return report
        
    def print_report(self):
        """Print detailed profiling report."""
        report = self.get_report()
        
        print("\n=== PyTorch Operator Profiling Report ===")
        print(f"{'Operator':<20} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Calls':<8} {'Total (ms)':<12}")
        print("-" * 80)
        
        # Sort by total time descending
        sorted_ops = sorted(report.items(), key=lambda x: x[1]['total_time_ms'], reverse=True)
        
        for op_name, stats in sorted_ops:
            print(f"{op_name:<20} {stats['avg_time_ms']:<10.3f} {stats['min_time_ms']:<10.3f} "
                  f"{stats['max_time_ms']:<10.3f} {stats['total_calls']:<8} {stats['total_time_ms']:<12.3f}")
                  
    def save_report(self, filename):
        """Save profiling report to JSON file."""
        report = self.get_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Profiling report saved to: {filename}")

class DetailedOperatorProfiler:
    """More detailed profiler that tracks specific operations."""
    
    def __init__(self):
        self.op_timings = defaultdict(list)
        self.layer_timings = defaultdict(list)
        
    @contextmanager
    def profile_operation(self, op_name):
        """Context manager for profiling specific operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.op_timings[op_name].append(duration_ms)
            
    def profile_transformer_components(self, model, hidden_states, attention_mask, num_iterations=100):
        """Profile individual transformer components."""
        
        print(f"Profiling transformer components for {num_iterations} iterations...")
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(hidden_states, attention_mask)
                
        # Profile individual components
        for i in range(num_iterations):
            with torch.no_grad():
                # Input LayerNorm
                with self.profile_operation("InputLayerNorm"):
                    normed_input = model.input_layernorm(hidden_states)
                
                # Self Attention
                with self.profile_operation("SelfAttention_Total"):
                    attn_output = model.self_attn(normed_input, attention_mask)
                
                # Residual connection 1
                with self.profile_operation("ResidualConnection1"):
                    hidden_states_after_attn = hidden_states + attn_output
                
                # Post Attention LayerNorm
                with self.profile_operation("PostAttentionLayerNorm"):
                    normed_attn = model.post_attention_layernorm(hidden_states_after_attn)
                
                # FFN
                with self.profile_operation("FFN_Total"):
                    ffn_output = model.mlp(normed_attn)
                
                # Residual connection 2
                with self.profile_operation("ResidualConnection2"):
                    final_output = hidden_states_after_attn + ffn_output
                    
        # Profile attention sub-components
        self._profile_attention_components(model.self_attn, normed_input, attention_mask, num_iterations)
        
        # Profile FFN sub-components
        self._profile_ffn_components(model.mlp, normed_attn, num_iterations)
        
    def _profile_attention_components(self, attention_module, hidden_states, attention_mask, num_iterations):
        """Profile attention sub-components."""
        
        for i in range(num_iterations):
            with torch.no_grad():
                batch_size, seq_len, _ = hidden_states.size()
                
                # Q, K, V projections
                with self.profile_operation("Attention_QProjection"):
                    query_states = attention_module.q_proj(hidden_states)
                    
                with self.profile_operation("Attention_KProjection"):
                    key_states = attention_module.k_proj(hidden_states)
                    
                with self.profile_operation("Attention_VProjection"):
                    value_states = attention_module.v_proj(hidden_states)
                
                # Reshape operations
                with self.profile_operation("Attention_Reshape"):
                    query_states = query_states.view(batch_size, seq_len, attention_module.num_attention_heads, attention_module.head_dim).transpose(1, 2)
                    key_states = key_states.view(batch_size, seq_len, attention_module.num_key_value_heads, attention_module.head_dim).transpose(1, 2)
                    value_states = value_states.view(batch_size, seq_len, attention_module.num_key_value_heads, attention_module.head_dim).transpose(1, 2)
                
                # GQA repeat
                with self.profile_operation("Attention_GQARepeat"):
                    key_states = key_states.repeat_interleave(attention_module.num_attention_heads // attention_module.num_key_value_heads, dim=1)
                    value_states = value_states.repeat_interleave(attention_module.num_attention_heads // attention_module.num_key_value_heads, dim=1)
                
                # Attention computation
                with self.profile_operation("Attention_MatMul1"):
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention_module.scale
                
                # Softmax
                with self.profile_operation("Attention_Softmax"):
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                
                # Attention application
                with self.profile_operation("Attention_MatMul2"):
                    attn_output = torch.matmul(attn_weights, value_states)
                
                # Output projection
                with self.profile_operation("Attention_OutputProjection"):
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, attention_module.hidden_size)
                    attn_output = attention_module.o_proj(attn_output)
                    
    def _profile_ffn_components(self, ffn_module, hidden_states, num_iterations):
        """Profile FFN sub-components."""
        
        for i in range(num_iterations):
            with torch.no_grad():
                # Gate projection
                with self.profile_operation("FFN_GateProjection"):
                    gate = ffn_module.gate_proj(hidden_states)
                
                # SiLU activation
                with self.profile_operation("FFN_SiLUActivation"):
                    gate = torch.nn.functional.silu(gate)
                
                # Up projection
                with self.profile_operation("FFN_UpProjection"):
                    up = ffn_module.up_proj(hidden_states)
                
                # Gating
                with self.profile_operation("FFN_Gating"):
                    intermediate = gate * up
                
                # Down projection
                with self.profile_operation("FFN_DownProjection"):
                    output = ffn_module.down_proj(intermediate)
                    
    def print_report(self):
        """Print detailed profiling report."""
        print("\n=== Detailed Operator Profiling Report ===")
        print(f"{'Operation':<30} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Calls':<8} {'Total (ms)':<12}")
        print("-" * 100)
        
        # Sort by total time descending
        sorted_ops = sorted(self.op_timings.items(), 
                           key=lambda x: sum(x[1]), reverse=True)
        
        for op_name, times in sorted_ops:
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                calls = len(times)
                
                print(f"{op_name:<30} {avg_time:<10.3f} {min_time:<10.3f} "
                      f"{max_time:<10.3f} {calls:<8} {total_time:<12.3f}")

def main():
    """Main profiling function."""
    
    # Create model and inputs
    model = create_transformer_model()
    hidden_states, attention_mask = create_sample_inputs(batch_size=1, seq_len=40)
    
    print("=== PyTorch Transformer Profiling ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input shape: {hidden_states.shape}")
    
    # Method 1: Hook-based profiling
    print("\n1. Hook-based Profiling:")
    hook_profiler = OperatorProfiler()
    hook_profiler.register_hooks(model)
    
    # Run inference multiple times
    num_iterations = 100
    for i in range(num_iterations):
        with torch.no_grad():
            _ = model(hidden_states, attention_mask)
    
    hook_profiler.print_report()
    hook_profiler.save_report("pytorch_hook_profiling.json")
    hook_profiler.remove_hooks()
    
    # Method 2: Detailed component profiling
    print("\n2. Detailed Component Profiling:")
    detailed_profiler = DetailedOperatorProfiler()
    detailed_profiler.profile_transformer_components(model, hidden_states, attention_mask, num_iterations)
    detailed_profiler.print_report()

if __name__ == "__main__":
    main()

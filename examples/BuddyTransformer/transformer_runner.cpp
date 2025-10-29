// ===- transformer_runner.cpp ---------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ===---------------------------------------------------------------------------
//
// DeepSeek R1 Transformer Block Performance Runner
//
// ===---------------------------------------------------------------------------

#include <buddy/Core/Container.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// External function declaration (generated from MLIR)
extern "C" {
void _mlir_ciface_subgraph0(
    MemRef<float, 3> *result, MemRef<float, 3> *hidden_states1,
    MemRef<float, 3> *hidden_states2, MemRef<float, 1> *input_layernorm_weight,
    MemRef<float, 2> *q_weight, MemRef<float, 2> *k_weight,
    MemRef<float, 2> *v_weight, MemRef<long long, 2> *attention_mask,
    MemRef<float, 2> *o_weight, MemRef<float, 3> *hidden_states3,
    MemRef<float, 1> *post_attention_layernorm_weight,
    MemRef<float, 2> *gate_proj_weight, MemRef<float, 2> *up_proj_weight,
    MemRef<float, 2> *down_proj_weight);
}

// Load parameters from binary file
void loadParameters(const std::string &filename, MemRef<float, 1> &params) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open parameter file: " + filename);
  }

  file.read(reinterpret_cast<char *>(params.getData()),
            sizeof(float) * params.getSize());

  if (file.fail()) {
    throw std::runtime_error("Error reading parameter file");
  }
}

// Generate random input data
void fillRandomData(MemRef<float, 3> &memref, float min_val = -1.0f,
                    float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  float *data = memref.getData();
  for (size_t i = 0; i < memref.getSize(); ++i) {
    data[i] = dis(gen);
  }
}

int main(int argc, char **argv) {
  // Configuration
  const int64_t batch_size = 1;
  const int64_t seq_len = 40;
  const int64_t hidden_size = 1536;
  const int64_t intermediate_size = 8960;
  const int64_t num_attention_heads = 12;
  const int64_t num_key_value_heads = 2;
  const int num_warmup = 5;
  const int num_iterations = 100;
  const size_t total_params = 46795776; // From the MLIR analysis

  std::cout << "DeepSeek R1 Transformer Block Performance Test\n";
  std::cout << "===============================================\n";
  std::cout << "Batch size: " << batch_size << "\n";
  std::cout << "Sequence length: " << seq_len << "\n";
  std::cout << "Hidden size: " << hidden_size << "\n";
  std::cout << "Intermediate size: " << intermediate_size << "\n";
  std::cout << "Attention heads: " << num_attention_heads << "\n";
  std::cout << "Key-value heads: " << num_key_value_heads << "\n";
  std::cout << "Warmup iterations: " << num_warmup << "\n";
  std::cout << "Test iterations: " << num_iterations << "\n\n";

  try {
    // Initialize containers
    std::cout << "Initializing containers...\n";

    // Set up paths
    std::string buddyTransformerBuildDir = BUDDY_TRANSFORMER_EXAMPLE_BUILD_PATH;
    const std::string paramsDir = buddyTransformerBuildDir + "/arg0.data";

    std::cout << "Looking for parameters at: " << paramsDir << "\n";

    // Parameters container
    MemRef<float, 1> paramsContainer({total_params});
    loadParameters(paramsDir, paramsContainer);
    std::cout << "Loaded " << paramsContainer.getSize() << " parameters\n";

    // Split parameters into weight matrices based on MLIR layout
    float *params_data = paramsContainer.getData();
    size_t offset = 0;

    // RMSNorm weights (input_layernorm)
    MemRef<float, 1> input_layernorm_weight({hidden_size});
    std::copy(params_data + offset, params_data + offset + hidden_size,
              input_layernorm_weight.getData());
    offset += hidden_size;

    // Q projection: 1536 x 1536
    MemRef<float, 2> q_weight({hidden_size, hidden_size});
    const size_t q_size = hidden_size * hidden_size;
    std::copy(params_data + offset, params_data + offset + q_size,
              q_weight.getData());
    offset += q_size;

    // K projection: 256 x 1536 (GQA: 2 heads * 128 dim)
    const size_t kv_proj_size =
        num_key_value_heads * (hidden_size / num_attention_heads) * hidden_size;
    MemRef<float, 2> k_weight(
        {num_key_value_heads * (hidden_size / num_attention_heads),
         hidden_size});
    std::copy(params_data + offset, params_data + offset + kv_proj_size,
              k_weight.getData());
    offset += kv_proj_size;

    // V projection: 256 x 1536 (GQA: 2 heads * 128 dim)
    MemRef<float, 2> v_weight(
        {num_key_value_heads * (hidden_size / num_attention_heads),
         hidden_size});
    std::copy(params_data + offset, params_data + offset + kv_proj_size,
              v_weight.getData());
    offset += kv_proj_size;

    // O projection: 1536 x 1536
    MemRef<float, 2> o_weight({hidden_size, hidden_size});
    const size_t o_size = hidden_size * hidden_size;
    std::copy(params_data + offset, params_data + offset + o_size,
              o_weight.getData());
    offset += o_size;

    // RMSNorm weights (post_attention_layernorm)
    MemRef<float, 1> post_attention_layernorm_weight({hidden_size});
    std::copy(params_data + offset, params_data + offset + hidden_size,
              post_attention_layernorm_weight.getData());
    offset += hidden_size;

    // FFN gate projection: 8960 x 1536
    const size_t ffn_proj_size = intermediate_size * hidden_size;
    MemRef<float, 2> gate_proj_weight({intermediate_size, hidden_size});
    std::copy(params_data + offset, params_data + offset + ffn_proj_size,
              gate_proj_weight.getData());
    offset += ffn_proj_size;

    // FFN up projection: 8960 x 1536
    MemRef<float, 2> up_proj_weight({intermediate_size, hidden_size});
    std::copy(params_data + offset, params_data + offset + ffn_proj_size,
              up_proj_weight.getData());
    offset += ffn_proj_size;

    // FFN down projection: 1536 x 8960
    const size_t ffn_down_size = hidden_size * intermediate_size;
    MemRef<float, 2> down_proj_weight({hidden_size, intermediate_size});
    std::copy(params_data + offset, params_data + offset + ffn_down_size,
              down_proj_weight.getData());
    offset += ffn_down_size;

    // Input and output containers
    std::cout << "Generating input data...\n";
    MemRef<float, 3> hidden_states1({batch_size, seq_len, hidden_size});
    MemRef<float, 3> hidden_states2({batch_size, seq_len, hidden_size});
    MemRef<float, 3> hidden_states3({batch_size, seq_len, hidden_size});
    MemRef<long long, 2> attention_mask({batch_size, seq_len}, 1); // All ones
    MemRef<float, 3> output({batch_size, seq_len, hidden_size});

    // Fill with random data
    fillRandomData(hidden_states1);
    fillRandomData(hidden_states2);
    fillRandomData(hidden_states3);

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < num_warmup; ++i) {
      _mlir_ciface_subgraph0(
          &output, &hidden_states1, &hidden_states2, &input_layernorm_weight,
          &q_weight, &k_weight, &v_weight, &attention_mask, &o_weight,
          &hidden_states3, &post_attention_layernorm_weight, &gate_proj_weight,
          &up_proj_weight, &down_proj_weight);
    }

    // Performance measurement
    std::cout << "Running performance test...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
      _mlir_ciface_subgraph0(
          &output, &hidden_states1, &hidden_states2, &input_layernorm_weight,
          &q_weight, &k_weight, &v_weight, &attention_mask, &o_weight,
          &hidden_states3, &post_attention_layernorm_weight, &gate_proj_weight,
          &up_proj_weight, &down_proj_weight);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Results
    double avg_time_ms = duration.count() / 1000.0 / num_iterations;

    std::cout << "\nResults:\n";
    std::cout << "========\n";
    std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Average time per iteration: " << avg_time_ms << " ms\n";

    // Verify output (basic sanity check)
    float *output_data = output.getData();

    std::cout << "\nOutput validation:\n";
    std::cout << "First few output values: ";
    for (int i = 0; i < std::min(5, (int)output.getSize()); ++i) {
      std::cout << output_data[i] << " ";
    }
    std::cout << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

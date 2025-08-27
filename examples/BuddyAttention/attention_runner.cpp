// ===- attention_runner.cpp -----------------------------------------------
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
// DeepSeek R1 Attention Layer Performance Runner
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
void _mlir_ciface_subgraph0(MemRef<float, 3> *result,
                            MemRef<float, 2> *q_weight,
                            MemRef<float, 3> *hidden_states1,
                            MemRef<float, 2> *k_weight,
                            MemRef<float, 3> *hidden_states2,
                            MemRef<float, 2> *v_weight,
                            MemRef<float, 3> *hidden_states3,
                            MemRef<long long, 2> *attention_mask,
                            MemRef<float, 2> *o_weight);
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
void fillRandomData(MemRef<float, 3> &memref, float min_val = -1.0f, float max_val = 1.0f) {
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
  const int num_warmup = 5;
  const int num_iterations = 100;
  const size_t total_params = 9437184; // From the build output

  std::cout << "DeepSeek R1 Attention Performance Test\n";
  std::cout << "======================================\n";
  std::cout << "Batch size: " << batch_size << "\n";
  std::cout << "Sequence length: " << seq_len << "\n";
  std::cout << "Hidden size: " << hidden_size << "\n";
  std::cout << "Warmup iterations: " << num_warmup << "\n";
  std::cout << "Test iterations: " << num_iterations << "\n\n";

  try {
    // Initialize containers
    std::cout << "Initializing containers...\n";

    // Parameters container
    MemRef<float, 1> paramsContainer({total_params});
    loadParameters("arg0.data", paramsContainer);
    std::cout << "Loaded " << paramsContainer.getSize() << " parameters\n";

    // Split parameters into weight matrices
    const size_t weight_size = hidden_size * hidden_size;
    float *params_data = paramsContainer.getData();

    MemRef<float, 2> q_weight({hidden_size, hidden_size});
    MemRef<float, 2> k_weight({hidden_size, hidden_size});
    MemRef<float, 2> v_weight({hidden_size, hidden_size});
    MemRef<float, 2> o_weight({hidden_size, hidden_size});

    // Copy parameters to weight matrices
    std::copy(params_data, params_data + weight_size, q_weight.getData());
    std::copy(params_data + weight_size, params_data + 2 * weight_size, k_weight.getData());
    std::copy(params_data + 2 * weight_size, params_data + 3 * weight_size, v_weight.getData());
    std::copy(params_data + 3 * weight_size, params_data + 4 * weight_size, o_weight.getData());

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
      _mlir_ciface_subgraph0(&output, &q_weight, &hidden_states1,
                             &k_weight, &hidden_states2, &v_weight,
                             &hidden_states3, &attention_mask, &o_weight);
    }

    // Performance measurement
    std::cout << "Running performance test...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
      _mlir_ciface_subgraph0(&output, &q_weight, &hidden_states1,
                             &k_weight, &hidden_states2, &v_weight,
                             &hidden_states3, &attention_mask, &o_weight);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Results
    double avg_time_ms = duration.count() / 1000.0 / num_iterations;
    double throughput = (batch_size * seq_len) / avg_time_ms * 1000.0; // tokens/sec

    std::cout << "\nResults:\n";
    std::cout << "========\n";
    std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Average time per iteration: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << throughput << " tokens/sec\n";

    // Verify output (basic sanity check)
    bool has_nan = false;
    bool has_inf = false;
    float *output_data = output.getData();
    for (size_t i = 0; i < output.getSize(); ++i) {
      if (std::isnan(output_data[i])) has_nan = true;
      if (std::isinf(output_data[i])) has_inf = true;
    }

    std::cout << "\nOutput validation:\n";
    std::cout << "Has NaN: " << (has_nan ? "Yes" : "No") << "\n";
    std::cout << "Has Inf: " << (has_inf ? "Yes" : "No") << "\n";
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

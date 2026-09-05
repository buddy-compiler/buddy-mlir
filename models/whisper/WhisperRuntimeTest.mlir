//===- WhisperRuntimeTest.mlir - Whisper runtime test manifest -----------===//
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
//===----------------------------------------------------------------------===//

rhal.module @whisper_runtime_test attributes {
    version = "0.1.0",
    model_name = "whisper_runtime_test",
    vocab_uri = "file:vocab.txt",
    runner_library = "file:whisper_runner.so",
    params_size = "1",
    vocab_size = "3",
    max_token_len = "4",
    mel_bins = "80",
    audio_frames = "3000",
    enc_seq = "1",
    enc_dim = "1",
    sot_token = "0",
    eot_token = "2"} {
  rhal.constant @params {id = 1 : i32, storage = "external",
                         type = tensor<1xf32>, uri = "file:weights.bin"}
  rhal.codeobj @model {id = 1 : i32, kind = "host_shared_lib",
                       backend = "cpu", uri = "file:model.so"}
  rhal.buffer @audio_features {space = "host",
                               type = tensor<1x80x3000xf32>}
  rhal.buffer @decoder_tokens {space = "host", type = tensor<1x4xi64>}
  rhal.buffer @logits {space = "host", type = tensor<1x4x3xf32>}
  rhal.func @forward {
    inputs = ["audio_features", "decoder_tokens"], outputs = ["logits"],
    dispatch = "model", args = ["audio_features", "decoder_tokens", "logits"]}
}

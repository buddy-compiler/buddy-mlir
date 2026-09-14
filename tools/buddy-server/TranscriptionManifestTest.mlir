//===- TranscriptionManifestTest.mlir - Transcription test manifest -----===//
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

rhal.module @transcription_test attributes {
    version = "0.1.0",
    model_name = "fake_transcription",
    vocab_uri = "file:vocab.txt",
    runner_library = "file:runner.so",
    transcription_library = "file:transcription.so"} {
  rhal.constant @params {id = 1 : i32, storage = "external",
                         type = tensor<1xf32>, uri = "file:weights.bin"}
  rhal.codeobj @model {id = 1 : i32, kind = "host_shared_lib",
                       backend = "cpu", uri = "file:model.so"}
  rhal.buffer @input {space = "host", type = tensor<1xf32>}
  rhal.buffer @output {space = "host", type = tensor<1xf32>}
  rhal.func @forward {
    inputs = ["input"], outputs = ["output"], dispatch = "model",
    args = ["input", "output"]}
}

//===- VocoderJsonCodec.cpp - Latent-to-waveform JSON adapter -------------===//
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

#include "JsonCodec.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>

namespace buddy::server {

VocoderRequest parseVocoderRequest(const std::string &body) {
  auto parsed = llvm::json::parse(body);
  if (!parsed)
    throw JsonCodecError("invalid JSON: " + llvm::toString(parsed.takeError()));
  const auto *object = parsed->getAsObject();
  if (!object)
    throw JsonCodecError("JSON root must be an object");
  for (const auto &field : *object)
    if (field.first != "model" && field.first != "latents")
      throw JsonCodecError("unsupported vocoder field: " + field.first.str());
  VocoderRequest result;
  if (auto model = object->getString("model"))
    result.model = model->str();
  else if (object->find("model") != object->end())
    throw JsonCodecError("model must be a string");
  const auto *latents = object->getArray("latents");
  if (!latents || latents->empty() || latents->size() > 1048576)
    throw JsonCodecError(
        "latents must be a nonempty flat array of at most 1048576 floats");
  result.latents.reserve(latents->size());
  for (const auto &value : *latents) {
    const auto number = value.getAsNumber();
    if (!number || !std::isfinite(*number) ||
        std::abs(*number) > std::numeric_limits<float>::max())
      throw JsonCodecError("latents must contain finite float32 numbers");
    result.latents.push_back(static_cast<float>(*number));
  }
  return result;
}

std::string toJson(const VocoderResult &result) {
  llvm::json::Array audio;
  audio.reserve(result.samples.size());
  for (float sample : result.samples) {
    if (!std::isfinite(sample))
      throw std::runtime_error("vocoder returned a non-finite waveform");
    audio.push_back(static_cast<double>(sample));
  }
  llvm::json::Value value =
      llvm::json::Object{{"model", result.model},
                         {"sample_rate", result.sampleRate},
                         {"channels", 1},
                         {"audio", std::move(audio)},
                         {"inference_ms", result.inferenceMs}};
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << value;
  return output;
}

} // namespace buddy::server

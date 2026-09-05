//===- BackendSelection.cpp - buddy-server backend selection ------------===//
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

#include "BackendSelection.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace buddy {
namespace server {
namespace {

std::vector<BackendSelection> candidates(const BackendPluginPaths &paths) {
  std::vector<BackendSelection> values;
  if (!paths.resident.empty())
    values.push_back({BackendKind::Resident, paths.resident});
  if (!paths.embedding.empty())
    values.push_back({BackendKind::Embedding, paths.embedding});
  if (!paths.maskedLM.empty())
    values.push_back({BackendKind::MaskedLM, paths.maskedLM});
  if (!paths.transcription.empty())
    values.push_back({BackendKind::Transcription, paths.transcription});
  return values;
}

} // namespace

BackendPluginPaths
pluginPathsFromManifest(const buddy::runtime::ModelManifest &manifest) {
  return {manifest.servingLibraryPath, manifest.embeddingLibraryPath,
          manifest.maskedLMLibraryPath, manifest.transcriptionLibraryPath};
}

BackendSelection selectBackend(const BackendPluginPaths &explicitPaths,
                               const BackendPluginPaths &manifestPaths) {
  std::vector<BackendSelection> selected = candidates(explicitPaths);
  if (selected.size() > 1)
    throw std::invalid_argument(
        "choose at most one of --serving-so, --embedding-so, "
        "--masked-lm-so, and --transcription-so");
  if (selected.size() == 1)
    return std::move(selected.front());

  selected = candidates(manifestPaths);
  if (selected.size() != 1)
    throw std::invalid_argument(
        "manifest must provide exactly one backend plugin library");
  return std::move(selected.front());
}

const char *backendKindName(BackendKind kind) {
  switch (kind) {
  case BackendKind::Resident:
    return "resident";
  case BackendKind::Embedding:
    return "embedding";
  case BackendKind::MaskedLM:
    return "masked_lm";
  case BackendKind::Transcription:
    return "transcription";
  }
  return "unknown";
}

} // namespace server
} // namespace buddy

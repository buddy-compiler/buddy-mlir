//===- Llama31TTExecution.cpp - Resident Llama 3.1 TT context ------------===//
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

#include "buddy/runtime/models/Llama31TTExecution.h"

#include "buddy/LLM/ChatTemplate.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace buddy {
namespace runtime {
namespace {

namespace fs = std::filesystem;

constexpr int kBeginOfText = 128000;
constexpr int kEndOfText = 128001;
constexpr int kStartHeader = 128006;
constexpr int kEndHeader = 128007;
constexpr int kEom = 128008;
constexpr int kEot = 128009;

bool startsWith(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

bool environmentFlag(const char *name) {
  const char *value = std::getenv(name);
  if (!value)
    return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized == "1" || normalized == "on" || normalized == "true" ||
         normalized == "yes";
}

std::string lookupAttr(const ModelManifest &manifest, const std::string &key) {
  if (auto it = manifest.resolvedModuleAttrs.find(key);
      it != manifest.resolvedModuleAttrs.end())
    return it->second;
  if (auto it = manifest.moduleAttrs.find(key);
      it != manifest.moduleAttrs.end())
    return it->second;
  throw std::runtime_error("llama31_tt: manifest missing required field '" +
                           key + "'");
}

int parsePositiveIntAttr(const ModelManifest &manifest,
                         const std::string &key) {
  const std::string value = lookupAttr(manifest, key);
  try {
    std::size_t consumed = 0;
    const long long number = std::stoll(value, &consumed);
    if (consumed != value.size() || number <= 0 ||
        number > std::numeric_limits<int>::max())
      throw std::invalid_argument("range");
    return static_cast<int>(number);
  } catch (...) {
    throw std::runtime_error("llama31_tt: manifest field '" + key +
                             "' must be a positive integer, got '" + value +
                             "'");
  }
}

bool parseBoolAttr(const ModelManifest &manifest, const std::string &key) {
  std::string value = lookupAttr(manifest, key);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "1" || value == "true" || value == "on" || value == "yes")
    return true;
  if (value == "0" || value == "false" || value == "off" || value == "no")
    return false;
  throw std::runtime_error("llama31_tt: manifest field '" + key +
                           "' must be boolean, got '" + value + "'");
}

void requireFile(const std::string &path, const std::string &field) {
  std::error_code error;
  if (path.empty() || !fs::is_regular_file(path, error))
    throw std::runtime_error("llama31_tt: " + field +
                             " does not resolve to a file: " + path);
}

std::string findTTNNCodeObject(const ModelManifest &manifest,
                               const std::string &phase) {
  for (const auto &code : manifest.codeObjects) {
    const bool ttnn = code.backend == "ttnn" || code.backend == "tt" ||
                      code.backend == "tenstorrent";
    if (ttnn && code.name.find(phase) != std::string::npos) {
      requireFile(code.path, phase + "_ttnn");
      return code.path;
    }
  }
  throw std::runtime_error("llama31_tt: manifest missing TTNN code object '" +
                           phase + "_ttnn'");
}

std::string artifactRoot(const ModelManifest &manifest) {
  bool sawPrefill = false;
  bool sawDecode = false;
  std::array<bool, 10> required{};
  fs::path embeddedRoot;
  for (const auto &constant : manifest.constants) {
    if (!startsWith(constant.name, "artifact_"))
      continue;
    requireFile(constant.path, constant.name);
    sawPrefill |= startsWith(constant.name, "artifact_prefill_");
    sawDecode |= startsWith(constant.name, "artifact_decode_");
    const std::array<std::string_view, 5> names = {
        "slot_roles_json", "shapes_json", "dtypes_json", "summary_json",
        "weights_bin"};
    for (int phase = 0; phase < 2; ++phase) {
      const std::string prefix =
          phase == 0 ? "artifact_prefill_" : "artifact_decode_";
      if (!startsWith(constant.name, prefix))
        continue;
      for (std::size_t index = 0; index < names.size(); ++index)
        required[phase * 5 + index] =
            required[phase * 5 + index] ||
            constant.name == prefix + std::string(names[index]);
    }
    if (embeddedRoot.empty() && startsWith(constant.uri, "payload:"))
      embeddedRoot = fs::path(constant.path).parent_path() / "chat_artifacts";
  }
  if (!sawPrefill || !sawDecode)
    throw std::runtime_error(
        "llama31_tt: manifest must contain prefill and decode artifact_* "
        "constants");
  for (std::size_t index = 0; index < required.size(); ++index)
    if (!required[index])
      throw std::runtime_error(
          "llama31_tt: manifest is missing a required phase artifact "
          "constant");

  if (!embeddedRoot.empty()) {
    for (const auto &constant : manifest.constants) {
      if (!startsWith(constant.name, "artifact_") ||
          !startsWith(constant.uri, "payload:"))
        continue;
      const std::string rest = constant.name.substr(9);
      const std::size_t split = rest.find('_');
      if (split == std::string::npos)
        continue;
      const std::string phase = rest.substr(0, split);
      std::string filename = rest.substr(split + 1);
      if (filename == "slot_roles_json")
        filename = "slot_roles.json";
      else if (filename == "shapes_json")
        filename = "shapes.json";
      else if (filename == "dtypes_json")
        filename = "dtypes.json";
      else if (filename == "summary_json")
        filename = "summary.json";
      else if (filename == "weights_bin")
        filename = "weights.bin";
      else if (filename == "inv_freq_npy")
        filename = "inv_freq.npy";
      fs::create_directories(fs::path(embeddedRoot) / phase);
      std::error_code copyError;
      fs::copy_file(constant.path, fs::path(embeddedRoot) / phase / filename,
                    fs::copy_options::overwrite_existing, copyError);
      if (copyError)
        throw std::runtime_error("llama31_tt: cannot materialize artifact " +
                                 constant.name + ": " + copyError.message());
    }
    return embeddedRoot.string();
  }
  const std::string configured = lookupAttr(manifest, "artifacts_uri");
  std::error_code error;
  if (!fs::is_directory(configured, error))
    throw std::runtime_error(
        "llama31_tt: artifacts_uri does not resolve to a directory: " +
        configured);
  return configured;
}

std::string tokenizerRoot(const ModelManifest &manifest) {
  fs::path embeddedRoot;
  bool sawTokenizer = false;
  for (const auto &constant : manifest.constants) {
    if (!startsWith(constant.name, "tokenizer_"))
      continue;
    requireFile(constant.path, constant.name);
    sawTokenizer = true;
    if (embeddedRoot.empty() && startsWith(constant.uri, "payload:"))
      embeddedRoot = fs::path(constant.path).parent_path();
  }
  if (sawTokenizer && !embeddedRoot.empty()) {
    for (const auto &constant : manifest.constants) {
      if (!startsWith(constant.name, "tokenizer_"))
        continue;
      std::string filename = constant.name.substr(10);
      if (filename == "tokenizer_json")
        filename = "tokenizer.json";
      else if (filename == "tokenizer_model")
        filename = "tokenizer.model";
      else if (filename == "tokenizer_config_json")
        filename = "tokenizer_config.json";
      else if (filename == "special_tokens_map_json")
        filename = "special_tokens_map.json";
      else if (filename == "generation_config_json")
        filename = "generation_config.json";
      fs::create_directories(embeddedRoot / "tokenizer");
      std::error_code copyError;
      fs::copy_file(constant.path, embeddedRoot / "tokenizer" / filename,
                    fs::copy_options::overwrite_existing, copyError);
      if (copyError)
        throw std::runtime_error("llama31_tt: cannot materialize tokenizer " +
                                 constant.name + ": " + copyError.message());
    }
    return (embeddedRoot / "tokenizer").string();
  }

  fs::path configured(lookupAttr(manifest, "tokenizer_uri"));
  std::error_code error;
  if (fs::is_regular_file(configured, error))
    return configured.parent_path().string();
  error.clear();
  if (fs::is_directory(configured, error))
    return configured.string();
  throw std::runtime_error(
      "llama31_tt: tokenizer_uri is neither an embedded tokenizer nor a "
      "local file/directory: " +
      configured.string());
}

std::vector<uint8_t> decodeBase64(std::string_view text) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<int, 256> reverse{};
  reverse.fill(-1);
  for (int index = 0; index < 64; ++index)
    reverse[static_cast<unsigned char>(alphabet[index])] = index;
  std::vector<uint8_t> bytes;
  int value = 0;
  int bits = -8;
  for (unsigned char c : text) {
    if (c == '=')
      break;
    if (std::isspace(c))
      continue;
    if (reverse[c] < 0)
      throw std::runtime_error("llama31_tt: malformed base64 tokenizer token");
    value = (value << 6) | reverse[c];
    bits += 6;
    if (bits >= 0) {
      bytes.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
      bits -= 8;
    }
  }
  return bytes;
}

std::string nextId() {
  static std::atomic<unsigned long long> next{1};
  return "llama31-tt-cmpl-" + std::to_string(next.fetch_add(1));
}

} // namespace

class Llama31TTExecution::Tokenizer {
public:
  explicit Tokenizer(const fs::path &root) { load(root); }

  std::vector<int> encode(std::string_view text, bool addSpecial) const {
    std::vector<int> ids;
    if (addSpecial)
      ids.push_back(kBeginOfText);
    std::size_t position = 0;
    while (position < text.size()) {
      if (auto special = matchSpecial(text, position)) {
        ids.push_back(special->first);
        position += special->second;
        continue;
      }
      std::size_t next = position + 1;
      while (next < text.size() && !matchSpecial(text, next))
        ++next;
      appendBPE(ids, text.substr(position, next - position));
      position = next;
    }
    return ids;
  }

  std::optional<int> specialTokenId(const std::string &text) const {
    auto found = specialToId.find(text);
    return found == specialToId.end() ? std::nullopt
                                      : std::optional<int>(found->second);
  }

private:
  void load(const fs::path &root) {
    fs::path model = root / "original" / "tokenizer.model";
    if (!fs::is_regular_file(model))
      model = root / "tokenizer.model";
    if (fs::is_regular_file(model))
      loadTiktoken(model);
    else if (fs::is_regular_file(root / "tokenizer.json"))
      loadJson(root / "tokenizer.json");
    else
      throw std::runtime_error(
          "llama31_tt: tokenizer.model/tokenizer.json is missing under " +
          root.string());
    if (tokenToId.empty())
      throw std::runtime_error("llama31_tt: tokenizer vocabulary is empty");
    addSpecial(kBeginOfText, "<|begin_of_text|>");
    addSpecial(kEndOfText, "<|end_of_text|>");
    addSpecial(kStartHeader, "<|start_header_id|>");
    addSpecial(kEndHeader, "<|end_header_id|>");
    addSpecial(kEom, "<|eom_id|>");
    addSpecial(kEot, "<|eot_id|>");
  }

  void loadTiktoken(const fs::path &path) {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("llama31_tt: cannot read " + path.string());
    std::string line;
    while (std::getline(input, line)) {
      const std::size_t space = line.find(' ');
      if (space == std::string::npos)
        continue;
      const std::vector<uint8_t> decoded =
          decodeBase64(std::string_view(line).substr(0, space));
      const int id = std::stoi(line.substr(space + 1));
      tokenToId.emplace(
          std::string(reinterpret_cast<const char *>(decoded.data()),
                      decoded.size()),
          id);
    }
  }

  void loadJson(const fs::path &path) {
    auto buffer = llvm::MemoryBuffer::getFile(path.string());
    if (!buffer)
      throw std::runtime_error("llama31_tt: cannot read " + path.string());
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed)
      throw std::runtime_error("llama31_tt: cannot parse tokenizer.json: " +
                               llvm::toString(parsed.takeError()));
    auto *root = parsed->getAsObject();
    if (!root)
      throw std::runtime_error(
          "llama31_tt: tokenizer.json root is not an object");
    auto *model = root ? root->getObject("model") : nullptr;
    auto *vocab = model ? model->getObject("vocab") : nullptr;
    if (!vocab)
      throw std::runtime_error(
          "llama31_tt: tokenizer.json is missing model.vocab");
    for (const auto &entry : *vocab) {
      if (auto id = entry.second.getAsInteger()) {
        std::string token = entry.first.str();
        // SentencePiece tokenizer.json files spell a leading space as U+2581.
        // Keep the byte-level representation used by the request encoder.
        const std::string marker = "\xe2\x96\x81";
        for (std::size_t position = token.find(marker);
             position != std::string::npos;
             position = token.find(marker, position + 1))
          token.replace(position, marker.size(), " ");
        tokenToId[token] = static_cast<int>(*id);
      }
    }
    if (auto *added = root->getArray("added_tokens")) {
      for (const auto &value : *added) {
        auto *object = value.getAsObject();
        std::optional<int64_t> id =
            object ? object->getInteger("id") : std::optional<int64_t>{};
        std::optional<llvm::StringRef> content =
            object ? object->getString("content")
                   : std::optional<llvm::StringRef>{};
        if (id && content)
          addSpecial(static_cast<int>(*id), content->str());
      }
    }
  }

  void addSpecial(int id, const std::string &text) { specialToId[text] = id; }

  std::optional<std::pair<int, std::size_t>>
  matchSpecial(std::string_view text, std::size_t position) const {
    for (const auto &entry : specialToId) {
      if (text.substr(position, entry.first.size()) == entry.first)
        return std::make_pair(entry.second, entry.first.size());
    }
    return std::nullopt;
  }

  void appendBPE(std::vector<int> &ids, std::string_view text) const {
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    for (unsigned char byte : text)
      pieces.emplace_back(1, static_cast<char>(byte));
    while (pieces.size() > 1) {
      int bestRank = std::numeric_limits<int>::max();
      std::size_t best = pieces.size();
      for (std::size_t index = 0; index + 1 < pieces.size(); ++index) {
        auto found = tokenToId.find(pieces[index] + pieces[index + 1]);
        if (found != tokenToId.end() && found->second < bestRank) {
          bestRank = found->second;
          best = index;
        }
      }
      if (best == pieces.size())
        break;
      pieces[best] += pieces[best + 1];
      pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    for (const std::string &piece : pieces) {
      if (auto found = tokenToId.find(piece); found != tokenToId.end()) {
        ids.push_back(found->second);
        continue;
      }
      throw std::runtime_error(
          "llama31_tt: tokenizer has no rank for an input byte");
    }
  }

  std::unordered_map<std::string, int> tokenToId;
  std::unordered_map<std::string, int> specialToId;
};

Llama31TTExecution::Llama31TTExecution() = default;
Llama31TTExecution::~Llama31TTExecution() = default;

void Llama31TTExecution::load(const ResidentModelConfig &config) {
  if (loaded)
    throw std::runtime_error("llama31_tt: execution is already loaded");
  if (config.raxPath.empty())
    throw std::runtime_error(
        "llama31_tt: --model <path.rax> is required for serving");

  ModelManifest manifest = ModelManifest::loadFromRax(config.raxPath);
  if (manifest.modelName.empty() ||
      !startsWith(manifest.modelName, "llama31_tt"))
    throw std::runtime_error(
        "llama31_tt: manifest model_name must identify llama31_tt, got '" +
        manifest.modelName + "'");

  Metadata candidate;
  candidate.modelName = manifest.modelName;
  candidate.maxCacheLen = parsePositiveIntAttr(manifest, "max_cache_len");
  candidate.batchSize = parsePositiveIntAttr(manifest, "batch_size");
  if (candidate.batchSize != 1)
    throw std::runtime_error(
        "llama31_tt: buddy-server supports only canonical batch_size=1; "
        "manifest batch_size=" +
        std::to_string(candidate.batchSize));
  candidate.ignoreEOS = parseBoolAttr(manifest, "ignore_eos");
  candidate.promptFormat = lookupAttr(manifest, "prompt_format");
  if (candidate.promptFormat != "chat" &&
      candidate.promptFormat != "completion")
    throw std::runtime_error("llama31_tt: unsupported prompt_format '" +
                             candidate.promptFormat + "'");
  candidate.prefillKVOutputOrder =
      lookupAttr(manifest, "prefill_kv_output_order");
  if (candidate.prefillKVOutputOrder != "key_value" &&
      candidate.prefillKVOutputOrder != "value_key")
    throw std::runtime_error(
        "llama31_tt: unsupported prefill_kv_output_order '" +
        candidate.prefillKVOutputOrder + "'");
  candidate.prefillPath = findTTNNCodeObject(manifest, "prefill");
  candidate.decodePath = findTTNNCodeObject(manifest, "decode");
  candidate.artifactsPath = artifactRoot(manifest);
  candidate.tokenizerPath = tokenizerRoot(manifest);

  auto candidateTokenizer =
      std::make_unique<Tokenizer>(fs::path(candidate.tokenizerPath));
  metadataValue = std::move(candidate);
  tokenizer = std::move(candidateTokenizer);
  chatTemplatePath = config.chatTemplatePath;
  loaded = true;
  reset();
}

void Llama31TTExecution::reset() {
  if (resetCallback)
    resetCallback();
  cachePositionValue = 0;
}

TokenizeResult Llama31TTExecution::tokenize(const std::string &text,
                                            bool countOnly,
                                            bool addSpecial) const {
  if (!loaded || !tokenizer)
    throw std::runtime_error("llama31_tt: execution is not loaded");
  std::vector<int> ids = tokenizer->encode(text, addSpecial);
  if (ids.size() > static_cast<std::size_t>(metadataValue.maxCacheLen))
    throw std::runtime_error(
        "llama31_tt: tokenized input exceeds max_cache_len=" +
        std::to_string(metadataValue.maxCacheLen));
  TokenizeResult result;
  result.count = ids.size();
  if (!countOnly)
    result.tokens = std::move(ids);
  return result;
}

std::string
Llama31TTExecution::renderChat(const ChatCompletionRequest &request) const {
  if (!loaded)
    throw std::runtime_error("llama31_tt: execution is not loaded");
  std::vector<buddy::Message> messages;
  if (request.messages.empty()) {
    if (request.input.empty())
      throw std::invalid_argument("llama31_tt: chat request has no messages");
    messages.push_back({"user", request.input});
  } else {
    for (const ChatMessage &message : request.messages) {
      if (message.role != "system" && message.role != "user" &&
          message.role != "assistant")
        throw std::invalid_argument("llama31_tt: unsupported chat role '" +
                                    message.role + "'");
      if (!message.images.empty())
        throw std::invalid_argument(
            "llama31_tt: image inputs are not supported");
      messages.push_back({message.role, message.content});
    }
  }

  if (!chatTemplatePath.empty())
    return buddy::ChatTemplate::fromFile(chatTemplatePath).apply(messages);

  std::string prompt = "<|begin_of_text|>";
  for (const buddy::Message &message : messages) {
    prompt += "<|start_header_id|>" + message.role + "<|end_header_id|>\n\n" +
              message.content + "<|eot_id|>";
  }
  prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return prompt;
}

CompletionResult
Llama31TTExecution::generate(const std::string &prompt,
                             const SamplingParams &sampling,
                             const CompletionStreamCallback &callback) {
  if (!loaded || !tokenizer)
    throw std::runtime_error("llama31_tt: execution is not loaded");
  if (prompt.empty())
    throw std::invalid_argument("llama31_tt: completion prompt is empty");
  if (sampling.maxTokens < 0)
    throw std::invalid_argument("llama31_tt: max_tokens must not be negative");

  reset();
  try {
    BackendRequest request;
    const bool promptHasBos = startsWith(prompt, "<|begin_of_text|>");
    request.promptTokens = tokenizer->encode(prompt, !promptHasBos);
    if (request.promptTokens.empty() ||
        request.promptTokens.size() >=
            static_cast<std::size_t>(metadataValue.maxCacheLen))
      throw std::invalid_argument(
          "llama31_tt: prompt must leave room in max_cache_len=" +
          std::to_string(metadataValue.maxCacheLen));
    request.sampling = sampling;
    request.cachePosition = static_cast<int>(request.promptTokens.size());
    request.maxCacheLen = metadataValue.maxCacheLen;
    request.ignoreEOS = metadataValue.ignoreEOS;
    const int availableTokens =
        metadataValue.maxCacheLen - request.cachePosition;
    if (request.sampling.maxTokens == 0 ||
        request.sampling.maxTokens > availableTokens)
      request.sampling.maxTokens = availableTokens;
    cachePositionValue = request.cachePosition;

    const std::string completionId = nextId();
    auto emit = [&](CompletionChunk chunk) {
      if (!callback)
        return true;
      if (chunk.id.empty())
        chunk.id = completionId;
      if (chunk.model.empty())
        chunk.model = metadataValue.modelName;
      return callback(chunk);
    };

    CompletionResult result;
    if (backendValue) {
      result = backendValue(request, emit);
    } else if (environmentFlag("BUDDY_LLAMA31_TT_FAKE_EXECUTION")) {
      const std::string fake = "llama31_tt fake response";
      const int available = metadataValue.maxCacheLen - request.cachePosition;
      const int limit = sampling.maxTokens > 0
                            ? std::min(sampling.maxTokens, available)
                            : available;
      result.finishReason = limit < static_cast<int>(fake.size())
                                ? FinishReason::Length
                                : FinishReason::Stop;
      for (int index = 0;
           index < limit && index < static_cast<int>(fake.size()); ++index) {
        const std::string delta(1, fake[static_cast<std::size_t>(index)]);
        CompletionChunk chunk;
        chunk.delta = delta;
        chunk.tokenId = -1;
        if (!emit(chunk)) {
          result.finishReason = FinishReason::Cancelled;
          break;
        }
        result.content += delta;
        ++result.usage.completionTokens;
        ++cachePositionValue;
      }
    } else {
      throw std::runtime_error(
          "llama31_tt: TTNN resident execution backend is unavailable in "
          "this build; use a hardware-enabled serving backend (tests may set "
          "BUDDY_LLAMA31_TT_FAKE_EXECUTION=1)");
    }

    result.id = result.id.empty() ? completionId : result.id;
    result.model = metadataValue.modelName;
    if (result.usage.promptTokens == 0)
      result.usage.promptTokens = static_cast<int>(request.promptTokens.size());
    result.usage.totalTokens =
        result.usage.promptTokens + result.usage.completionTokens;
    reset();
    return result;
  } catch (...) {
    reset();
    throw;
  }
}

} // namespace runtime
} // namespace buddy

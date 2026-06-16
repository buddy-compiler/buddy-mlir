//===- Llama31TTRunner.cpp - Tenstorrent Llama 3.1 runner -----------------===//
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

#include "buddy/runtime/models/Llama31TTRunner.h"

#include "buddy/LLM/ChatTemplate.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/Sampler.h"

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal {
class Tensor {
public:
  bool is_allocated() const;
};
} // namespace tt::tt_metal

namespace buddy {
namespace runtime {
namespace {

namespace fs = std::filesystem;
using TTDataType = ::tt::target::DataType;

extern "C" int Py_IsInitialized(void);

constexpr int kBeginOfText = 128000;
constexpr int kEndOfText = 128001;
constexpr int kStartHeader = 128006;
constexpr int kEndHeader = 128007;
constexpr int kEom = 128008;
constexpr int kEot = 128009;

static std::string getEnvOr(const char *name, const std::string &fallback) {
  if (const char *value = std::getenv(name))
    if (value[0] != '\0')
      return value;
  return fallback;
}

static constexpr const char *kAnsiReset = "\033[0m";
static constexpr const char *kAnsiBlueBold = "\033[34;1m";
static constexpr const char *kAnsiYellowBold = "\033[33;1m";

static std::string colorLabel(const std::string &label,
                              const char *color = kAnsiYellowBold) {
  return std::string(color) + label + kAnsiReset;
}

static std::string formatFixed(double value, int precision = 4) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << value;
  return os.str();
}

static void printLlamaLog(const std::string &msg, bool suppress = false) {
  if (suppress)
    return;
  std::cout << colorLabel("[Log]", kAnsiBlueBold) << " " << msg << "\n";
}

static void waitDecodeOutputs(std::vector<::tt::runtime::Tensor> &outputs,
                              int tokenOutputIndex) {
  if (std::getenv("BUDDY_LLAMA31_SKIP_DECODE_WAIT") != nullptr)
    return;
  if (std::getenv("BUDDY_LLAMA31_WAIT_TOKEN_ONLY") == nullptr) {
    ::tt::runtime::wait(outputs);
    return;
  }
  if (tokenOutputIndex < 0)
    tokenOutputIndex = static_cast<int>(outputs.size()) + tokenOutputIndex;
  if (tokenOutputIndex < 0 ||
      tokenOutputIndex >= static_cast<int>(outputs.size()))
    throw std::runtime_error(
        "llama31_tt: decode token output index out of range");
  ::tt::runtime::wait(outputs[static_cast<size_t>(tokenOutputIndex)]);
}

struct OfficialTokenReference {
  std::string sourcePath;
  std::vector<int> promptTokens;
  std::vector<int> forcedTokens;
  std::vector<std::array<int, 5>> top5Tokens;
  int maxPredictions = 500;
  std::string traceOut;
};

struct DecodeTiming {
  int count = 0;
  double wallSeconds = 0.0;
  double buildInputsSeconds = 0.0;
  double submitSeconds = 0.0;
  double outputSeconds = 0.0;
  double logitsToHostSeconds = 0.0;
  double kvRelayoutSeconds = 0.0;
  double bookkeepingSeconds = 0.0;
  double firstWallSeconds = 0.0;
  double firstBuildInputsSeconds = 0.0;
  double firstSubmitSeconds = 0.0;
  double firstOutputSeconds = 0.0;
  double firstLogitsToHostSeconds = 0.0;
  double firstKVRelayoutSeconds = 0.0;
  double firstBookkeepingSeconds = 0.0;

  void add(double wall, double buildInputs, double submit, double output,
           double logitsToHost, double kvRelayout, double bookkeeping) {
    if (count == 0) {
      firstWallSeconds = wall;
      firstBuildInputsSeconds = buildInputs;
      firstSubmitSeconds = submit;
      firstOutputSeconds = output;
      firstLogitsToHostSeconds = logitsToHost;
      firstKVRelayoutSeconds = kvRelayout;
      firstBookkeepingSeconds = bookkeeping;
    }
    ++count;
    wallSeconds += wall;
    buildInputsSeconds += buildInputs;
    submitSeconds += submit;
    outputSeconds += output;
    logitsToHostSeconds += logitsToHost;
    kvRelayoutSeconds += kvRelayout;
    bookkeepingSeconds += bookkeeping;
  }

  double steadyWallSeconds() const {
    return count > 1 ? wallSeconds - firstWallSeconds : 0.0;
  }
  double steadySubmitSeconds() const {
    return count > 1 ? submitSeconds - firstSubmitSeconds : 0.0;
  }
  double steadyBuildInputsSeconds() const {
    return count > 1 ? buildInputsSeconds - firstBuildInputsSeconds : 0.0;
  }
  double steadyOutputSeconds() const {
    return count > 1 ? outputSeconds - firstOutputSeconds : 0.0;
  }
  double steadyLogitsToHostSeconds() const {
    return count > 1 ? logitsToHostSeconds - firstLogitsToHostSeconds : 0.0;
  }
  double steadyKVRelayoutSeconds() const {
    return count > 1 ? kvRelayoutSeconds - firstKVRelayoutSeconds : 0.0;
  }
  double steadyBookkeepingSeconds() const {
    return count > 1 ? bookkeepingSeconds - firstBookkeepingSeconds : 0.0;
  }
};

static int getEnvIntOr(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

static std::string readTextFile(const fs::path &path);

static std::vector<int> parseJsonIntArray(const llvm::json::Object &obj,
                                          llvm::StringRef key) {
  const llvm::json::Array *arr = obj.getArray(key);
  if (!arr)
    throw std::runtime_error("llama31_tt: official reference missing array " +
                             key.str());
  std::vector<int> out;
  out.reserve(arr->size());
  for (const llvm::json::Value &value : *arr) {
    std::optional<int64_t> integer = value.getAsInteger();
    if (!integer)
      throw std::runtime_error("llama31_tt: official reference array " +
                               key.str() + " contains a non-integer");
    out.push_back(static_cast<int>(*integer));
  }
  return out;
}

static std::vector<std::array<int, 5>>
parseJsonTop5Array(const llvm::json::Object &obj, llvm::StringRef key) {
  const llvm::json::Array *rows = obj.getArray(key);
  if (!rows)
    throw std::runtime_error("llama31_tt: official reference missing array " +
                             key.str());
  std::vector<std::array<int, 5>> out;
  out.reserve(rows->size());
  for (const llvm::json::Value &rowValue : *rows) {
    const llvm::json::Array *row = rowValue.getAsArray();
    if (!row || row->size() < 5)
      throw std::runtime_error("llama31_tt: official top5 row is malformed");
    std::array<int, 5> values{};
    for (size_t i = 0; i < values.size(); ++i) {
      std::optional<int64_t> integer = (*row)[i].getAsInteger();
      if (!integer)
        throw std::runtime_error(
            "llama31_tt: official top5 row contains a non-integer");
      values[i] = static_cast<int>(*integer);
    }
    out.push_back(values);
  }
  return out;
}

static std::optional<OfficialTokenReference>
loadOfficialTokenReferenceFromEnv() {
  const char *pathEnv = std::getenv("BUDDY_LLAMA31_OFFICIAL_REFERENCE_JSON");
  if (!pathEnv || pathEnv[0] == '\0')
    return std::nullopt;

  fs::path path(pathEnv);
  std::string text = readTextFile(path);
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed)
    throw std::runtime_error("llama31_tt: cannot parse official reference " +
                             path.string() + ": " +
                             llvm::toString(parsed.takeError()));
  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj)
    throw std::runtime_error("llama31_tt: official reference root is not JSON "
                             "object: " +
                             path.string());

  OfficialTokenReference ref;
  ref.sourcePath = path.string();
  ref.promptTokens = parseJsonIntArray(*obj, "prompt_tokens");
  ref.forcedTokens = parseJsonIntArray(*obj, "forced_reference_tokens");
  ref.top5Tokens = parseJsonTop5Array(*obj, "official_top5_tokens");
  ref.maxPredictions =
      getEnvIntOr("BUDDY_LLAMA31_OFFICIAL_MAX_PREDICTIONS", 500);
  if (ref.maxPredictions <= 0)
    ref.maxPredictions = 500;
  if (const char *trace = std::getenv("BUDDY_LLAMA31_OFFICIAL_TRACE_OUT"))
    ref.traceOut = trace;
  if (ref.promptTokens.empty() || ref.forcedTokens.empty() ||
      ref.top5Tokens.empty())
    throw std::runtime_error("llama31_tt: official reference is empty");
  return ref;
}

static std::pair<double, double>
computeOfficialAccuracy(const std::vector<int> &predicted,
                        const std::vector<std::array<int, 5>> &top5Tokens) {
  const size_t n = std::min(predicted.size(), top5Tokens.size());
  if (n == 0)
    return {0.0, 0.0};
  size_t top1 = 0;
  size_t top5 = 0;
  for (size_t i = 0; i < n; ++i) {
    if (predicted[i] == top5Tokens[i][0])
      ++top1;
    if (std::find(top5Tokens[i].begin(), top5Tokens[i].end(), predicted[i]) !=
        top5Tokens[i].end())
      ++top5;
  }
  return {100.0 * static_cast<double>(top1) / static_cast<double>(n),
          100.0 * static_cast<double>(top5) / static_cast<double>(n)};
}

static void writeIntVectorJson(std::ostream &os, const std::vector<int> &xs) {
  os << "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i)
      os << ",";
    os << xs[i];
  }
  os << "]";
}

static void writeJsonString(std::ostream &os, std::string_view text) {
  os << '"';
  for (unsigned char c : text) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (c < 0x20) {
        os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
           << static_cast<int>(c) << std::dec << std::setfill(' ');
      } else {
        os << static_cast<char>(c);
      }
      break;
    }
  }
  os << '"';
}

static void writeStringVectorJson(std::ostream &os,
                                  const std::vector<std::string> &xs) {
  os << "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i)
      os << ",";
    writeJsonString(os, xs[i]);
  }
  os << "]";
}

static void
writeIntVectorVectorJson(std::ostream &os,
                         const std::vector<std::vector<int>> &xs) {
  os << "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i)
      os << ",";
    writeIntVectorJson(os, xs[i]);
  }
  os << "]";
}

static void writeBatchTrace(const std::string &path,
                            const std::string &modelName,
                            const std::vector<std::string> &prompts,
                            const std::vector<std::vector<int>> &promptTokens,
                            int prefillPromptLen, int maxCacheLen,
                            bool leftPadPrompts, double prefillSeconds,
                            const DecodeTiming &timing,
                            double deferredTokenReadbackSeconds,
                            const std::vector<std::vector<int>> &generated,
                            const std::vector<std::string> &generatedText) {
  if (path.empty())
    return;
  fs::path outPath(path);
  std::error_code ec;
  if (!outPath.parent_path().empty())
    fs::create_directories(outPath.parent_path(), ec);
  std::ofstream os(outPath);
  if (!os)
    throw std::runtime_error("llama31_tt: cannot write batch trace " +
                             outPath.string());

  os << "{\n";
  os << "  \"model_name\": ";
  writeJsonString(os, modelName);
  os << ",\n";
  os << "  \"batch_size\": " << prompts.size() << ",\n";
  os << "  \"prefill_prompt_length\": " << prefillPromptLen << ",\n";
  os << "  \"max_cache_len\": " << maxCacheLen << ",\n";
  os << "  \"left_pad_prompts\": "
     << (leftPadPrompts ? "true" : "false") << ",\n";
  os << "  \"prefill_seconds\": " << formatFixed(prefillSeconds, 6)
     << ",\n";
  os << "  \"decode_count\": " << timing.count << ",\n";
  os << "  \"decode_wall_seconds\": " << formatFixed(timing.wallSeconds, 6)
     << ",\n";
  os << "  \"decode_steady_wall_seconds\": "
     << formatFixed(timing.steadyWallSeconds(), 6) << ",\n";
  os << "  \"decode_submit_seconds\": "
     << formatFixed(timing.submitSeconds, 6) << ",\n";
  os << "  \"decode_steady_submit_seconds\": "
     << formatFixed(timing.steadySubmitSeconds(), 6) << ",\n";
  os << "  \"decode_logits_to_host_seconds\": "
     << formatFixed(timing.logitsToHostSeconds, 6) << ",\n";
  os << "  \"decode_steady_logits_to_host_seconds\": "
     << formatFixed(timing.steadyLogitsToHostSeconds(), 6) << ",\n";
  os << "  \"deferred_token_readback_seconds\": "
     << formatFixed(deferredTokenReadbackSeconds, 6) << ",\n";
  os << "  \"decode_wall_plus_deferred_readback_seconds\": "
     << formatFixed(timing.wallSeconds + deferredTokenReadbackSeconds, 6)
     << ",\n";
  os << "  \"decode_steady_wall_plus_deferred_readback_seconds\": "
     << formatFixed(timing.steadyWallSeconds() + deferredTokenReadbackSeconds,
                    6)
     << ",\n";
  os << "  \"prompts\": ";
  writeStringVectorJson(os, prompts);
  os << ",\n";
  os << "  \"prompt_tokens\": ";
  writeIntVectorVectorJson(os, promptTokens);
  os << ",\n";
  os << "  \"generated_tokens\": ";
  writeIntVectorVectorJson(os, generated);
  os << ",\n";
  os << "  \"generated_text\": ";
  writeStringVectorJson(os, generatedText);
  os << "\n";
  os << "}\n";
}

static void writeOfficialTrace(const OfficialTokenReference &ref,
                               const std::vector<int> &predicted,
                               int promptLength, double prefillSeconds,
                               const DecodeTiming &timing,
                               bool decodeExtractsKVOutputs, double top1,
                               double top5) {
  if (ref.traceOut.empty())
    return;
  fs::path outPath(ref.traceOut);
  std::error_code ec;
  if (!outPath.parent_path().empty())
    fs::create_directories(outPath.parent_path(), ec);
  std::ofstream os(outPath);
  if (!os)
    throw std::runtime_error("llama31_tt: cannot write official trace " +
                             outPath.string());
  os << "{\n";
  os << "  \"reference_json\": \"" << ref.sourcePath << "\",\n";
  os << "  \"prompt_length\": " << promptLength << ",\n";
  os << "  \"decode_count\": " << timing.count << ",\n";
  os << "  \"predicted_tokens\": ";
  writeIntVectorJson(os, predicted);
  os << ",\n";
  os << "  \"top1_pct\": " << formatFixed(top1, 6) << ",\n";
  os << "  \"top5_pct\": " << formatFixed(top5, 6) << ",\n";
  os << "  \"prefill_seconds\": " << formatFixed(prefillSeconds, 6) << ",\n";
  os << "  \"decode_extract_kv_outputs\": "
     << (decodeExtractsKVOutputs ? "true" : "false") << ",\n";
  os << "  \"decode_wall_seconds\": " << formatFixed(timing.wallSeconds, 6)
     << ",\n";
  os << "  \"decode_steady_wall_seconds\": "
     << formatFixed(timing.steadyWallSeconds(), 6) << ",\n";
  os << "  \"first_decode_seconds\": "
     << formatFixed(timing.firstWallSeconds, 6) << ",\n";
  os << "  \"decode_build_inputs_seconds\": "
     << formatFixed(timing.buildInputsSeconds, 6) << ",\n";
  os << "  \"decode_steady_build_inputs_seconds\": "
     << formatFixed(timing.steadyBuildInputsSeconds(), 6) << ",\n";
  os << "  \"first_decode_build_inputs_seconds\": "
     << formatFixed(timing.firstBuildInputsSeconds, 6) << ",\n";
  os << "  \"decode_submit_seconds\": "
     << formatFixed(timing.submitSeconds, 6) << ",\n";
  os << "  \"decode_steady_submit_seconds\": "
     << formatFixed(timing.steadySubmitSeconds(), 6) << ",\n";
  os << "  \"first_decode_submit_seconds\": "
     << formatFixed(timing.firstSubmitSeconds, 6) << ",\n";
  os << "  \"decode_output_seconds\": "
     << formatFixed(timing.outputSeconds, 6) << ",\n";
  os << "  \"decode_steady_output_seconds\": "
     << formatFixed(timing.steadyOutputSeconds(), 6) << ",\n";
  os << "  \"first_decode_output_seconds\": "
     << formatFixed(timing.firstOutputSeconds, 6) << ",\n";
  os << "  \"decode_logits_to_host_seconds\": "
     << formatFixed(timing.logitsToHostSeconds, 6) << ",\n";
  os << "  \"decode_steady_logits_to_host_seconds\": "
     << formatFixed(timing.steadyLogitsToHostSeconds(), 6) << ",\n";
  os << "  \"first_decode_logits_to_host_seconds\": "
     << formatFixed(timing.firstLogitsToHostSeconds, 6) << ",\n";
  os << "  \"decode_kv_relayout_seconds\": "
     << formatFixed(timing.kvRelayoutSeconds, 6) << ",\n";
  os << "  \"decode_steady_kv_relayout_seconds\": "
     << formatFixed(timing.steadyKVRelayoutSeconds(), 6) << ",\n";
  os << "  \"first_decode_kv_relayout_seconds\": "
     << formatFixed(timing.firstKVRelayoutSeconds, 6) << ",\n";
  os << "  \"decode_bookkeeping_seconds\": "
     << formatFixed(timing.bookkeepingSeconds, 6) << ",\n";
  os << "  \"decode_steady_bookkeeping_seconds\": "
     << formatFixed(timing.steadyBookkeepingSeconds(), 6) << ",\n";
  os << "  \"first_decode_bookkeeping_seconds\": "
     << formatFixed(timing.firstBookkeepingSeconds, 6) << "\n";
  os << "}\n";
}

static std::string lookupAttr(const ModelManifest &manifest,
                              const std::string &key,
                              const std::string &fallback = "") {
  auto resolved = manifest.resolvedModuleAttrs.find(key);
  if (resolved != manifest.resolvedModuleAttrs.end())
    return resolved->second;
  auto raw = manifest.moduleAttrs.find(key);
  if (raw != manifest.moduleAttrs.end())
    return raw->second;
  return fallback;
}

static bool lookupBoolAttr(const ModelManifest &manifest,
                           const std::string &key, bool fallback) {
  const std::string value = lookupAttr(manifest, key);
  if (value.empty())
    return fallback;
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

static int64_t lookupIntAttr(const ModelManifest &manifest,
                             const std::string &key, int64_t fallback) {
  const std::string value = lookupAttr(manifest, key);
  if (value.empty())
    return fallback;
  try {
    return std::stoll(value);
  } catch (...) {
    return fallback;
  }
}

static void raiseMemoryLimitForLlama() {
#if defined(__unix__) || defined(__APPLE__)
  constexpr rlim_t target = static_cast<rlim_t>(95000000) * 1024;
  auto raiseOne = [](int resource) {
    struct rlimit limit{};
    if (getrlimit(resource, &limit) != 0)
      return;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur >= target)
      return;
    rlim_t requested = target;
    if (limit.rlim_max != RLIM_INFINITY && requested > limit.rlim_max)
      requested = limit.rlim_max;
    if (requested <= limit.rlim_cur)
      return;
    limit.rlim_cur = requested;
    setrlimit(resource, &limit);
  };
  raiseOne(RLIMIT_AS);
  raiseOne(RLIMIT_RSS);
#endif
}

static void keepLibPythonVisibleForTTMLIRRuntime() {
  // The current tt-mlir runtime shared object is built together with tt-metal's
  // Python bindings and has an indirect libpython dependency. Reference one
  // stable Python C-API symbol so linkers using --as-needed keep libpython as a
  // direct dependency of buddy-cli. This does not initialize Python or import
  // any Python modules.
  if (std::getenv("BUDDY_LLAMA31_CHECK_PYTHON_ABI"))
    (void)Py_IsInitialized();
}

static std::string findTTNNArtifact(const ModelManifest &manifest,
                                    const std::string &phase) {
  for (const auto &codeObject : manifest.codeObjects) {
    const bool ttBackend = codeObject.backend == "tt" ||
                           codeObject.backend == "ttnn" ||
                           codeObject.backend == "tenstorrent";
    if (!ttBackend)
      continue;
    if (codeObject.name.find(phase) != std::string::npos)
      return codeObject.path;
  }
  throw std::runtime_error("llama31_tt: missing TTNN code object for phase '" +
                           phase + "'");
}

static bool parseArtifactConstantName(const std::string &name,
                                      std::string &phase,
                                      std::string &filename) {
  static const std::string prefix = "artifact_";
  if (name.rfind(prefix, 0) != 0)
    return false;

  std::string rest = name.substr(prefix.size());
  if (rest.rfind("prefill_", 0) == 0) {
    phase = "prefill";
    rest = rest.substr(std::string("prefill_").size());
  } else if (rest.rfind("decode_", 0) == 0) {
    phase = "decode";
    rest = rest.substr(std::string("decode_").size());
  } else {
    return false;
  }

  if (rest == "weights_bin")
    filename = "weights.bin";
  else if (rest == "slot_roles_json")
    filename = "slot_roles.json";
  else if (rest == "shapes_json")
    filename = "shapes.json";
  else if (rest == "dtypes_json")
    filename = "dtypes.json";
  else if (rest == "summary_json")
    filename = "summary.json";
  else if (rest == "inv_freq_npy")
    filename = "inv_freq.npy";
  else
    return false;
  return true;
}

static bool parseTokenizerConstantName(const std::string &name,
                                       std::string &filename) {
  static const std::string prefix = "tokenizer_";
  if (name.rfind(prefix, 0) != 0)
    return false;

  const std::string rest = name.substr(prefix.size());
  if (rest == "tokenizer_json")
    filename = "tokenizer.json";
  else if (rest == "tokenizer_model")
    filename = "tokenizer.model";
  else if (rest == "tokenizer_config_json")
    filename = "tokenizer_config.json";
  else if (rest == "special_tokens_map_json")
    filename = "special_tokens_map.json";
  else if (rest == "generation_config_json")
    filename = "generation_config.json";
  else
    return false;
  return true;
}

static void linkOrCopyPayloadFile(const fs::path &src, const fs::path &dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  if (ec)
    throw std::runtime_error("llama31_tt: cannot create artifact directory: " +
                             dst.parent_path().string() + ": " + ec.message());

  if (fs::exists(dst, ec)) {
    if (!ec && fs::equivalent(src, dst, ec))
      return;
    ec.clear();
    fs::remove(dst, ec);
    if (ec)
      throw std::runtime_error("llama31_tt: cannot replace artifact file: " +
                               dst.string() + ": " + ec.message());
  }

  fs::create_symlink(src, dst, ec);
  if (!ec)
    return;

  ec.clear();
  fs::create_hard_link(src, dst, ec);
  if (!ec)
    return;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec)
    throw std::runtime_error("llama31_tt: cannot materialize artifact " +
                             src.string() + " -> " + dst.string() + ": " +
                             ec.message());
}

static std::string materializeEmbeddedArtifacts(const ModelManifest &manifest,
                                                const fs::path &raxDir) {
  bool hasArtifactConstants = false;
  fs::path root;

  for (const auto &constant : manifest.constants) {
    std::string phase;
    std::string filename;
    if (!parseArtifactConstantName(constant.name, phase, filename))
      continue;
    if (constant.path.empty())
      continue;
    if (constant.uri.rfind("payload:", 0) != 0)
      continue;

    if (!hasArtifactConstants) {
      const fs::path payloadParent = fs::path(constant.path).parent_path();
      root = payloadParent.empty() ? raxDir / "chat_artifacts"
                                   : payloadParent / "chat_artifacts";
      hasArtifactConstants = true;
    }

    linkOrCopyPayloadFile(fs::path(constant.path), root / phase / filename);
  }

  return hasArtifactConstants ? fs::absolute(root).string() : "";
}

static fs::path resolveEmbeddedTokenizerPath(const ModelManifest &manifest) {
  bool hasTokenizerConstants = false;
  fs::path root;

  for (const auto &constant : manifest.constants) {
    std::string filename;
    if (!parseTokenizerConstantName(constant.name, filename))
      continue;
    if (constant.path.empty())
      continue;
    if (constant.uri.rfind("payload:", 0) != 0)
      continue;

    const fs::path payloadPath = fs::path(constant.path);
    if (!hasTokenizerConstants) {
      const fs::path payloadParent = payloadPath.parent_path();
      root = payloadParent.empty() ? fs::path("tokenizer")
                                   : payloadParent / "tokenizer";
      hasTokenizerConstants = true;
    }

    linkOrCopyPayloadFile(payloadPath, root / filename);
  }

  return hasTokenizerConstants ? fs::absolute(root) : fs::path();
}

static uint16_t readU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t readU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static std::string readTextFile(const fs::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("llama31_tt: cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

class MappedFile {
public:
  MappedFile() = default;
  explicit MappedFile(const fs::path &path) { open(path); }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  MappedFile(MappedFile &&other) noexcept { moveFrom(other); }
  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      close();
      moveFrom(other);
    }
    return *this;
  }
  ~MappedFile() { close(); }

  void open(const fs::path &path) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      throw std::runtime_error("llama31_tt: cannot open " + path.string() +
                               ": " + std::strerror(errno));

    struct stat st{};
    if (::fstat(fd_, &st) != 0)
      throw std::runtime_error("llama31_tt: cannot stat " + path.string());
    if (st.st_size <= 0)
      throw std::runtime_error("llama31_tt: empty file: " + path.string());
    size_ = static_cast<size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("llama31_tt: mmap failed for " + path.string());
    }
  }

  const uint8_t *data() const { return static_cast<const uint8_t *>(data_); }
  size_t size() const { return size_; }
  const fs::path &path() const { return path_; }

private:
  void close() {
    if (data_) {
      ::munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  void moveFrom(MappedFile &other) {
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    path_ = std::move(other.path_);
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  int fd_ = -1;
  void *data_ = nullptr;
  size_t size_ = 0;
  fs::path path_;
};

struct NpyView {
  const uint8_t *data = nullptr;
  size_t bytes = 0;
  std::string descr;
  std::vector<uint32_t> shape;
  bool fortranOrder = false;
};

struct RawTensorView {
  const uint8_t *data = nullptr;
  size_t bytes = 0;
};

static std::string extractPythonDictString(const std::string &header,
                                           const std::string &key) {
  const size_t keyPos = header.find("'" + key + "'");
  if (keyPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing key '" + key +
                             "'");
  const size_t colon = header.find(':', keyPos);
  const size_t quote0 = header.find('\'', colon);
  const size_t quote1 = header.find('\'', quote0 + 1);
  if (colon == std::string::npos || quote0 == std::string::npos ||
      quote1 == std::string::npos)
    throw std::runtime_error("llama31_tt: malformed .npy header key '" + key +
                             "'");
  return header.substr(quote0 + 1, quote1 - quote0 - 1);
}

static std::vector<uint32_t> extractPythonDictShape(const std::string &header) {
  const size_t keyPos = header.find("'shape'");
  if (keyPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing shape");
  const size_t open = header.find('(', keyPos);
  const size_t close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos)
    throw std::runtime_error("llama31_tt: malformed .npy shape");
  std::vector<uint32_t> shape;
  size_t pos = open + 1;
  while (pos < close) {
    while (pos < close && (header[pos] == ' ' || header[pos] == ','))
      ++pos;
    if (pos >= close)
      break;
    size_t end = pos;
    while (end < close && std::isdigit(static_cast<unsigned char>(header[end])))
      ++end;
    if (end == pos)
      break;
    const uint64_t dim = std::stoull(header.substr(pos, end - pos));
    if (dim > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("llama31_tt: .npy dimension too large");
    shape.push_back(static_cast<uint32_t>(dim));
    pos = end;
  }
  return shape;
}

static NpyView parseNpy(const uint8_t *base, size_t size,
                        const std::string &label) {
  static constexpr std::array<uint8_t, 6> magic = {0x93, 'N', 'U',
                                                   'M',  'P', 'Y'};
  if (size < 10 || std::memcmp(base, magic.data(), magic.size()) != 0)
    throw std::runtime_error("llama31_tt: invalid .npy blob: " + label);
  const uint8_t major = base[6];
  uint32_t headerLen = 0;
  size_t headerOffset = 0;
  if (major == 1) {
    headerLen = readU16LE(base + 8);
    headerOffset = 10;
  } else if (major == 2 || major == 3) {
    if (size < 12)
      throw std::runtime_error("llama31_tt: truncated .npy header: " + label);
    headerLen = readU32LE(base + 8);
    headerOffset = 12;
  } else {
    throw std::runtime_error("llama31_tt: unsupported .npy version in " +
                             label);
  }
  if (headerOffset + headerLen > size)
    throw std::runtime_error("llama31_tt: truncated .npy header: " + label);

  const std::string header(reinterpret_cast<const char *>(base + headerOffset),
                           headerLen);
  NpyView view;
  view.descr = extractPythonDictString(header, "descr");
  view.shape = extractPythonDictShape(header);
  const size_t fortranPos = header.find("'fortran_order'");
  if (fortranPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing fortran_order");
  const size_t colon = header.find(':', fortranPos);
  view.fortranOrder = header.find("True", colon) != std::string::npos &&
                      header.find("True", colon) < header.find(',', colon);
  if (view.fortranOrder)
    throw std::runtime_error("llama31_tt: Fortran-order .npy is unsupported: " +
                             label);

  const size_t dataOffset = headerOffset + headerLen;
  view.data = base + dataOffset;
  view.bytes = size - dataOffset;
  return view;
}

class NpyFile {
public:
  NpyFile() = default;
  explicit NpyFile(const fs::path &path) : mapped_(path) {
    view_ = parseNpy(mapped_.data(), mapped_.size(), path.string());
  }
  const NpyView &view() const { return view_; }

private:
  MappedFile mapped_;
  NpyView view_;
};

struct RoleEntry {
  uint32_t slot = 0;
  std::string role;
  std::vector<uint32_t> shape;
  std::string dtype;
  uint64_t weightOffset = 0;
  uint64_t weightBytes = 0;
};

static std::string errorToString(llvm::Error error) {
  return llvm::toString(std::move(error));
}

static std::vector<RoleEntry> parseRoles(const fs::path &path) {
  auto parsed = llvm::json::parse(readTextFile(path));
  if (!parsed)
    throw std::runtime_error("llama31_tt: JSON parse failed for " +
                             path.string() + ": " +
                             errorToString(parsed.takeError()));
  const auto *array = parsed->getAsArray();
  if (!array)
    throw std::runtime_error("llama31_tt: slot_roles.json must be an array");

  std::vector<RoleEntry> roles;
  roles.reserve(array->size());
  for (const auto &value : *array) {
    const auto *object = value.getAsObject();
    if (!object)
      throw std::runtime_error("llama31_tt: malformed slot role entry");
    RoleEntry entry;
    auto slot = object->getInteger("slot");
    auto role = object->getString("role");
    auto dtype = object->getString("dtype");
    const auto *shape = object->getArray("shape");
    if (!slot || !role || !dtype || !shape)
      throw std::runtime_error("llama31_tt: incomplete slot role entry");
    entry.slot = static_cast<uint32_t>(*slot);
    entry.role = role->str();
    entry.dtype = dtype->str();
    if (entry.role == "weight") {
      auto offset = object->getInteger("weight_offset");
      auto nbytes = object->getInteger("weight_nbytes");
      if (!offset || !nbytes || *offset < 0 || *nbytes <= 0)
        throw std::runtime_error(
            "llama31_tt: weight slot role missing byte range");
      entry.weightOffset = static_cast<uint64_t>(*offset);
      entry.weightBytes = static_cast<uint64_t>(*nbytes);
    }
    for (const auto &dimValue : *shape) {
      auto dim = dimValue.getAsInteger();
      if (!dim || *dim < 0 ||
          *dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("llama31_tt: invalid tensor shape in " +
                                 path.string());
      entry.shape.push_back(static_cast<uint32_t>(*dim));
    }
    roles.push_back(std::move(entry));
  }
  std::sort(
      roles.begin(), roles.end(),
      [](const RoleEntry &a, const RoleEntry &b) { return a.slot < b.slot; });
  for (size_t i = 0; i < roles.size(); ++i)
    if (roles[i].slot != i)
      throw std::runtime_error("llama31_tt: non-contiguous slot_roles.json");
  return roles;
}

struct PhaseCtx {
  std::string phase;
  fs::path dir;
  std::vector<RoleEntry> roles;
  fs::path weightsPath;
  fs::path invFreqPath;
  std::unique_ptr<MappedFile> weights;
  std::unique_ptr<NpyFile> invFreq;

  PhaseCtx(const fs::path &artifactsDir, std::string phaseName)
      : phase(std::move(phaseName)), dir(artifactsDir / phase),
        roles(parseRoles(dir / "slot_roles.json")) {
    weightsPath = dir / "weights.bin";
    if (fs::exists(weightsPath))
      weights = std::make_unique<MappedFile>(weightsPath);
    invFreqPath = dir / "inv_freq.npy";
    if (fs::exists(invFreqPath))
      invFreq = std::make_unique<NpyFile>(invFreqPath);

    for (const RoleEntry &role : roles)
      if (role.role == "weight") {
        if (!weights)
          throw std::runtime_error("llama31_tt: missing weights.bin for " +
                                   phase);
        if (role.weightOffset > weights->size() ||
            role.weightBytes > weights->size() - role.weightOffset) {
          throw std::runtime_error("llama31_tt: weight byte range out of "
                                   "weights.bin for " +
                                   phase);
        }
      }
  }

  RawTensorView weightForSlot(uint32_t slot) const {
    if (slot >= roles.size() || roles[slot].role != "weight")
      throw std::runtime_error("llama31_tt: slot is not a weight: " +
                               std::to_string(slot));
    if (!weights)
      throw std::runtime_error("llama31_tt: missing weights.bin for " + phase);
    const RoleEntry &role = roles[slot];
    return RawTensorView{weights->data() + role.weightOffset,
                         static_cast<size_t>(role.weightBytes)};
  }
};

static uint64_t volume(const std::vector<uint32_t> &shape) {
  uint64_t result = 1;
  for (uint32_t dim : shape)
    result *= dim;
  return result;
}

static std::vector<uint32_t>
contiguousStride(const std::vector<uint32_t> &shape) {
  std::vector<uint32_t> stride(shape.size(), 1);
  uint64_t running = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    stride[i - 1] = static_cast<uint32_t>(running);
    running *= shape[i - 1];
  }
  return stride;
}

static uint32_t itemSizeFor(TTDataType dtype) {
  switch (dtype) {
  case TTDataType::Float32:
  case TTDataType::Int32:
  case TTDataType::UInt32:
    return 4;
  case TTDataType::Float16:
  case TTDataType::BFloat16:
  case TTDataType::UInt16:
  case TTDataType::Int16:
    return 2;
  case TTDataType::UInt8:
  case TTDataType::Int8:
  case TTDataType::Bool:
    return 1;
  case TTDataType::Float64:
  case TTDataType::Int64:
  case TTDataType::UInt64:
    return 8;
  default:
    throw std::runtime_error("llama31_tt: unsupported scalar tensor dtype");
  }
}

static std::vector<uint8_t> integerBuffer(const std::vector<int64_t> &values,
                                          TTDataType dtype) {
  std::vector<uint8_t> buffer(values.size() * itemSizeFor(dtype));
  auto write = [&](auto tag) {
    using T = decltype(tag);
    T *out = reinterpret_cast<T *>(buffer.data());
    for (size_t i = 0; i < values.size(); ++i)
      out[i] = static_cast<T>(values[i]);
  };
  switch (dtype) {
  case TTDataType::Int64:
    write(int64_t{});
    break;
  case TTDataType::UInt64:
    write(uint64_t{});
    break;
  case TTDataType::Int32:
    write(int32_t{});
    break;
  case TTDataType::UInt32:
    write(uint32_t{});
    break;
  case TTDataType::Int16:
    write(int16_t{});
    break;
  case TTDataType::UInt16:
    write(uint16_t{});
    break;
  case TTDataType::Int8:
    write(int8_t{});
    break;
  case TTDataType::UInt8:
  case TTDataType::Bool:
    write(uint8_t{});
    break;
  default:
    throw std::runtime_error("llama31_tt: expected integer input dtype");
  }
  return buffer;
}

static std::vector<uint8_t> zeroBuffer(const ::tt::runtime::TensorDesc &desc) {
  return std::vector<uint8_t>(volume(desc.shape) * desc.elementSize(), 0);
}

static uint16_t floatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static std::vector<uint8_t>
floatBuffer(const std::vector<float> &values, TTDataType dtype) {
  std::vector<uint8_t> buffer(values.size() * itemSizeFor(dtype));
  switch (dtype) {
  case TTDataType::Float32: {
    auto *out = reinterpret_cast<float *>(buffer.data());
    std::copy(values.begin(), values.end(), out);
    break;
  }
  case TTDataType::BFloat16: {
    auto *out = reinterpret_cast<uint16_t *>(buffer.data());
    for (size_t i = 0; i < values.size(); ++i)
      out[i] = floatToBf16Bits(values[i]);
    break;
  }
  default:
    throw std::runtime_error("llama31_tt: expected floating-point input dtype");
  }
  return buffer;
}

static ::tt::runtime::Tensor
hostTensorFromBytes(const void *data, const ::tt::runtime::TensorDesc &desc) {
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createOwnedHostTensor(
      data, desc.shape, stride, desc.elementSize(), desc.dataType);
}

static ::tt::runtime::Tensor
borrowedHostTensorFromRaw(const RawTensorView &view,
                          const ::tt::runtime::TensorDesc &desc) {
  const uint64_t expectedBytes = volume(desc.shape) * desc.elementSize();
  if (view.bytes != expectedBytes)
    throw std::runtime_error("llama31_tt: raw weight tensor byte size does not "
                             "match TTNN input descriptor");
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createBorrowedHostTensor(
      const_cast<uint8_t *>(view.data), desc.shape, stride, desc.elementSize(),
      desc.dataType);
}

static ::tt::runtime::Tensor
borrowedHostTensorFromNpy(const NpyView &view,
                          const ::tt::runtime::TensorDesc &desc) {
  const uint64_t expectedBytes = volume(desc.shape) * desc.elementSize();
  if (view.bytes < expectedBytes)
    throw std::runtime_error("llama31_tt: .npy tensor is smaller than TTNN "
                             "input descriptor");
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createBorrowedHostTensor(
      const_cast<uint8_t *>(view.data), desc.shape, stride, desc.elementSize(),
      desc.dataType);
}

static ::tt::runtime::Tensor toDevice(::tt::runtime::Tensor host,
                                      ::tt::runtime::Device device,
                                      ::tt::runtime::Binary binary,
                                      uint32_t programIndex, uint32_t slot,
                                      bool retain = false) {
  auto layout = ::tt::runtime::getLayout(binary, programIndex, slot);
  auto deviceTensor = ::tt::runtime::toLayout(host, device, layout, retain);
  if (retain)
    ::tt::runtime::setTensorRetain(deviceTensor, true);
  return deviceTensor;
}

static bool sameFile(const fs::path &lhs, const fs::path &rhs) {
  if (lhs.empty() || rhs.empty())
    return false;
  std::error_code ec;
  const bool equivalent = fs::equivalent(lhs, rhs, ec);
  return !ec && equivalent;
}

static bool startsWith(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

static bool isRemoteModelId(const std::string &value) {
  return !value.empty() && value.find('/') != std::string::npos &&
         !startsWith(value, "/") && !startsWith(value, "./") &&
         !startsWith(value, "../") && !startsWith(value, "file:");
}

static std::vector<uint8_t> decodeBase64(const std::string &text) {
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<int, 256> reverse;
  reverse.fill(-1);
  for (int i = 0; i < 64; ++i)
    reverse[static_cast<unsigned char>(table[i])] = i;

  std::vector<uint8_t> out;
  int value = 0;
  int bits = -8;
  for (unsigned char c : text) {
    if (c == '=')
      break;
    if (std::isspace(c))
      continue;
    const int r = reverse[c];
    if (r < 0)
      throw std::runtime_error("llama31_tt: invalid base64 token in tokenizer");
    value = (value << 6) | r;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
      bits -= 8;
    }
  }
  return out;
}

class Llama31Tokenizer {
public:
  explicit Llama31Tokenizer(const fs::path &modelDir) { load(modelDir); }

  std::vector<int> encodeCompletion(const std::string &prompt) const {
    std::vector<int> ids;
    ids.push_back(kBeginOfText);
    appendText(ids, prompt);
    return ids;
  }

  std::vector<int> encodeChat(const std::string &userMessage,
                              const std::string &systemPrompt = "") const {
    std::vector<int> ids;
    ids.push_back(kBeginOfText);
    if (!systemPrompt.empty()) {
      appendHeader(ids, "system");
      appendText(ids, systemPrompt);
      ids.push_back(kEot);
    }
    appendHeader(ids, "user");
    appendText(ids, userMessage);
    ids.push_back(kEot);
    appendHeader(ids, "assistant");
    appendText(ids, "");
    return ids;
  }

  std::vector<int> encodeChatTemplate(const buddy::ChatTemplate &chatTemplate,
                                      const std::string &userMessage) const {
    const std::vector<buddy::Message> messages = {{"user", userMessage}};
    return encodeTextWithSpecialTokens(chatTemplate.apply(messages));
  }

  std::optional<int> specialTokenId(std::string_view text) const {
    auto it = specialTextToId_.find(std::string(text));
    if (it == specialTextToId_.end())
      return std::nullopt;
    return it->second;
  }

  std::string decodeToken(int id) const {
    auto special = specialIdToText_.find(id);
    if (special != specialIdToText_.end())
      return special->second;
    auto it = idToBytes_.find(id);
    if (it == idToBytes_.end())
      return "";
    return it->second;
  }

  std::string decodeTokens(const std::vector<int> &ids,
                           bool skipSpecial = false) const {
    std::string out;
    for (int id : ids) {
      if (skipSpecial && specialIdToText_.count(id))
        continue;
      out += decodeToken(id);
    }
    return out;
  }

private:
  void load(const fs::path &modelDir) {
    fs::path tokenizerModel = modelDir / "original" / "tokenizer.model";
    if (!fs::exists(tokenizerModel))
      tokenizerModel = modelDir / "tokenizer.model";
    if (fs::exists(tokenizerModel)) {
      loadTiktokenModel(tokenizerModel);
    } else if (fs::exists(modelDir / "tokenizer.json")) {
      loadTokenizerJson(modelDir / "tokenizer.json");
    } else {
      throw std::runtime_error(
          "llama_tt: tokenizer.model/tokenizer.json not found in the .rax "
          "payload or local model directory. Repackage with embedded tokenizer "
          "files, or set LLAMA_MODEL_PATH, LLAMA31_MODEL_PATH, or "
          "LLAMA32_MODEL_PATH to a local Llama model directory.");
    }

    if (tokenToId_.empty())
      throw std::runtime_error("llama31_tt: empty tokenizer ranks");
    addDefaultSpecialTokens();
  }

  void loadTiktokenModel(const fs::path &tokenizerModel) {
    std::ifstream input(tokenizerModel);
    if (!input)
      throw std::runtime_error("llama31_tt: cannot read " +
                               tokenizerModel.string());

    std::string line;
    while (std::getline(input, line)) {
      if (line.empty())
        continue;
      const size_t space = line.find(' ');
      if (space == std::string::npos)
        continue;
      std::vector<uint8_t> bytes = decodeBase64(line.substr(0, space));
      const int rank = std::stoi(line.substr(space + 1));
      std::string token(reinterpret_cast<const char *>(bytes.data()),
                        bytes.size());
      tokenToId_[token] = rank;
      idToBytes_[rank] = std::move(token);
    }
  }

  static std::unordered_map<uint32_t, uint8_t> buildByteDecoder() {
    std::vector<uint32_t> bs;
    for (uint32_t c = '!'; c <= '~'; ++c)
      bs.push_back(c);
    for (uint32_t c = 0x00a1; c <= 0x00ac; ++c)
      bs.push_back(c);
    for (uint32_t c = 0x00ae; c <= 0x00ff; ++c)
      bs.push_back(c);

    std::unordered_map<uint32_t, uint8_t> out;
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
      uint32_t cp = b;
      if (std::find(bs.begin(), bs.end(), b) == bs.end())
        cp = 256 + extra++;
      out[cp] = static_cast<uint8_t>(b);
    }
    return out;
  }

  static bool nextUtf8Codepoint(std::string_view text, size_t &pos,
                                uint32_t &cp) {
    if (pos >= text.size())
      return false;
    const uint8_t c0 = static_cast<uint8_t>(text[pos]);
    if (c0 < 0x80) {
      cp = c0;
      ++pos;
      return true;
    }

    auto continuation = [&](size_t i) -> uint8_t {
      if (i >= text.size())
        throw std::runtime_error("llama_tt: truncated UTF-8 in tokenizer");
      const uint8_t c = static_cast<uint8_t>(text[i]);
      if ((c & 0xc0) != 0x80)
        throw std::runtime_error("llama_tt: malformed UTF-8 in tokenizer");
      return c & 0x3f;
    };

    if ((c0 & 0xe0) == 0xc0) {
      cp = ((c0 & 0x1f) << 6) | continuation(pos + 1);
      pos += 2;
      return true;
    }
    if ((c0 & 0xf0) == 0xe0) {
      cp = ((c0 & 0x0f) << 12) | (continuation(pos + 1) << 6) |
           continuation(pos + 2);
      pos += 3;
      return true;
    }
    if ((c0 & 0xf8) == 0xf0) {
      cp = ((c0 & 0x07) << 18) | (continuation(pos + 1) << 12) |
           (continuation(pos + 2) << 6) | continuation(pos + 3);
      pos += 4;
      return true;
    }
    throw std::runtime_error("llama_tt: malformed UTF-8 in tokenizer");
  }

  static std::string decodeTokenizerJsonToken(std::string_view token) {
    static const auto byteDecoder = buildByteDecoder();
    std::string out;
    size_t pos = 0;
    while (pos < token.size()) {
      const size_t byteStart = pos;
      uint32_t cp = 0;
      nextUtf8Codepoint(token, pos, cp);
      auto it = byteDecoder.find(cp);
      if (it != byteDecoder.end()) {
        out.push_back(static_cast<char>(it->second));
      } else {
        out.append(token.substr(byteStart, pos - byteStart));
      }
    }
    return out;
  }

  void loadTokenizerJson(const fs::path &tokenizerJson) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(tokenizerJson.string());
    if (!bufOrErr)
      throw std::runtime_error("llama31_tt: cannot read " +
                               tokenizerJson.string());
    auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
    if (!parsed)
      throw std::runtime_error("llama31_tt: cannot parse tokenizer.json: " +
                               llvm::toString(parsed.takeError()));

    auto *root = parsed->getAsObject();
    auto *model = root ? root->getObject("model") : nullptr;
    auto *vocab = model ? model->getObject("vocab") : nullptr;
    if (!vocab)
      throw std::runtime_error("llama31_tt: tokenizer.json missing model.vocab");
    for (const auto &kv : *vocab) {
      auto id = kv.second.getAsInteger();
      if (!id)
        continue;
      std::string token = decodeTokenizerJsonToken(kv.first.str());
      tokenToId_[token] = static_cast<int>(*id);
      idToBytes_[static_cast<int>(*id)] = std::move(token);
    }

    if (auto *added = root->getArray("added_tokens")) {
      for (const auto &entry : *added) {
        auto *obj = entry.getAsObject();
        if (!obj)
          continue;
        auto id = obj->getInteger("id");
        auto content = obj->getString("content");
        if (!id || !content)
          continue;
        specialIdToText_[static_cast<int>(*id)] = content->str();
        specialTextToId_[content->str()] = static_cast<int>(*id);
      }
    }
  }

  void addDefaultSpecialTokens() {
    specialIdToText_[kBeginOfText] = "<|begin_of_text|>";
    specialIdToText_[kEndOfText] = "<|end_of_text|>";
    specialIdToText_[kStartHeader] = "<|start_header_id|>";
    specialIdToText_[kEndHeader] = "<|end_header_id|>";
    specialIdToText_[kEom] = "<|eom_id|>";
    specialIdToText_[kEot] = "<|eot_id|>";

    for (const auto &[id, text] : specialIdToText_)
      specialTextToId_[text] = id;
  }

  void appendHeader(std::vector<int> &ids, const std::string &role) const {
    ids.push_back(kStartHeader);
    appendText(ids, role);
    ids.push_back(kEndHeader);
    appendText(ids, "\n\n");
  }

  static bool isAlpha(unsigned char c) { return std::isalpha(c) != 0; }
  static bool isDigit(unsigned char c) { return std::isdigit(c) != 0; }
  static bool isSpace(unsigned char c) { return std::isspace(c) != 0; }
  static bool isNewline(unsigned char c) { return c == '\n' || c == '\r'; }
  static bool isWord(unsigned char c) { return isAlpha(c) || isDigit(c); }

  static bool matchContraction(std::string_view text, size_t pos, size_t &len) {
    static constexpr std::array<std::string_view, 7> suffixes = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (std::string_view suffix : suffixes) {
      if (pos + suffix.size() > text.size())
        continue;
      bool ok = true;
      for (size_t i = 0; i < suffix.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(text[pos + i]);
        unsigned char b = static_cast<unsigned char>(suffix[i]);
        ok &= std::tolower(a) == std::tolower(b);
      }
      if (ok) {
        len = suffix.size();
        return true;
      }
    }
    return false;
  }

  static std::vector<std::string> splitTiktokenApprox(std::string_view text) {
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
      size_t contractionLen = 0;
      if (matchContraction(text, i, contractionLen)) {
        pieces.emplace_back(text.substr(i, contractionLen));
        i += contractionLen;
        continue;
      }

      unsigned char c = static_cast<unsigned char>(text[i]);
      if (c == ' ' && i + 1 < text.size() &&
          isAlpha(static_cast<unsigned char>(text[i + 1]))) {
        size_t j = i + 2;
        while (j < text.size() && isAlpha(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (isAlpha(c)) {
        size_t j = i + 1;
        while (j < text.size() && isAlpha(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (isDigit(c)) {
        size_t j = i + 1;
        while (j < text.size() && j - i < 3 &&
               isDigit(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (c == ' ' && i + 1 < text.size()) {
        unsigned char n = static_cast<unsigned char>(text[i + 1]);
        if (!isSpace(n) && !isWord(n)) {
          size_t j = i + 2;
          while (j < text.size()) {
            unsigned char p = static_cast<unsigned char>(text[j]);
            if (isSpace(p) || isWord(p))
              break;
            ++j;
          }
          while (j < text.size() &&
                 isNewline(static_cast<unsigned char>(text[j])))
            ++j;
          pieces.emplace_back(text.substr(i, j - i));
          i = j;
          continue;
        }
      }
      if (!isSpace(c) && !isWord(c)) {
        size_t j = i + 1;
        while (j < text.size()) {
          unsigned char p = static_cast<unsigned char>(text[j]);
          if (isSpace(p) || isWord(p))
            break;
          ++j;
        }
        while (j < text.size() &&
               isNewline(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }

      size_t j = i + 1;
      while (j < text.size() && isSpace(static_cast<unsigned char>(text[j])) &&
             !isNewline(static_cast<unsigned char>(text[j])))
        ++j;
      if (j < text.size() && isNewline(static_cast<unsigned char>(text[j]))) {
        while (j < text.size() &&
               isNewline(static_cast<unsigned char>(text[j])))
          ++j;
      } else if (j < text.size()) {
        j = i + 1;
      }
      pieces.emplace_back(text.substr(i, j - i));
      i = j;
    }
    return pieces;
  }

  void appendText(std::vector<int> &ids, std::string_view text) const {
    for (const std::string &piece : splitTiktokenApprox(text)) {
      std::vector<std::string> parts;
      parts.reserve(piece.size());
      for (unsigned char c : piece)
        parts.emplace_back(1, static_cast<char>(c));

      while (parts.size() > 1) {
        int bestRank = std::numeric_limits<int>::max();
        size_t bestIndex = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
          std::string merged = parts[i] + parts[i + 1];
          auto it = tokenToId_.find(merged);
          if (it != tokenToId_.end() && it->second < bestRank) {
            bestRank = it->second;
            bestIndex = i;
          }
        }
        if (bestIndex == std::numeric_limits<size_t>::max())
          break;
        std::vector<std::string> next;
        next.reserve(parts.size() - 1);
        for (size_t i = 0; i < parts.size(); ++i) {
          if (i == bestIndex) {
            next.push_back(parts[i] + parts[i + 1]);
            ++i;
          } else {
            next.push_back(std::move(parts[i]));
          }
        }
        parts = std::move(next);
      }

      for (const std::string &part : parts) {
        auto it = tokenToId_.find(part);
        if (it != tokenToId_.end()) {
          ids.push_back(it->second);
          continue;
        }
        for (unsigned char c : part) {
          std::string byte(1, static_cast<char>(c));
          auto byteIt = tokenToId_.find(byte);
          if (byteIt == tokenToId_.end())
            throw std::runtime_error("llama31_tt: tokenizer byte missing");
          ids.push_back(byteIt->second);
        }
      }
    }
  }

  std::optional<std::pair<int, size_t>> matchSpecialToken(std::string_view text,
                                                          size_t pos) const {
    for (const auto &[tokenText, id] : specialTextToId_) {
      if (pos + tokenText.size() > text.size())
        continue;
      if (text.substr(pos, tokenText.size()) == tokenText)
        return std::make_pair(id, tokenText.size());
    }
    return std::nullopt;
  }

  std::vector<int> encodeTextWithSpecialTokens(std::string_view text) const {
    std::vector<int> ids;
    size_t pos = 0;
    while (pos < text.size()) {
      if (auto match = matchSpecialToken(text, pos)) {
        ids.push_back(match->first);
        pos += match->second;
        continue;
      }

      size_t next = pos + 1;
      while (next < text.size() && !matchSpecialToken(text, next))
        ++next;
      appendText(ids, text.substr(pos, next - pos));
      pos = next;
    }
    return ids;
  }

  std::unordered_map<std::string, int> tokenToId_;
  std::unordered_map<int, std::string> idToBytes_;
  std::unordered_map<int, std::string> specialIdToText_;
  std::unordered_map<std::string, int> specialTextToId_;
};

static float halfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x03ff;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static float bf16BitsToFloat(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static std::vector<float> loadLogits(const std::vector<std::byte> &bytes,
                                     TTDataType dtype, size_t start,
                                     size_t count) {
  if (count == 0)
    throw std::runtime_error("llama31_tt: empty logits tensor");

  std::vector<float> logits(count);
  switch (dtype) {
  case TTDataType::Float32: {
    const float *data = reinterpret_cast<const float *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = data[start + i];
    break;
  }
  case TTDataType::BFloat16: {
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = bf16BitsToFloat(data[start + i]);
    break;
  }
  case TTDataType::Float16: {
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = halfBitsToFloat(data[start + i]);
    break;
  }
  default:
    throw std::runtime_error("llama31_tt: unsupported logits dtype");
  }
  return logits;
}

static bool isTokenIdDataType(TTDataType dtype) {
  return dtype == TTDataType::Int32 || dtype == TTDataType::Int64 ||
         dtype == TTDataType::UInt32;
}

static int sampleTensor(const std::vector<std::byte> &bytes, TTDataType dtype,
                        size_t start, size_t count, buddy::Sampler &sampler,
                        const std::vector<int> &recentTokens) {
  const size_t elementSize = itemSizeFor(dtype);
  const size_t elements = bytes.size() / elementSize;
  if (start >= elements || count > elements - start)
    throw std::runtime_error(
        "llama31_tt: output tensor sample range is out of bounds (start=" +
        std::to_string(start) + ", count=" + std::to_string(count) +
        ", elements=" + std::to_string(elements) + ")");
  switch (dtype) {
  case TTDataType::Int32: {
    const int32_t *data = reinterpret_cast<const int32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::Int64: {
    const int64_t *data = reinterpret_cast<const int64_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::UInt32: {
    const uint32_t *data = reinterpret_cast<const uint32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  default:
    std::vector<float> logits = loadLogits(bytes, dtype, start, count);
    return sampler.sample(logits.data(), logits.size(), recentTokens);
  }
}

static int readTokenIdAt(const std::vector<std::byte> &bytes, TTDataType dtype,
                         size_t start) {
  const size_t elementSize = itemSizeFor(dtype);
  const size_t elements = bytes.size() / elementSize;
  if (start >= elements)
    throw std::runtime_error(
        "llama31_tt: token tensor read range is out of bounds");
  switch (dtype) {
  case TTDataType::Int32: {
    const int32_t *data = reinterpret_cast<const int32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::Int64: {
    const int64_t *data = reinterpret_cast<const int64_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::UInt32: {
    const uint32_t *data = reinterpret_cast<const uint32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  default:
    throw std::runtime_error("llama31_tt: expected token-id tensor dtype");
  }
}

static std::vector<int>
readTokenTensorToHost(::tt::runtime::Tensor &tensor, uint32_t batchSize) {
  auto hostShards = ::tt::runtime::toHost(tensor, /*untilize=*/false,
                                          /*blocking=*/true);
  if (hostShards.empty())
    throw std::runtime_error("llama31_tt: token toHost returned no shards");
  auto host = hostShards.front();
  const TTDataType dtype = ::tt::runtime::getTensorDataType(host);
  if (!isTokenIdDataType(dtype))
    throw std::runtime_error("llama31_tt: deferred readback expected token ids");
  const auto shape = ::tt::runtime::getTensorShape(host);
  const auto data = ::tt::runtime::getTensorDataBuffer(host);
  const size_t elements = data.size() / itemSizeFor(dtype);
  if (elements == 0)
    throw std::runtime_error("llama31_tt: empty deferred token tensor");

  size_t seq = 1;
  if (shape.size() >= 2)
    seq = shape.back();
  else if (batchSize > 1 && elements >= batchSize)
    seq = 1;
  else
    seq = elements;

  std::vector<int> tokens;
  tokens.reserve(batchSize);
  for (uint32_t b = 0; b < batchSize; ++b) {
    const size_t start = shape.size() >= 2
                             ? std::min<size_t>(
                                   static_cast<size_t>(b) * seq +
                                       (seq == 0 ? 0 : seq - 1),
                                   elements - 1)
                             : std::min<size_t>(b, elements - 1);
    tokens.push_back(readTokenIdAt(data, dtype, start));
  }
  return tokens;
}

struct ExtractedOutput {
  int token = 0;
  std::vector<int> tokens;
  std::optional<::tt::runtime::Tensor> tokenDevice;
  std::unordered_map<std::string, ::tt::runtime::Tensor> kvDevice;
  double logitsToHostSeconds = 0.0;
  double kvRelayoutSeconds = 0.0;
};

struct StaticInputCacheResult {
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> tensors;
  uint32_t uploaded = 0;
  uint32_t reused = 0;
};

struct DecodeRuntimeInputCache {
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> attentionMasks;
  std::unordered_map<int, std::unordered_map<uint32_t, ::tt::runtime::Tensor>>
      cachePositions;
  std::unordered_map<int, std::unordered_map<uint32_t, ::tt::runtime::Tensor>>
      sdpaMasks;
  std::unordered_map<int, std::unordered_map<uint32_t, ::tt::runtime::Tensor>>
      sdpaPositions;
  std::unordered_map<int, std::unordered_map<uint32_t, ::tt::runtime::Tensor>>
      ropeCos;
  std::unordered_map<int, std::unordered_map<uint32_t, ::tt::runtime::Tensor>>
      ropeSin;
  uint32_t uploadedAttentionMasks = 0;
  uint32_t uploadedCachePositions = 0;
  uint32_t uploadedSdpaMasks = 0;
  uint32_t uploadedSdpaPositions = 0;
  uint32_t uploadedRopeCos = 0;
  uint32_t uploadedRopeSin = 0;
  double uploadSeconds = 0.0;
};

static uint64_t getEnvUInt64(const char *name, uint64_t fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0')
    return fallback;
  return static_cast<uint64_t>(parsed);
}

static ExtractedOutput extractLogitsAndKV(
    std::vector<::tt::runtime::Tensor> &outputs, int logitsIndex,
    uint32_t tokenPosition, ::tt::runtime::Device device,
    ::tt::runtime::Binary targetBinary, uint32_t targetProgramIndex,
    const std::unordered_map<std::string, uint32_t> &targetKVInputSlots,
    const std::vector<std::string> &kvRoleMap, buddy::Sampler &sampler,
    const std::vector<int> &recentTokens, bool extractKVOutputs = true,
    uint32_t batchSize = 1,
    const std::vector<uint32_t> *tokenPositions = nullptr,
    const std::vector<std::vector<int>> *recentTokensByBatch = nullptr,
    bool retainTokenOutput = false, bool deferTokenHostReadback = false) {
  if (logitsIndex < 0)
    logitsIndex = static_cast<int>(outputs.size()) + logitsIndex;
  if (logitsIndex < 0 || logitsIndex >= static_cast<int>(outputs.size()))
    throw std::runtime_error("llama31_tt: logits output index out of range");

  const bool retainDecodeLogits =
      std::getenv("BUDDY_LLAMA31_RETAIN_DECODE_LOGITS") != nullptr;

  ExtractedOutput extracted;
  auto t0 = std::chrono::steady_clock::now();
  const TTDataType outputDtype =
      ::tt::runtime::getTensorDataType(outputs[logitsIndex]);
  const bool untilizeOutput = !isTokenIdDataType(outputDtype);
  if (deferTokenHostReadback) {
    if (!isTokenIdDataType(outputDtype))
      throw std::runtime_error(
          "llama31_tt: deferred decode token readback requires token-id output");
    if (extractKVOutputs)
      throw std::runtime_error(
          "llama31_tt: deferred decode token readback does not support "
          "extracting decode KV outputs");
  }
  if (retainDecodeLogits) {
    try {
      ::tt::runtime::setTensorRetain(outputs[logitsIndex], true);
    } catch (...) {
    }
  }
  if (retainTokenOutput || deferTokenHostReadback) {
    try {
      ::tt::runtime::setTensorRetain(outputs[logitsIndex], true);
      extracted.tokenDevice = outputs[logitsIndex];
    } catch (...) {
      extracted.tokenDevice.reset();
    }
  }
  if (deferTokenHostReadback) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (static_cast<int>(i) == logitsIndex)
        continue;
      try {
        ::tt::runtime::setTensorRetain(outputs[i], true);
      } catch (...) {
      }
    }
    return extracted;
  }
  auto hostShards =
      ::tt::runtime::toHost(outputs[logitsIndex], untilizeOutput, true);
  if (hostShards.empty())
    throw std::runtime_error("llama31_tt: logits toHost returned no shards");
  auto host = hostShards.front();
  const TTDataType dtype = ::tt::runtime::getTensorDataType(host);
  const auto shape = ::tt::runtime::getTensorShape(host);
  const auto data = ::tt::runtime::getTensorDataBuffer(host);
  auto t1 = std::chrono::steady_clock::now();
  extracted.logitsToHostSeconds =
      std::chrono::duration<double>(t1 - t0).count();

  size_t vocab = 0;
  size_t seq = 1;
  auto sampleAt = [&](size_t start, const std::vector<int> &recent) {
    return sampleTensor(data, dtype, start, vocab, sampler, recent);
  };
  auto tokenPosForBatch = [&](uint32_t batch) -> size_t {
    if (tokenPositions && batch < tokenPositions->size())
      return (*tokenPositions)[batch];
    return tokenPosition;
  };
  auto recentForBatch = [&](uint32_t batch) -> const std::vector<int> & {
    if (recentTokensByBatch && batch < recentTokensByBatch->size())
      return (*recentTokensByBatch)[batch];
    return recentTokens;
  };
  if (isTokenIdDataType(dtype)) {
    const size_t elements = data.size() / itemSizeFor(dtype);
    if (elements == 0)
      throw std::runtime_error("llama31_tt: empty token-id tensor");
    vocab = 1;
    if (shape.size() >= 2)
      seq = shape.back();
    else if (batchSize > 1 && elements >= batchSize)
      seq = 1;
    else
      seq = elements;
    extracted.tokens.reserve(batchSize);
    for (uint32_t b = 0; b < batchSize; ++b) {
      const size_t start = shape.size() >= 2
                               ? std::min<size_t>(
                                     static_cast<size_t>(b) * seq +
                                         std::min<size_t>(
                                             tokenPosForBatch(b),
                                             seq == 0 ? 0 : seq - 1),
                                     elements - 1)
                               : std::min<size_t>(b, elements - 1);
      extracted.tokens.push_back(sampleAt(start, recentForBatch(b)));
    }
  } else if (shape.size() >= 3) {
    vocab = shape.back();
    seq = shape[shape.size() - 2];
    extracted.tokens.reserve(batchSize);
    for (uint32_t b = 0; b < batchSize; ++b) {
      const size_t pos = std::min<size_t>(tokenPosForBatch(b), seq - 1);
      extracted.tokens.push_back(
          sampleAt((static_cast<size_t>(b) * seq + pos) * vocab,
                   recentForBatch(b)));
    }
  } else if (shape.size() >= 2) {
    vocab = shape.back();
    extracted.tokens.reserve(batchSize);
    for (uint32_t b = 0; b < batchSize; ++b)
      extracted.tokens.push_back(
          sampleAt(static_cast<size_t>(b) * vocab, recentForBatch(b)));
  } else {
    vocab = data.size() / itemSizeFor(dtype);
    extracted.tokens.push_back(sampleAt(0, recentTokens));
  }
  if (extracted.tokens.empty())
    throw std::runtime_error("llama31_tt: no tokens sampled from output");
  extracted.token = extracted.tokens.front();

  if (!extractKVOutputs) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (static_cast<int>(i) == logitsIndex)
        continue;
      try {
        ::tt::runtime::setTensorRetain(outputs[i], true);
      } catch (...) {
      }
    }
    if (!retainDecodeLogits && !retainTokenOutput)
      ::tt::runtime::deallocateTensor(outputs[logitsIndex], true);
    return extracted;
  }

  t0 = std::chrono::steady_clock::now();
  size_t kvIter = 0;
  std::vector<bool> consumed(outputs.size(), false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (static_cast<int>(i) == logitsIndex)
      continue;
    if (kvIter >= kvRoleMap.size())
      break;
    const std::string &role = kvRoleMap[kvIter++];
    auto slotIt = targetKVInputSlots.find(role);
    if (slotIt == targetKVInputSlots.end())
      throw std::runtime_error("llama31_tt: missing target KV slot for " +
                               role);
    auto layout = ::tt::runtime::getLayout(targetBinary, targetProgramIndex,
                                           slotIt->second);
    try {
      ::tt::runtime::setTensorRetain(outputs[i], true);
    } catch (...) {
    }
    if (::tt::runtime::hasLayout(outputs[i], layout)) {
      extracted.kvDevice[role] = outputs[i];
    } else {
      extracted.kvDevice[role] =
          ::tt::runtime::toLayout(outputs[i], device, layout);
    }
    consumed[i] = true;
  }
  t1 = std::chrono::steady_clock::now();
  extracted.kvRelayoutSeconds = std::chrono::duration<double>(t1 - t0).count();

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (retainTokenOutput && static_cast<int>(i) == logitsIndex)
      continue;
    if (consumed[i])
      continue;
    bool force = static_cast<int>(i) == logitsIndex;
    ::tt::runtime::deallocateTensor(outputs[i], force);
  }
  return extracted;
}

static std::vector<std::string>
buildPrefillKVOutputRoles(const std::vector<std::string> &decodeKVOutputRoles,
                          const std::string &order) {
  if (order == "key_value")
    return decodeKVOutputRoles;
  if (order != "value_key")
    throw std::runtime_error("llama_tt: unsupported prefill_kv_output_order '" +
                             order + "'");

  if (decodeKVOutputRoles.size() % 2 != 0)
    throw std::runtime_error(
        "llama_tt: value_key prefill KV order requires paired KV roles");

  std::vector<std::string> prefillRoles;
  prefillRoles.reserve(decodeKVOutputRoles.size());
  for (size_t i = 0; i < decodeKVOutputRoles.size(); i += 2) {
    const std::string &keyRole = decodeKVOutputRoles[i];
    const std::string &valueRole = decodeKVOutputRoles[i + 1];
    if (!startsWith(keyRole, "past_K_") || !startsWith(valueRole, "past_V_") ||
        keyRole.substr(7) != valueRole.substr(7))
      throw std::runtime_error(
          "llama_tt: decode KV roles are not ordered as K,V pairs");
    prefillRoles.push_back(valueRole);
    prefillRoles.push_back(keyRole);
  }
  return prefillRoles;
}

static bool sameTensorDesc(const ::tt::runtime::TensorDesc &lhs,
                           const ::tt::runtime::TensorDesc &rhs) {
  return lhs.shape == rhs.shape && lhs.dataType == rhs.dataType;
}

static bool sameStaticPayload(const PhaseCtx &lhsCtx, uint32_t lhsSlot,
                              const PhaseCtx &rhsCtx, uint32_t rhsSlot) {
  const RoleEntry &lhs = lhsCtx.roles[lhsSlot];
  const RoleEntry &rhs = rhsCtx.roles[rhsSlot];
  if (lhs.role != rhs.role || lhs.shape != rhs.shape || lhs.dtype != rhs.dtype)
    return false;

  if (lhs.role == "weight") {
    return lhs.weightOffset == rhs.weightOffset &&
           lhs.weightBytes == rhs.weightBytes &&
           sameFile(lhsCtx.weightsPath, rhsCtx.weightsPath);
  }
  if (lhs.role == "inv_freq")
    return sameFile(lhsCtx.invFreqPath, rhsCtx.invFreqPath);
  return false;
}

static std::optional<uint32_t> findReusableStaticInputSlot(
    const PhaseCtx &sourceCtx,
    const std::vector<::tt::runtime::TensorDesc> &sourceInputs,
    const PhaseCtx &targetCtx,
    const std::vector<::tt::runtime::TensorDesc> &targetInputs,
    uint32_t targetSlot) {
  for (uint32_t sourceSlot = 0; sourceSlot < sourceCtx.roles.size();
       ++sourceSlot) {
    const RoleEntry &sourceRole = sourceCtx.roles[sourceSlot];
    if (sourceRole.role != "weight" && sourceRole.role != "inv_freq")
      continue;
    if (!sameStaticPayload(sourceCtx, sourceSlot, targetCtx, targetSlot))
      continue;
    if (!sameTensorDesc(sourceInputs[sourceSlot], targetInputs[targetSlot]))
      continue;
    return sourceSlot;
  }
  return std::nullopt;
}

static StaticInputCacheResult
precacheStaticInputs(::tt::runtime::Device device, ::tt::runtime::Binary binary,
                     uint32_t programIndex, const PhaseCtx &ctx,
                     const std::vector<::tt::runtime::TensorDesc> &inputs,
                     const PhaseCtx *reuseCtx = nullptr,
                     const std::vector<::tt::runtime::TensorDesc>
                         *reuseInputs = nullptr,
                     const std::unordered_map<uint32_t, ::tt::runtime::Tensor>
                         *reuseCache = nullptr,
                     const char *phaseName = "static",
                     bool suppress = false) {
  StaticInputCacheResult result;
  const bool progress =
      !suppress && std::getenv("BUDDY_LLAMA31_STATIC_UPLOAD_PROGRESS");
  const bool retainUploadedStatic =
      phaseName && std::string_view(phaseName) == "decode";
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    const RoleEntry &role = ctx.roles[slot];
    if (role.role != "weight" && role.role != "inv_freq")
      continue;

    if (reuseCtx && reuseInputs && reuseCache) {
      if (auto sourceSlot =
              findReusableStaticInputSlot(*reuseCtx, *reuseInputs, ctx, inputs,
                                          slot)) {
        auto reusable = reuseCache->find(*sourceSlot);
        if (reusable != reuseCache->end()) {
          auto layout = ::tt::runtime::getLayout(binary, programIndex, slot);
          if (::tt::runtime::hasLayout(reusable->second, layout)) {
            // Reused static tensors alias the prefill cache; keep the handle
            // retained so decode submits can reuse it across iterations.
            try {
              ::tt::runtime::setTensorRetain(reusable->second, true);
            } catch (...) {
            }
            result.tensors[slot] = reusable->second;
            ++result.reused;
            continue;
          }
        }
      }
    }

    ::tt::runtime::Tensor host;
    if (role.role == "weight") {
      host = borrowedHostTensorFromRaw(ctx.weightForSlot(slot), inputs[slot]);
    } else {
      if (!ctx.invFreq)
        throw std::runtime_error("llama31_tt: missing inv_freq.npy");
      host = borrowedHostTensorFromNpy(ctx.invFreq->view(), inputs[slot]);
    }
    if (progress) {
      const uint64_t bytes =
          role.role == "weight" ? role.weightBytes : ctx.invFreq->view().bytes;
      std::cout << colorLabel("[Static upload]") << " " << phaseName
                << " slot " << slot << " " << role.role << " "
                << bytes << " bytes\n";
      std::cout.flush();
    }
    result.tensors[slot] = toDevice(host, device, binary, programIndex, slot,
                                    retainUploadedStatic);
    ++result.uploaded;
    if (progress) {
      std::cout << colorLabel("[Static upload]") << " " << phaseName
                << " slot " << slot << " done\n";
      std::cout.flush();
    }
  }
  return result;
}

static std::vector<uint8_t> buildAttentionMaskBuffer(
    const ::tt::runtime::TensorDesc &desc,
    const std::vector<std::vector<int>> &attentionMaskByBatch,
    std::string_view errorPrefix) {
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(volume(desc.shape)));
  for (const auto &mask : attentionMaskByBatch)
    values.insert(values.end(), mask.begin(), mask.end());
  if (values.size() != static_cast<size_t>(volume(desc.shape)))
    throw std::runtime_error(std::string(errorPrefix) +
                             " attention_mask shape does not match "
                             "batch/mask buffer");
  return integerBuffer(values, desc.dataType);
}

static std::vector<uint8_t>
buildCachePositionBuffer(const ::tt::runtime::TensorDesc &desc, int cachePos) {
  std::vector<int64_t> values(static_cast<size_t>(volume(desc.shape)),
                              cachePos);
  return integerBuffer(values, desc.dataType);
}

static std::vector<uint8_t> buildSdpaMaskBuffer(
    const ::tt::runtime::TensorDesc &desc,
    const std::vector<std::vector<int>> &attentionMaskByBatch, int cachePos,
    std::string_view errorPrefix) {
  if (desc.shape.size() != 4)
    throw std::runtime_error(std::string(errorPrefix) +
                             " sdpa_mask must be rank-4");
  const uint32_t batch = desc.shape[0];
  const uint32_t groups = desc.shape[1];
  const uint32_t heads = desc.shape[2];
  const uint32_t seq = desc.shape[3];
  if (attentionMaskByBatch.size() != batch)
    throw std::runtime_error(std::string(errorPrefix) +
                             " sdpa_mask batch mismatch");

  const float negInf = -std::numeric_limits<float>::infinity();
  std::vector<float> values;
  values.reserve(static_cast<size_t>(volume(desc.shape)));
  for (uint32_t b = 0; b < batch; ++b) {
    if (attentionMaskByBatch[b].size() < seq)
      throw std::runtime_error(std::string(errorPrefix) +
                               " sdpa_mask sequence mismatch");
    for (uint32_t g = 0; g < groups; ++g) {
      (void)g;
      for (uint32_t h = 0; h < heads; ++h) {
        (void)h;
        for (uint32_t i = 0; i < seq; ++i) {
          const bool visible =
              attentionMaskByBatch[b][i] != 0 &&
              static_cast<int>(i) <= cachePos;
          values.push_back(visible ? 0.0f : negInf);
        }
      }
    }
  }
  return floatBuffer(values, desc.dataType);
}

static std::vector<float> findInvFreqValues(const PhaseCtx &ctx,
                                            size_t expectedSize) {
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    const RoleEntry &role = ctx.roles[slot];
    if (role.role != "weight" || role.dtype != "float32" ||
        role.shape.size() != 1 || role.shape[0] != expectedSize)
      continue;
    RawTensorView view = ctx.weightForSlot(slot);
    if (view.bytes != expectedSize * sizeof(float))
      throw std::runtime_error("llama31_tt: invalid inv_freq byte size");
    const float *data = reinterpret_cast<const float *>(view.data);
    return std::vector<float>(data, data + expectedSize);
  }
  throw std::runtime_error("llama31_tt: could not find inv_freq weight slot");
}

static std::vector<uint8_t> buildRopeTrigBuffer(
    const ::tt::runtime::TensorDesc &desc, const PhaseCtx &ctx, int cachePos,
    bool cosine, std::string_view errorPrefix) {
  if (desc.shape.size() != 4 || desc.shape[0] != 1 || desc.shape[1] != 1 ||
      desc.shape[2] != 1 || desc.shape[3] % 2 != 0)
    throw std::runtime_error(std::string(errorPrefix) +
                             " rope tensor must have shape 1x1x1x2N");
  const size_t width = desc.shape[3];
  const size_t halfWidth = width / 2;
  std::vector<float> invFreq = findInvFreqValues(ctx, halfWidth);
  std::vector<float> values;
  values.reserve(width);
  for (size_t i = 0; i < width; ++i) {
    const float angle = invFreq[i % halfWidth] * static_cast<float>(cachePos);
    values.push_back(cosine ? std::cos(angle) : std::sin(angle));
  }
  return floatBuffer(values, desc.dataType);
}

static DecodeRuntimeInputCache precacheDecodeRuntimeInputs(
    ::tt::runtime::Device device, ::tt::runtime::Binary binary,
    uint32_t programIndex, const PhaseCtx &ctx,
    const std::vector<::tt::runtime::TensorDesc> &programInputs,
    const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cachedInputs,
    const std::vector<std::vector<int>> &attentionMaskByBatch, int firstCachePos,
    int maxCacheLen, bool suppress) {
  DecodeRuntimeInputCache result;
  if (std::getenv("BUDDY_LLAMA31_DISABLE_DECODE_RUNTIME_CACHE") != nullptr)
    return result;

  auto uploadStart = std::chrono::steady_clock::now();
  std::vector<uint32_t> cachePositionSlots;
  std::vector<uint32_t> sdpaMaskSlots;
  std::vector<uint32_t> sdpaPositionSlots;
  std::vector<uint32_t> ropeCosSlots;
  std::vector<uint32_t> ropeSinSlots;
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    if (cachedInputs.find(slot) != cachedInputs.end())
      continue;
    const RoleEntry &role = ctx.roles[slot];
    const auto &desc = programInputs[slot];
    if (role.role == "attention_mask") {
      std::vector<uint8_t> buffer = buildAttentionMaskBuffer(
          desc, attentionMaskByBatch, "llama31_tt: decode");
      auto host = hostTensorFromBytes(buffer.data(), desc);
      result.attentionMasks[slot] =
          toDevice(host, device, binary, programIndex, slot, true);
      ++result.uploadedAttentionMasks;
    } else if (role.role == "cache_position") {
      cachePositionSlots.push_back(slot);
    } else if (role.role == "sdpa_mask") {
      sdpaMaskSlots.push_back(slot);
    } else if (role.role == "sdpa_position") {
      sdpaPositionSlots.push_back(slot);
    } else if (role.role == "rope_cos") {
      ropeCosSlots.push_back(slot);
    } else if (role.role == "rope_sin") {
      ropeSinSlots.push_back(slot);
    }
  }

  const bool disableCachePositionCache =
      std::getenv("BUDDY_LLAMA31_DISABLE_CACHE_POSITION_CACHE") != nullptr;
  const int begin = std::max(0, firstCachePos);
  const int end = std::max(begin, maxCacheLen);
  const uint64_t cachePositionUploads =
      static_cast<uint64_t>(std::max(0, end - begin)) *
      static_cast<uint64_t>(cachePositionSlots.size());
  const uint64_t cachePositionUploadLimit = getEnvUInt64(
      "BUDDY_LLAMA31_CACHE_POSITION_PRECACHE_LIMIT", 8192);

  if (!disableCachePositionCache &&
      cachePositionUploads <= cachePositionUploadLimit) {
    for (int pos = begin; pos < end; ++pos) {
      auto &positionTensors = result.cachePositions[pos];
      for (uint32_t slot : cachePositionSlots) {
        const auto &desc = programInputs[slot];
        std::vector<uint8_t> buffer = buildCachePositionBuffer(desc, pos);
        auto host = hostTensorFromBytes(buffer.data(), desc);
        positionTensors[slot] =
            toDevice(host, device, binary, programIndex, slot, true);
        ++result.uploadedCachePositions;
      }
      auto &sdpaMaskTensors = result.sdpaMasks[pos];
      for (uint32_t slot : sdpaMaskSlots) {
        const auto &desc = programInputs[slot];
        std::vector<uint8_t> buffer = buildSdpaMaskBuffer(
            desc, attentionMaskByBatch, pos, "llama31_tt: decode");
        auto host = hostTensorFromBytes(buffer.data(), desc);
        sdpaMaskTensors[slot] =
            toDevice(host, device, binary, programIndex, slot, true);
        ++result.uploadedSdpaMasks;
      }
      auto &sdpaPositionTensors = result.sdpaPositions[pos];
      for (uint32_t slot : sdpaPositionSlots) {
        const auto &desc = programInputs[slot];
        std::vector<uint8_t> buffer = buildCachePositionBuffer(desc, pos);
        auto host = hostTensorFromBytes(buffer.data(), desc);
        sdpaPositionTensors[slot] =
            toDevice(host, device, binary, programIndex, slot, true);
        ++result.uploadedSdpaPositions;
      }
      auto &ropeCosTensors = result.ropeCos[pos];
      for (uint32_t slot : ropeCosSlots) {
        const auto &desc = programInputs[slot];
        std::vector<uint8_t> buffer = buildRopeTrigBuffer(
            desc, ctx, pos, true, "llama31_tt: decode");
        auto host = hostTensorFromBytes(buffer.data(), desc);
        ropeCosTensors[slot] =
            toDevice(host, device, binary, programIndex, slot, true);
        ++result.uploadedRopeCos;
      }
      auto &ropeSinTensors = result.ropeSin[pos];
      for (uint32_t slot : ropeSinSlots) {
        const auto &desc = programInputs[slot];
        std::vector<uint8_t> buffer = buildRopeTrigBuffer(
            desc, ctx, pos, false, "llama31_tt: decode");
        auto host = hostTensorFromBytes(buffer.data(), desc);
        ropeSinTensors[slot] =
            toDevice(host, device, binary, programIndex, slot, true);
        ++result.uploadedRopeSin;
      }
    }
  } else if (!suppress &&
             (!cachePositionSlots.empty() || !sdpaMaskSlots.empty() ||
              !sdpaPositionSlots.empty() || !ropeCosSlots.empty() ||
              !ropeSinSlots.empty())) {
    printLlamaLog("decode cache_position precache skipped: " +
                      std::to_string(cachePositionUploads) +
                      " tensors exceeds limit " +
                      std::to_string(cachePositionUploadLimit),
                  suppress);
  }

  auto uploadEnd = std::chrono::steady_clock::now();
  result.uploadSeconds =
      std::chrono::duration<double>(uploadEnd - uploadStart).count();
  if (!suppress && (result.uploadedAttentionMasks ||
                    result.uploadedCachePositions ||
                    result.uploadedSdpaMasks ||
                    result.uploadedSdpaPositions ||
                    result.uploadedRopeCos ||
                    result.uploadedRopeSin)) {
    printLlamaLog("decode runtime tensors ready: uploaded " +
                      std::to_string(result.uploadedAttentionMasks) +
                      " attention_mask + " +
                      std::to_string(result.uploadedCachePositions) +
                      " cache_position + " +
                      std::to_string(result.uploadedSdpaMasks) +
                      " sdpa_mask + " +
                      std::to_string(result.uploadedSdpaPositions) +
                      " sdpa_position + " +
                      std::to_string(result.uploadedRopeCos) +
                      " rope_cos + " +
                      std::to_string(result.uploadedRopeSin) +
                      " rope_sin in " +
                      formatFixed(result.uploadSeconds) + "s",
                  suppress);
  }
  return result;
}

static void deallocateDecodeRuntimeInputCache(
    DecodeRuntimeInputCache &cache) {
  for (auto &entry : cache.attentionMasks) {
    try {
      ::tt::runtime::deallocateTensor(entry.second, true);
    } catch (...) {
    }
  }
  cache.attentionMasks.clear();
  for (auto &positionEntry : cache.cachePositions) {
    for (auto &entry : positionEntry.second) {
      try {
        ::tt::runtime::deallocateTensor(entry.second, true);
      } catch (...) {
      }
    }
  }
  cache.cachePositions.clear();
  for (auto &maskEntry : cache.sdpaMasks) {
    for (auto &entry : maskEntry.second) {
      try {
        ::tt::runtime::deallocateTensor(entry.second, true);
      } catch (...) {
      }
    }
  }
  cache.sdpaMasks.clear();
  for (auto &positionEntry : cache.sdpaPositions) {
    for (auto &entry : positionEntry.second) {
      try {
        ::tt::runtime::deallocateTensor(entry.second, true);
      } catch (...) {
      }
    }
  }
  cache.sdpaPositions.clear();
  for (auto &entryByPosition : cache.ropeCos) {
    for (auto &entry : entryByPosition.second) {
      try {
        ::tt::runtime::deallocateTensor(entry.second, true);
      } catch (...) {
      }
    }
  }
  cache.ropeCos.clear();
  for (auto &entryByPosition : cache.ropeSin) {
    for (auto &entry : entryByPosition.second) {
      try {
        ::tt::runtime::deallocateTensor(entry.second, true);
      } catch (...) {
      }
    }
  }
  cache.ropeSin.clear();
}

static std::vector<::tt::runtime::Tensor> buildPrefillInputs(
    ::tt::runtime::Device device, ::tt::runtime::Binary binary,
    uint32_t programIndex, const PhaseCtx &ctx,
    const std::vector<::tt::runtime::TensorDesc> &programInputs,
    const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cachedInputs,
    const std::vector<std::vector<int>> &paddedTokensByBatch,
    const std::vector<std::vector<int>> &attentionMaskByBatch) {
  std::vector<::tt::runtime::Tensor> tensors(programInputs.size());
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    auto cached = cachedInputs.find(slot);
    if (cached != cachedInputs.end()) {
      tensors[slot] = cached->second;
      continue;
    }
    const auto &desc = programInputs[slot];
    const RoleEntry &role = ctx.roles[slot];
    std::vector<uint8_t> buffer;
    if (role.role == "input_ids") {
      std::vector<int64_t> values;
      values.reserve(static_cast<size_t>(volume(desc.shape)));
      for (const auto &paddedTokens : paddedTokensByBatch)
        values.insert(values.end(), paddedTokens.begin(), paddedTokens.end());
      if (values.size() != static_cast<size_t>(volume(desc.shape)))
        throw std::runtime_error("llama31_tt: prefill input_ids shape does not "
                                 "match batch/token buffer");
      buffer = integerBuffer(values, desc.dataType);
    } else if (role.role == "attention_mask") {
      buffer = buildAttentionMaskBuffer(desc, attentionMaskByBatch,
                                        "llama31_tt: prefill");
    } else if (role.role == "cache_position") {
      std::vector<int64_t> values(volume(desc.shape));
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<int64_t>(i);
      buffer = integerBuffer(values, desc.dataType);
    } else if (startsWith(role.role, "past_K_") ||
               startsWith(role.role, "past_V_")) {
      buffer = zeroBuffer(desc);
    } else {
      throw std::runtime_error("llama31_tt: unexpected prefill role " +
                               role.role);
    }
    auto host = hostTensorFromBytes(buffer.data(), desc);
    tensors[slot] = toDevice(host, device, binary, programIndex, slot);
  }
  return tensors;
}

static std::vector<::tt::runtime::Tensor> buildDecodeInputs(
    ::tt::runtime::Device device, ::tt::runtime::Binary binary,
    uint32_t programIndex, const PhaseCtx &ctx,
    const std::vector<::tt::runtime::TensorDesc> &programInputs,
    const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cachedInputs,
    const std::vector<int> &tokens, int cachePos,
    const std::unordered_map<std::string, ::tt::runtime::Tensor> &pastKV,
    const std::vector<std::vector<int>> &attentionMaskByBatch,
    const DecodeRuntimeInputCache *runtimeInputCache = nullptr,
    const ::tt::runtime::Tensor *deviceInputIds = nullptr) {
  std::vector<::tt::runtime::Tensor> tensors(programInputs.size());
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    auto cached = cachedInputs.find(slot);
    if (cached != cachedInputs.end()) {
      tensors[slot] = cached->second;
      continue;
    }
    const auto &desc = programInputs[slot];
    const RoleEntry &role = ctx.roles[slot];
    if (startsWith(role.role, "past_K_") || startsWith(role.role, "past_V_")) {
      auto kv = pastKV.find(role.role);
      if (kv == pastKV.end())
        throw std::runtime_error("llama31_tt: missing KV tensor " + role.role);
      ::tt::runtime::setTensorRetain(kv->second, true);
      tensors[slot] = kv->second;
      continue;
    }
    if (runtimeInputCache && role.role == "attention_mask") {
      auto cachedMask = runtimeInputCache->attentionMasks.find(slot);
      if (cachedMask != runtimeInputCache->attentionMasks.end()) {
        tensors[slot] = cachedMask->second;
        continue;
      }
    }
    if (runtimeInputCache && role.role == "cache_position") {
      auto cachedPosition = runtimeInputCache->cachePositions.find(cachePos);
      if (cachedPosition != runtimeInputCache->cachePositions.end()) {
        auto cachedSlot = cachedPosition->second.find(slot);
        if (cachedSlot != cachedPosition->second.end()) {
          tensors[slot] = cachedSlot->second;
          continue;
        }
      }
    }
    if (runtimeInputCache && role.role == "sdpa_mask") {
      auto cachedMask = runtimeInputCache->sdpaMasks.find(cachePos);
      if (cachedMask != runtimeInputCache->sdpaMasks.end()) {
        auto cachedSlot = cachedMask->second.find(slot);
        if (cachedSlot != cachedMask->second.end()) {
          tensors[slot] = cachedSlot->second;
          continue;
        }
      }
    }
    if (runtimeInputCache && role.role == "sdpa_position") {
      auto cachedPosition = runtimeInputCache->sdpaPositions.find(cachePos);
      if (cachedPosition != runtimeInputCache->sdpaPositions.end()) {
        auto cachedSlot = cachedPosition->second.find(slot);
        if (cachedSlot != cachedPosition->second.end()) {
          tensors[slot] = cachedSlot->second;
          continue;
        }
      }
    }
    if (runtimeInputCache && role.role == "rope_cos") {
      auto cachedByPosition = runtimeInputCache->ropeCos.find(cachePos);
      if (cachedByPosition != runtimeInputCache->ropeCos.end()) {
        auto cachedSlot = cachedByPosition->second.find(slot);
        if (cachedSlot != cachedByPosition->second.end()) {
          tensors[slot] = cachedSlot->second;
          continue;
        }
      }
    }
    if (runtimeInputCache && role.role == "rope_sin") {
      auto cachedByPosition = runtimeInputCache->ropeSin.find(cachePos);
      if (cachedByPosition != runtimeInputCache->ropeSin.end()) {
        auto cachedSlot = cachedByPosition->second.find(slot);
        if (cachedSlot != cachedByPosition->second.end()) {
          tensors[slot] = cachedSlot->second;
          continue;
        }
      }
    }

    std::vector<uint8_t> buffer;
    if (role.role == "input_ids") {
      if (deviceInputIds) {
        try {
          auto layout = ::tt::runtime::getLayout(binary, programIndex, slot);
          if (::tt::runtime::getTensorShape(*deviceInputIds) == desc.shape &&
              ::tt::runtime::getTensorDataType(*deviceInputIds) ==
                  desc.dataType &&
              ::tt::runtime::hasLayout(*deviceInputIds, layout)) {
            ::tt::runtime::setTensorRetain(*deviceInputIds, true);
            tensors[slot] = *deviceInputIds;
            continue;
          }
        } catch (...) {
        }
      }
      std::vector<int64_t> values(tokens.begin(), tokens.end());
      if (values.size() == 1 && volume(desc.shape) > 1)
        values.assign(static_cast<size_t>(volume(desc.shape)), values.front());
      if (values.size() != static_cast<size_t>(volume(desc.shape)))
        throw std::runtime_error("llama31_tt: decode input_ids shape does not "
                                 "match batch token buffer");
      buffer = integerBuffer(values, desc.dataType);
    } else if (role.role == "attention_mask") {
      buffer =
          buildAttentionMaskBuffer(desc, attentionMaskByBatch, "llama31_tt: decode");
    } else if (role.role == "cache_position") {
      buffer = buildCachePositionBuffer(desc, cachePos);
    } else if (role.role == "sdpa_mask") {
      buffer = buildSdpaMaskBuffer(desc, attentionMaskByBatch, cachePos,
                                   "llama31_tt: decode");
    } else if (role.role == "sdpa_position") {
      buffer = buildCachePositionBuffer(desc, cachePos);
    } else if (role.role == "rope_cos") {
      buffer = buildRopeTrigBuffer(desc, ctx, cachePos, true,
                                   "llama31_tt: decode");
    } else if (role.role == "rope_sin") {
      buffer = buildRopeTrigBuffer(desc, ctx, cachePos, false,
                                   "llama31_tt: decode");
    } else {
      throw std::runtime_error("llama31_tt: unexpected decode role " +
                               role.role);
    }
    auto host = hostTensorFromBytes(buffer.data(), desc);
    tensors[slot] = toDevice(host, device, binary, programIndex, slot);
  }
  return tensors;
}

static void retainKVCache(
    std::unordered_map<std::string, ::tt::runtime::Tensor> &pastKV) {
  for (auto &kv : pastKV) {
    try {
      ::tt::runtime::setTensorRetain(kv.second, true);
    } catch (...) {
    }
  }
}

static void debugKVCacheState(
    const std::unordered_map<std::string, ::tt::runtime::Tensor> &pastKV,
    const std::string &label) {
  if (std::getenv("BUDDY_LLAMA31_DEBUG_ALLOCATIONS") == nullptr)
    return;
  std::cout << colorLabel("[KV alloc]", kAnsiBlueBold) << " " << label;
  for (const auto &kv : pastKV) {
    std::cout << " " << kv.first << "="
              << (::tt::runtime::isTensorAllocated(kv.second) ? "1" : "0");
  }
  std::cout << "\n";
}

static void debugTensorVectorState(
    const std::vector<::tt::runtime::Tensor> &tensors,
    const std::vector<RoleEntry> &roles, const std::string &label) {
  if (std::getenv("BUDDY_LLAMA31_DEBUG_ALLOCATIONS") == nullptr)
    return;
  std::cout << colorLabel("[Tensor alloc]", kAnsiBlueBold) << " " << label;
  bool anyMissing = false;
  for (size_t i = 0; i < tensors.size() && i < roles.size(); ++i) {
    if (!::tt::runtime::isTensorAllocated(tensors[i])) {
      if (!anyMissing) {
        std::cout << " missing:";
        anyMissing = true;
      }
      std::cout << " " << i << ":" << roles[i].role;
    }
  }
  if (!anyMissing)
    std::cout << " all allocated";
  std::cout << "\n";
}

static void debugTensorState(const std::vector<::tt::runtime::Tensor> &tensors,
                             const std::string &label) {
  if (std::getenv("BUDDY_LLAMA31_DEBUG_ALLOCATIONS") == nullptr)
    return;
  std::cout << colorLabel("[Tensor vec]", kAnsiBlueBold) << " " << label;
  bool anyMissing = false;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!::tt::runtime::isTensorAllocated(tensors[i])) {
      if (!anyMissing) {
        std::cout << " missing:";
        anyMissing = true;
      }
      std::cout << " " << i;
    }
  }
  if (!anyMissing)
    std::cout << " all allocated";
  std::cout << "\n";
}

static fs::path resolveTokenizerPath(const ModelManifest &manifest,
                                     const std::string &tokenizerAttr) {
  const fs::path embeddedTokenizer = resolveEmbeddedTokenizerPath(manifest);
  if (!embeddedTokenizer.empty())
    return embeddedTokenizer;

  std::string value = getEnvOr("LLAMA_MODEL_PATH",
                               getEnvOr("LLAMA32_MODEL_PATH",
                                        getEnvOr("LLAMA31_MODEL_PATH",
                                                 tokenizerAttr)));
  if (startsWith(value, "file:"))
    value = value.substr(5);
  if (isRemoteModelId(value))
    throw std::runtime_error(
        "llama_tt: tokenizer/model path is a remote Hugging Face id ('" +
        value +
        "'). Pure C++ buddy-cli runtime does not download models; use a .rax "
        "package with embedded tokenizer files, or set "
        "LLAMA_MODEL_PATH, LLAMA31_MODEL_PATH, or LLAMA32_MODEL_PATH to a "
        "local Llama model directory when "
        "running or packaging.");
  return fs::absolute(fs::path(value));
}

static bool isValidTTMetalRuntimeRoot(const fs::path &root) {
  std::error_code ec;
  return fs::is_directory(root / "tt_metal" / "soc_descriptors", ec);
}

static void addTTMetalRootCandidates(std::vector<fs::path> &candidates,
                                     const fs::path &anchor) {
  if (anchor.empty())
    return;

  std::error_code ec;
  fs::path current = fs::absolute(anchor, ec);
  if (ec)
    current = anchor;
  if (fs::is_regular_file(current, ec))
    current = current.parent_path();

  while (!current.empty()) {
    candidates.push_back(current / "thirdparty" / "tt-mlir" / "third_party" /
                         "tt-metal" / "src" / "tt-metal");
    candidates.push_back(current / "third_party" / "tt-metal" / "src" /
                         "tt-metal");
    if (current == current.root_path())
      break;
    current = current.parent_path();
  }
}

static void ensureTTMetalRuntimeRoot(const fs::path &raxDir) {
  if (const char *root = std::getenv("TT_METAL_RUNTIME_ROOT")) {
    if (root[0] != '\0' && isValidTTMetalRuntimeRoot(root)) {
      setenv("TT_METAL_HOME", root, 1);
      return;
    }
  }

  std::vector<fs::path> candidates;
  if (const char *ttmlirSource = std::getenv("TTMLIR_SOURCE"))
    candidates.push_back(fs::path(ttmlirSource) / "third_party" / "tt-metal" /
                         "src" / "tt-metal");
  if (const char *repoRoot = std::getenv("BUDDY_REPO_ROOT"))
    candidates.push_back(fs::path(repoRoot) / "thirdparty" / "tt-mlir" /
                         "third_party" / "tt-metal" / "src" / "tt-metal");
  addTTMetalRootCandidates(candidates, raxDir);
  addTTMetalRootCandidates(candidates, fs::current_path());

  for (const fs::path &candidate : candidates) {
    if (!isValidTTMetalRuntimeRoot(candidate))
      continue;
    const std::string root = candidate.string();
    const std::string buildRoot = (candidate / "build").string();
    setenv("TT_METAL_RUNTIME_ROOT", root.c_str(), 1);
    setenv("TT_METAL_HOME", root.c_str(), 1);
    setenv("TT_METAL_BUILD_HOME", buildRoot.c_str(), 1);
    return;
  }
}

} // namespace

void Llama31TTRunner::run(const RunConfig &cfg) {
  raiseMemoryLimitForLlama();
  keepLibPythonVisibleForTTMLIRRuntime();

  if (cfg.raxPath.empty())
    throw std::runtime_error("llama31_tt requires --model <path.rax>");
  const bool suppress = cfg.suppressStats;

  const ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
  const std::string prefillTTNN = findTTNNArtifact(manifest, "prefill");
  const std::string decodeTTNN = findTTNNArtifact(manifest, "decode");

  const fs::path raxDir = fs::absolute(fs::path(cfg.raxPath)).parent_path();
  std::string artifacts = materializeEmbeddedArtifacts(manifest, raxDir);
  if (artifacts.empty())
    artifacts = lookupAttr(manifest, "artifacts_uri",
                           (raxDir / "chat_artifacts").string());

  const int maxCacheLen =
      static_cast<int>(lookupIntAttr(manifest, "max_cache_len", 1024));
  const int eosToken =
      static_cast<int>(lookupIntAttr(manifest, "eos_token_id", kEot));
  const bool ignoreEOS = lookupBoolAttr(manifest, "ignore_eos", false);
  const std::string modelName =
      manifest.modelName.empty() ? "llama_tt" : manifest.modelName;
  const std::string promptFormat =
      lookupAttr(manifest, "prompt_format", "chat");
  if (promptFormat != "chat" && promptFormat != "completion")
    throw std::runtime_error("llama_tt: unsupported prompt_format '" +
                             promptFormat + "'");
  const std::string prefillKVOutputOrder =
      lookupAttr(manifest, "prefill_kv_output_order", "key_value");
  const int manifestBatchSize =
      static_cast<int>(lookupIntAttr(manifest, "batch_size", 1));
  const int batchSize = cfg.batchSize > 0 ? cfg.batchSize : manifestBatchSize;
  if (batchSize <= 0)
    throw std::runtime_error("llama_tt: batch size must be positive");
  if (cfg.batchSize > 0 && cfg.batchSize != manifestBatchSize)
    throw std::runtime_error(
        "llama_tt: --batch-size does not match fixed-batch package");
  std::vector<int> stopTokenIds;
  auto addStopToken = [&](int token) {
    if (std::find(stopTokenIds.begin(), stopTokenIds.end(), token) ==
        stopTokenIds.end())
      stopTokenIds.push_back(token);
  };
  if (!ignoreEOS) {
    addStopToken(eosToken);
    addStopToken(kEndOfText);
    addStopToken(kEom);
    addStopToken(kEot);
  }
  auto isStopToken = [&](int token) {
    if (ignoreEOS)
      return false;
    return std::find(stopTokenIds.begin(), stopTokenIds.end(), token) !=
           stopTokenIds.end();
  };
  const uint32_t programIndex =
      static_cast<uint32_t>(lookupIntAttr(manifest, "program_index", 0));
  const fs::path tokenizerPath = resolveTokenizerPath(manifest, lookupAttr(
      manifest, "tokenizer_uri", "meta-llama/Llama-3.1-8B-Instruct"));
  ensureTTMetalRuntimeRoot(raxDir);

  if (!suppress) {
    std::cout << "[buddy-cli] dispatch " << modelName
              << " via native Tenstorrent C++ runtime\n";
    std::cout << "[buddy-cli] prefill: " << prefillTTNN << "\n";
    std::cout << "[buddy-cli] decode:  " << decodeTTNN << "\n";
    std::cout << "[buddy-cli] artifacts: " << artifacts << "\n";
  }

  const auto tokenizerStart = std::chrono::steady_clock::now();
  Llama31Tokenizer tokenizer(tokenizerPath);
  const auto tokenizerEnd = std::chrono::steady_clock::now();
  printLlamaLog("tokenizer ready in " +
                    formatFixed(std::chrono::duration<double, std::milli>(
                                    tokenizerEnd - tokenizerStart)
                                    .count()) +
                    "ms",
                suppress);

  std::string prompt = cfg.prompt;
  if (prompt.empty() && cfg.prompts.empty()) {
    std::cout << "Prompt: ";
    std::getline(std::cin, prompt);
    std::cout << "\n";
  }
  const bool officialTokenMatching = prompt == "official-refpt";
  std::optional<OfficialTokenReference> officialRef;
  if (officialTokenMatching) {
    if (batchSize != 1)
      throw std::runtime_error(
          "llama31_tt: official-refpt token matching requires batch_size=1");
    officialRef = loadOfficialTokenReferenceFromEnv();
    if (!officialRef)
      throw std::runtime_error(
          "llama31_tt: --prompt official-refpt requires "
          "BUDDY_LLAMA31_OFFICIAL_REFERENCE_JSON");
    printLlamaLog("official token matching reference: " +
                      officialRef->sourcePath,
                  suppress);
  }

  auto prefillBinary = ::tt::runtime::Binary::loadFromPath(prefillTTNN.c_str());
  auto decodeBinary = ::tt::runtime::Binary::loadFromPath(decodeTTNN.c_str());
  ::tt::runtime::setCompatibleDeviceRuntime(prefillBinary);

  const auto prefillInputs = prefillBinary.getProgramInputs(programIndex);
  const auto decodeInputs = decodeBinary.getProgramInputs(programIndex);
  PhaseCtx prefillCtx(fs::path(artifacts), "prefill");
  PhaseCtx decodeCtx(fs::path(artifacts), "decode");
  if (prefillInputs.size() != prefillCtx.roles.size())
    throw std::runtime_error("llama31_tt: prefill input count mismatch");
  if (decodeInputs.size() != decodeCtx.roles.size())
    throw std::runtime_error("llama31_tt: decode input count mismatch");

  std::unordered_map<std::string, uint32_t> decodeKVInputSlots;
  std::vector<std::string> decodeKVOutputRoles;
  for (uint32_t slot = 0; slot < decodeCtx.roles.size(); ++slot) {
    const std::string &role = decodeCtx.roles[slot].role;
    if (startsWith(role, "past_K_") || startsWith(role, "past_V_")) {
      decodeKVInputSlots[role] = slot;
      decodeKVOutputRoles.push_back(role);
    }
  }
  if (decodeKVOutputRoles.empty())
    throw std::runtime_error("llama31_tt: decode artifacts have no KV slots");
  const std::vector<std::string> prefillKVOutputRoles =
      buildPrefillKVOutputRoles(decodeKVOutputRoles, prefillKVOutputOrder);

  ::tt::runtime::MeshDeviceOptions options;
  options.meshShape = std::vector<uint32_t>{1, 1};
  std::optional<::tt::runtime::Device> device;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> prefillCache;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> decodeCache;
  buddy::Sampler sampler(cfg.samplerConfig);

  try {
    device = ::tt::runtime::openMeshDevice(options);
    if (!suppress) {
      std::cout << kAnsiYellowBold
                << modelName
                << " Inference Powered by Buddy Compiler (Tenstorrent TTIR, "
                   "native C++)"
                << kAnsiReset << "\n";
    }

    auto cacheStart = std::chrono::steady_clock::now();
    StaticInputCacheResult prefillStatic = precacheStaticInputs(
        *device, prefillBinary, programIndex, prefillCtx, prefillInputs,
        nullptr, nullptr, nullptr, "prefill", suppress);
    prefillCache = std::move(prefillStatic.tensors);
    const bool disableStaticReuse =
        lookupBoolAttr(manifest, "disable_static_reuse", false) ||
        std::getenv("BUDDY_LLAMA31_DISABLE_STATIC_REUSE") != nullptr;
    StaticInputCacheResult decodeStatic =
        disableStaticReuse
            ? precacheStaticInputs(*device, decodeBinary, programIndex,
                                   decodeCtx, decodeInputs, nullptr, nullptr,
                                   nullptr, "decode", suppress)
            : precacheStaticInputs(*device, decodeBinary, programIndex,
                                   decodeCtx, decodeInputs, &prefillCtx,
                                   &prefillInputs, &prefillCache, "decode",
                                   suppress);
    decodeCache = std::move(decodeStatic.tensors);
    auto cacheEnd = std::chrono::steady_clock::now();
    printLlamaLog(
        "static tensors ready: uploaded " +
            std::to_string(prefillStatic.uploaded) + " prefill + " +
            std::to_string(decodeStatic.uploaded) + " decode, reused " +
            std::to_string(decodeStatic.reused) + " decode in " +
            formatFixed(
                std::chrono::duration<double>(cacheEnd - cacheStart).count()) +
            "s",
        suppress);
    const bool extractDecodeKVOutputs =
        std::getenv("BUDDY_LLAMA31_EXTRACT_DECODE_KV_OUTPUTS") != nullptr;
    printLlamaLog(std::string("decode KV output handling: ") +
                      (extractDecodeKVOutputs
                           ? "extract returned cache handles"
                           : "reuse persistent input cache handles"),
                  suppress);

    std::vector<std::string> promptBatch = cfg.prompts;
    if (promptBatch.empty())
      promptBatch.push_back(prompt);
    if (promptBatch.size() != 1 &&
        promptBatch.size() != static_cast<size_t>(batchSize))
      throw std::runtime_error(
          "llama31_tt: --prompt-file must contain either 1 prompt or exactly "
          "batch_size prompts");
    if (promptBatch.size() == 1 && batchSize > 1)
      promptBatch.assign(static_cast<size_t>(batchSize), promptBatch.front());

    std::unique_ptr<buddy::ChatTemplate> chatTemplate;
    if (!officialRef && !cfg.chatTemplatePath.empty()) {
      chatTemplate = std::make_unique<buddy::ChatTemplate>(
          buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
      for (int token : chatTemplate->stopTokenIds())
        addStopToken(token);
      for (const std::string &tokenText : chatTemplate->stopTokens()) {
        if (auto token = tokenizer.specialTokenId(tokenText))
          addStopToken(*token);
      }
      printLlamaLog("chat template loaded: " + cfg.chatTemplatePath, suppress);
    }

    auto encodePrompt = [&](const std::string &text) {
      if (officialRef)
        return officialRef->promptTokens;
      if (chatTemplate)
        return tokenizer.encodeChatTemplate(*chatTemplate, text);
      if (promptFormat == "completion")
        return tokenizer.encodeCompletion(text);
      return tokenizer.encodeChat(text);
    };

    std::vector<std::vector<int>> tokenIdsBatch;
    tokenIdsBatch.reserve(static_cast<size_t>(batchSize));
    size_t minPromptTokens = std::numeric_limits<size_t>::max();
    size_t maxPromptTokens = 0;
    for (int b = 0; b < batchSize; ++b) {
      std::vector<int> ids = encodePrompt(promptBatch[static_cast<size_t>(b)]);
      minPromptTokens = std::min(minPromptTokens, ids.size());
      maxPromptTokens = std::max(maxPromptTokens, ids.size());
      tokenIdsBatch.push_back(std::move(ids));
    }
    if (tokenIdsBatch.empty())
      throw std::runtime_error("llama31_tt: no prompts to run");

    const int prefillPromptLen =
        cfg.promptLength > 0 ? cfg.promptLength : static_cast<int>(maxPromptTokens);
    if (prefillPromptLen <= 0)
      throw std::runtime_error("llama31_tt: prompt length must be positive");
    if (prefillPromptLen >= maxCacheLen)
      throw std::runtime_error("llama31_tt: prompt is longer than max cache");
    if (maxPromptTokens > static_cast<size_t>(prefillPromptLen))
      throw std::runtime_error(
          "llama31_tt: at least one prompt is longer than --prompt-length");

    std::vector<int> tokenIds = tokenIdsBatch.front();
    printLlamaLog("prompt tokens = " + std::to_string(minPromptTokens) +
                      (minPromptTokens == maxPromptTokens
                           ? ""
                           : ".." + std::to_string(maxPromptTokens)) +
                      ", prefill length = " +
                      std::to_string(prefillPromptLen) + ", batch = " +
                      std::to_string(batchSize),
                  suppress);
    std::vector<int> recentTokens = tokenIds;
    std::vector<std::vector<int>> recentTokensBatch = tokenIdsBatch;

    const int padToken =
        officialRef ? 0
                    : (!cfg.prompts.empty() && promptFormat == "completion"
                           ? kEndOfText
                           : eosToken);
    const bool leftPadPrompts = !cfg.prompts.empty();
    std::vector<std::vector<int>> paddedBatch;
    std::vector<std::vector<int>> attentionMaskBatch;
    paddedBatch.reserve(static_cast<size_t>(batchSize));
    attentionMaskBatch.reserve(static_cast<size_t>(batchSize));
    for (int b = 0; b < batchSize; ++b) {
      const std::vector<int> &ids = tokenIdsBatch[static_cast<size_t>(b)];
      std::vector<int> padded(maxCacheLen, padToken);
      std::vector<int> attentionMask(maxCacheLen, 1);
      const size_t start =
          leftPadPrompts ? static_cast<size_t>(prefillPromptLen) - ids.size()
                         : 0;
      if (leftPadPrompts)
        std::fill(attentionMask.begin(), attentionMask.begin() + start, 0);
      std::copy(ids.begin(), ids.end(), padded.begin() + start);
      paddedBatch.push_back(std::move(padded));
      attentionMaskBatch.push_back(std::move(attentionMask));
    }
    std::vector<uint32_t> prefillTokenPositions(
        static_cast<size_t>(batchSize),
        static_cast<uint32_t>(prefillPromptLen - 1));

    auto prefillInputsRT =
        buildPrefillInputs(*device, prefillBinary, programIndex, prefillCtx,
                           prefillInputs, prefillCache, paddedBatch,
                           attentionMaskBatch);
    auto prefillStart = std::chrono::steady_clock::now();
    auto prefillOutputs = ::tt::runtime::submit(*device, prefillBinary,
                                                programIndex, prefillInputsRT);
    ::tt::runtime::wait(prefillOutputs);
    auto prefillEnd = std::chrono::steady_clock::now();
    const double prefillSeconds =
        std::chrono::duration<double>(prefillEnd - prefillStart).count();

    ExtractedOutput prefillExtracted = extractLogitsAndKV(
        prefillOutputs, -1, static_cast<uint32_t>(prefillPromptLen - 1), *device,
        decodeBinary, programIndex, decodeKVInputSlots, prefillKVOutputRoles,
        sampler, recentTokens, true, static_cast<uint32_t>(batchSize),
        &prefillTokenPositions, &recentTokensBatch);

    std::vector<int> generated;
    generated.push_back(prefillExtracted.token);
    recentTokens.push_back(prefillExtracted.token);
    std::vector<std::vector<int>> generatedBatch(static_cast<size_t>(batchSize));
    for (int b = 0; b < batchSize; ++b) {
      const int token =
          b < static_cast<int>(prefillExtracted.tokens.size())
              ? prefillExtracted.tokens[static_cast<size_t>(b)]
              : prefillExtracted.token;
      generatedBatch[static_cast<size_t>(b)].push_back(token);
      recentTokensBatch[static_cast<size_t>(b)].push_back(token);
    }
    std::unordered_map<std::string, ::tt::runtime::Tensor> pastKV =
        std::move(prefillExtracted.kvDevice);
    retainKVCache(pastKV);
    DecodeRuntimeInputCache decodeRuntimeInputCache =
        precacheDecodeRuntimeInputs(*device, decodeBinary, programIndex,
                                    decodeCtx, decodeInputs, decodeCache,
                                    attentionMaskBatch, prefillPromptLen,
                                    maxCacheLen, suppress);

    if (officialRef) {
      const int maxPredictions =
          std::min({officialRef->maxPredictions,
                    static_cast<int>(officialRef->forcedTokens.size()),
                    static_cast<int>(officialRef->top5Tokens.size()),
                    maxCacheLen - prefillPromptLen});
      if (maxPredictions <= 0)
        throw std::runtime_error(
            "llama31_tt: official token matching has no decode budget");

      std::vector<int> predicted;
      predicted.reserve(static_cast<size_t>(maxPredictions));
      int currentPrediction = prefillExtracted.token;
      int cachePos = prefillPromptLen;
      DecodeTiming decodeTiming;
      std::vector<::tt::runtime::Tensor> previousDecodeOutputs;

      for (int iter = 0; iter < maxPredictions && cachePos < maxCacheLen;
           ++iter) {
        predicted.push_back(currentPrediction);
        const int inputToken =
            officialRef->forcedTokens[static_cast<size_t>(std::min(
                iter, static_cast<int>(officialRef->forcedTokens.size()) - 1))];

      auto iterStart = std::chrono::steady_clock::now();
      debugKVCacheState(pastKV, "official iter " + std::to_string(iter));
      debugTensorState(previousDecodeOutputs,
                       "official prev outputs " + std::to_string(iter));
      auto decodeInputsRT = buildDecodeInputs(
          *device, decodeBinary, programIndex, decodeCtx, decodeInputs,
          decodeCache, std::vector<int>{inputToken}, cachePos, pastKV,
          attentionMaskBatch, &decodeRuntimeInputCache);
        debugTensorVectorState(decodeInputsRT, decodeCtx.roles,
                               "official iter " + std::to_string(iter));
        auto buildEnd = std::chrono::steady_clock::now();
        auto submitStart = buildEnd;
        auto decodeOutputs = ::tt::runtime::submit(
            *device, decodeBinary, programIndex, decodeInputsRT);
        waitDecodeOutputs(decodeOutputs, -1);
        auto submitEnd = std::chrono::steady_clock::now();

        ExtractedOutput decoded = extractLogitsAndKV(
            decodeOutputs, -1, 0, *device, decodeBinary, programIndex,
            decodeKVInputSlots, decodeKVOutputRoles, sampler, recentTokens,
            extractDecodeKVOutputs);
        previousDecodeOutputs = std::move(decodeOutputs);
        auto outputEnd = std::chrono::steady_clock::now();
        if (extractDecodeKVOutputs) {
          // Decode uses in-place TTNN paged_update_cache. These returned KV
          // tensors alias the persistent input cache handles; only use them
          // when explicitly requested for A/B isolation.
          pastKV = std::move(decoded.kvDevice);
          retainKVCache(pastKV);
        }

        currentPrediction = decoded.token;
        recentTokens.push_back(inputToken);
        ++cachePos;

        auto iterEnd = std::chrono::steady_clock::now();
        const double buildSeconds =
            std::chrono::duration<double>(buildEnd - iterStart).count();
        const double submitSeconds =
            std::chrono::duration<double>(submitEnd - submitStart).count();
        const double outputSeconds =
            std::chrono::duration<double>(outputEnd - submitEnd).count();
        const double iterSeconds =
            std::chrono::duration<double>(iterEnd - iterStart).count();
        const double bookkeepingSeconds =
            iterSeconds - buildSeconds - submitSeconds - outputSeconds;
        decodeTiming.add(iterSeconds, buildSeconds, submitSeconds,
                         outputSeconds, decoded.logitsToHostSeconds,
                         decoded.kvRelayoutSeconds, bookkeepingSeconds);

        if (!suppress &&
            (iter < 8 || (iter + 1) % 50 == 0 || iter + 1 == maxPredictions)) {
          std::cout << colorLabel("[Official " + std::to_string(iter) + "]")
                    << " predicted=" << predicted.back()
                    << " forced_in=" << inputToken
                    << " | Decode Time: " << formatFixed(iterSeconds) << "s\n";
        }
      }

      for (auto &kv : pastKV) {
        try {
          ::tt::runtime::deallocateTensor(kv.second, true);
        } catch (...) {
        }
      }
      previousDecodeOutputs.clear();
      deallocateDecodeRuntimeInputCache(decodeRuntimeInputCache);

      const auto [top1, top5] =
          computeOfficialAccuracy(predicted, officialRef->top5Tokens);
      const double steadyWall = decodeTiming.steadyWallSeconds();
      const double steadySubmit = decodeTiming.steadySubmitSeconds();
      const int steadyCount =
          decodeTiming.count > 1 ? decodeTiming.count - 1 : 0;
      auto perSteadyTokenMs = [steadyCount](double seconds) {
        return steadyCount > 0
                   ? 1000.0 * seconds / static_cast<double>(steadyCount)
                   : 0.0;
      };

      if (!suppress) {
        std::cout << "\n"
                  << colorLabel("[Official token matching]") << " "
                  << predicted.size() << " predictions"
                  << " against " << officialRef->sourcePath << "\n";
        std::cout << colorLabel("[Top1]") << " " << formatFixed(top1, 2)
                  << "%\n";
        std::cout << colorLabel("[Top5]") << " " << formatFixed(top5, 2)
                  << "%\n";
        std::cout << colorLabel("[Prefill]") << " "
                  << formatFixed(prefillSeconds) << "s\n";
        std::cout << colorLabel("[Decode wall]") << " "
                  << formatFixed(decodeTiming.wallSeconds > 0.0
                                     ? decodeTiming.count /
                                           decodeTiming.wallSeconds
                                     : 0.0)
                  << " tokens/s (" << decodeTiming.count << " tokens in "
                  << formatFixed(decodeTiming.wallSeconds) << "s)\n";
        std::cout << colorLabel("[Decode steady wall]") << " "
                  << formatFixed(steadyWall > 0.0
                                     ? steadyCount / steadyWall
                                     : 0.0)
                  << " tokens/s excluding first decode\n";
        std::cout << colorLabel("[Decode device-only]") << " "
                  << formatFixed(decodeTiming.submitSeconds > 0.0
                                     ? decodeTiming.count /
                                           decodeTiming.submitSeconds
                                     : 0.0)
                  << " tokens/s\n";
        std::cout << colorLabel("[Decode steady device-only]") << " "
                  << formatFixed(steadySubmit > 0.0
                                     ? steadyCount / steadySubmit
                                     : 0.0)
                  << " tokens/s excluding first decode\n";
        std::cout << colorLabel("[Decode steady breakdown]") << " "
                  << "build=" << formatFixed(perSteadyTokenMs(
                                     decodeTiming.steadyBuildInputsSeconds()))
                  << "ms, submit+wait="
                  << formatFixed(perSteadyTokenMs(steadySubmit))
                  << "ms, output="
                  << formatFixed(perSteadyTokenMs(
                         decodeTiming.steadyOutputSeconds()))
                  << "ms, logits_to_host="
                  << formatFixed(perSteadyTokenMs(
                         decodeTiming.steadyLogitsToHostSeconds()))
                  << "ms, kv_relayout="
                  << formatFixed(perSteadyTokenMs(
                         decodeTiming.steadyKVRelayoutSeconds()))
                  << "ms, bookkeeping="
                  << formatFixed(perSteadyTokenMs(
                         decodeTiming.steadyBookkeepingSeconds()))
                  << "ms\n";
      }

      writeOfficialTrace(*officialRef, predicted,
                         prefillPromptLen, prefillSeconds,
                         decodeTiming, extractDecodeKVOutputs, top1, top5);
      ::tt::runtime::closeMeshDevice(*device);
      return;
    }

    std::string streamedText;
    auto streamGeneratedText = [&]() {
      if (!suppress)
        return;
      std::string current = tokenizer.decodeTokens(generated, true);
      if (current.size() > streamedText.size()) {
        std::cout.write(current.data() + streamedText.size(),
                        current.size() - streamedText.size());
        std::cout.flush();
      }
      streamedText = std::move(current);
    };

    if (!suppress) {
      std::cout << colorLabel("[Iteration 0]")
                << (batchSize > 1 ? " Token[0]: " : " Token: ")
                << tokenizer.decodeToken(prefillExtracted.token)
                << " | Time: " << formatFixed(prefillSeconds) << "s\n";
    } else {
      streamGeneratedText();
    }

    const int maxNewTokens =
        cfg.maxNewTokens > 0 ? cfg.maxNewTokens : maxCacheLen;
    int cachePos = prefillPromptLen;
    DecodeTiming decodeTiming;
    std::vector<bool> stoppedBatch(static_cast<size_t>(batchSize), false);
    std::vector<::tt::runtime::Tensor> previousDecodeOutputs;
    std::optional<::tt::runtime::Tensor> deviceInputIdsForNext;
    const bool useDeviceTokenChain =
        std::getenv("BUDDY_LLAMA31_DISABLE_DEVICE_TOKEN_CHAIN") == nullptr;
    const bool requestedDeferredDecodeTokenReadback =
        cfg.deferDecodeTokenReadback ||
        std::getenv("BUDDY_LLAMA31_DEFER_DECODE_TOKEN_READBACK") != nullptr;
    const bool greedySamplerConfig = cfg.samplerConfig.temperature == 0.0f &&
                                     cfg.samplerConfig.topK == 0 &&
                                     cfg.samplerConfig.topP == 1.0f &&
                                     cfg.samplerConfig.minP == 0.0f &&
                                     cfg.samplerConfig.repeatPenalty == 1.0f;
    const bool deferDecodeTokenReadback =
        requestedDeferredDecodeTokenReadback && ignoreEOS &&
        useDeviceTokenChain && !extractDecodeKVOutputs && greedySamplerConfig;
    if (requestedDeferredDecodeTokenReadback && !deferDecodeTokenReadback) {
      printLlamaLog(
          "deferred decode token readback disabled: requires ignore_eos, "
          "greedy sampling, device token chain, and persistent decode KV "
          "cache reuse",
          suppress);
    } else if (deferDecodeTokenReadback) {
      printLlamaLog(
          "deferred decode token readback enabled: token tensors are read "
          "after the decode loop",
          suppress);
    }
    std::vector<::tt::runtime::Tensor> deferredDecodeTokenOutputs;
    for (int b = 0; b < batchSize; ++b)
      stoppedBatch[static_cast<size_t>(b)] =
          isStopToken(generatedBatch[static_cast<size_t>(b)].back());
    auto allStopped = [&]() {
      return std::all_of(stoppedBatch.begin(), stoppedBatch.end(),
                         [](bool stopped) { return stopped; });
    };
    int generatedTokenCount = static_cast<int>(generatedBatch.front().size());
    while (generatedTokenCount < maxNewTokens &&
           cachePos < maxCacheLen && !allStopped()) {
      std::vector<int> inputTokens(static_cast<size_t>(batchSize), eosToken);
      for (int b = 0; b < batchSize; ++b) {
        if (!stoppedBatch[static_cast<size_t>(b)])
          inputTokens[static_cast<size_t>(b)] =
              generatedBatch[static_cast<size_t>(b)].back();
      }
      const bool anyStopped =
          std::any_of(stoppedBatch.begin(), stoppedBatch.end(),
                      [](bool stopped) { return stopped; });
      if (deviceInputIdsForNext && anyStopped) {
        try {
          ::tt::runtime::deallocateTensor(*deviceInputIdsForNext, true);
        } catch (...) {
        }
        deviceInputIdsForNext.reset();
      }
      auto iterStart = std::chrono::steady_clock::now();
      debugKVCacheState(pastKV, "decode iter " + std::to_string(cachePos));
      debugTensorState(previousDecodeOutputs,
                       "decode prev outputs " + std::to_string(cachePos));
      auto decodeInputsRT = buildDecodeInputs(
          *device, decodeBinary, programIndex, decodeCtx, decodeInputs,
          decodeCache, inputTokens, cachePos, pastKV, attentionMaskBatch,
          &decodeRuntimeInputCache,
          deviceInputIdsForNext ? &*deviceInputIdsForNext : nullptr);
      debugTensorVectorState(decodeInputsRT, decodeCtx.roles,
                             "decode iter " + std::to_string(cachePos));
      auto buildEnd = std::chrono::steady_clock::now();
      auto submitStart = buildEnd;
      auto decodeOutputs = ::tt::runtime::submit(*device, decodeBinary,
                                                 programIndex, decodeInputsRT);
      waitDecodeOutputs(decodeOutputs, -1);
      auto submitEnd = std::chrono::steady_clock::now();
      if (deviceInputIdsForNext) {
        if (!deferDecodeTokenReadback) {
          try {
            ::tt::runtime::deallocateTensor(*deviceInputIdsForNext, true);
          } catch (...) {
          }
        }
        deviceInputIdsForNext.reset();
      }

      ExtractedOutput decoded = extractLogitsAndKV(
          decodeOutputs, -1, 0, *device, decodeBinary, programIndex,
          decodeKVInputSlots, decodeKVOutputRoles, sampler, recentTokens,
          extractDecodeKVOutputs, static_cast<uint32_t>(batchSize), nullptr,
          &recentTokensBatch, useDeviceTokenChain, deferDecodeTokenReadback);
      previousDecodeOutputs = std::move(decodeOutputs);
      auto outputEnd = std::chrono::steady_clock::now();
      deviceInputIdsForNext = std::move(decoded.tokenDevice);
      if (extractDecodeKVOutputs) {
        // Decode uses in-place TTNN paged_update_cache. These returned KV
        // tensors alias the persistent input cache handles; only use them when
        // explicitly requested for A/B isolation.
        pastKV = std::move(decoded.kvDevice);
        retainKVCache(pastKV);
      }

      std::optional<int> user0TokenForIterationLog;
      if (deferDecodeTokenReadback) {
        if (!deviceInputIdsForNext)
          throw std::runtime_error(
              "llama31_tt: deferred decode token readback lost token tensor");
        deferredDecodeTokenOutputs.push_back(*deviceInputIdsForNext);
      } else {
        const bool user0WasStopped =
            stoppedBatch.empty() ? false : stoppedBatch.front();
        const int user0Token = !decoded.tokens.empty() ? decoded.tokens.front()
                                                       : decoded.token;
        for (int b = 0; b < batchSize; ++b) {
          const size_t batchIndex = static_cast<size_t>(b);
          if (stoppedBatch[batchIndex])
            continue;
          const int token =
              b < static_cast<int>(decoded.tokens.size())
                  ? decoded.tokens[batchIndex]
                  : decoded.token;
          generatedBatch[batchIndex].push_back(token);
          recentTokensBatch[batchIndex].push_back(token);
          if (isStopToken(token))
            stoppedBatch[batchIndex] = true;
        }
        if (!user0WasStopped) {
          generated.push_back(user0Token);
          recentTokens.push_back(user0Token);
          user0TokenForIterationLog = user0Token;
        }
      }
      ++generatedTokenCount;
      ++cachePos;
      auto iterEnd = std::chrono::steady_clock::now();
      const double buildSeconds =
          std::chrono::duration<double>(buildEnd - iterStart).count();
      const double submitSeconds =
          std::chrono::duration<double>(submitEnd - submitStart).count();
      const double outputSeconds =
          std::chrono::duration<double>(outputEnd - submitEnd).count();
      const double iterSeconds =
          std::chrono::duration<double>(iterEnd - iterStart).count();
      const double bookkeepingSeconds =
          iterSeconds - buildSeconds - submitSeconds - outputSeconds;
      decodeTiming.add(iterSeconds, buildSeconds, submitSeconds, outputSeconds,
                       decoded.logitsToHostSeconds, decoded.kvRelayoutSeconds,
                       bookkeepingSeconds);

      if (!suppress) {
        std::cout << colorLabel("[Iteration " +
                                std::to_string(decodeTiming.count) + "]")
                  << (batchSize > 1 ? " Token[0]: " : " Token: ")
                  << (deferDecodeTokenReadback
                          ? std::string("<deferred>")
                          : user0TokenForIterationLog
                                ? tokenizer.decodeToken(
                                      *user0TokenForIterationLog)
                                : std::string("<stopped>"))
                  << " | Time: " << formatFixed(iterSeconds) << "s\n";
      } else {
        if (!deferDecodeTokenReadback)
          streamGeneratedText();
      }
    }
    if (deviceInputIdsForNext) {
      if (!deferDecodeTokenReadback) {
        try {
          ::tt::runtime::deallocateTensor(*deviceInputIdsForNext, true);
        } catch (...) {
        }
      }
      deviceInputIdsForNext.reset();
    }

    double deferredTokenReadbackSeconds = 0.0;
    if (deferDecodeTokenReadback) {
      auto readbackStart = std::chrono::steady_clock::now();
      for (auto &tokenTensor : deferredDecodeTokenOutputs) {
        std::vector<int> tokens =
            readTokenTensorToHost(tokenTensor, static_cast<uint32_t>(batchSize));
        for (int b = 0; b < batchSize; ++b) {
          const int token =
              b < static_cast<int>(tokens.size()) ? tokens[static_cast<size_t>(b)]
                                                  : tokens.front();
          generatedBatch[static_cast<size_t>(b)].push_back(token);
          recentTokensBatch[static_cast<size_t>(b)].push_back(token);
        }
        generated.push_back(tokens.front());
        recentTokens.push_back(tokens.front());
      }
      auto readbackEnd = std::chrono::steady_clock::now();
      deferredTokenReadbackSeconds =
          std::chrono::duration<double>(readbackEnd - readbackStart).count();
      for (auto &tokenTensor : deferredDecodeTokenOutputs) {
        try {
          ::tt::runtime::deallocateTensor(tokenTensor, true);
        } catch (...) {
        }
      }
      deferredDecodeTokenOutputs.clear();
      printLlamaLog("deferred decode token readback finished in " +
                        formatFixed(deferredTokenReadbackSeconds) + "s",
                    suppress);
    }

    for (auto &kv : pastKV) {
      try {
        ::tt::runtime::deallocateTensor(kv.second, true);
      } catch (...) {
      }
    }
    previousDecodeOutputs.clear();
    deallocateDecodeRuntimeInputCache(decodeRuntimeInputCache);

    std::vector<std::string> generatedTextBatch;
    generatedTextBatch.reserve(static_cast<size_t>(batchSize));
    for (const auto &tokens : generatedBatch)
      generatedTextBatch.push_back(tokenizer.decodeTokens(tokens, true));
    if (const char *traceOut = std::getenv("BUDDY_LLAMA31_BATCH_TRACE_OUT")) {
      writeBatchTrace(traceOut, modelName, promptBatch, tokenIdsBatch,
                      prefillPromptLen, maxCacheLen, leftPadPrompts,
                      prefillSeconds, decodeTiming,
                      deferredTokenReadbackSeconds, generatedBatch,
                      generatedTextBatch);
      printLlamaLog(std::string("batch trace written: ") + traceOut, suppress);
    }

    if (suppress) {
      if (deferDecodeTokenReadback && !generatedTextBatch.empty()) {
        std::cout << generatedTextBatch.front();
      }
      std::cout << "\n";
    } else {
      const double total = prefillSeconds + decodeTiming.wallSeconds +
                           deferredTokenReadbackSeconds;
      const double aggregateDecodeTokens =
          static_cast<double>(decodeTiming.count) * batchSize;
      const int steadyCount =
          decodeTiming.count > 1 ? decodeTiming.count - 1 : 0;
      const double aggregateSteadyDecodeTokens =
          static_cast<double>(steadyCount) * batchSize;
      const double steadyWall = decodeTiming.steadyWallSeconds();
      const double steadySubmit = decodeTiming.steadySubmitSeconds();
      auto perSteadyTokenMs = [steadyCount](double seconds) {
        return steadyCount > 0
                   ? 1000.0 * seconds / static_cast<double>(steadyCount)
                   : 0.0;
      };
      std::cout << "\n"
                << colorLabel("[Total time]") << " " << formatFixed(total)
                << "s\n";
      std::cout << colorLabel("[Batch]") << " " << batchSize << "\n";
      std::cout << colorLabel("[Prefilling]") << " "
                << formatFixed(prefillSeconds > 0.0
                                   ? static_cast<double>(prefillPromptLen) *
                                         batchSize /
                                         prefillSeconds
                                   : 0.0)
                << " aggregate tokens/s (prefill_time="
                << formatFixed(prefillSeconds)
                << "s)\n";
      std::cout << colorLabel("[Decoding wall]") << " "
                << formatFixed(decodeTiming.wallSeconds > 0.0
                                   ? aggregateDecodeTokens /
                                         decodeTiming.wallSeconds
                                   : 0.0)
                << " aggregate tokens/s (" << aggregateDecodeTokens
                << " tokens in " << formatFixed(decodeTiming.wallSeconds)
                << "s, "
                << formatFixed(decodeTiming.wallSeconds > 0.0
                                   ? decodeTiming.count /
                                         decodeTiming.wallSeconds
                                   : 0.0)
                << " tok/s/user)\n";
      if (deferDecodeTokenReadback) {
        const double decodeWallPlusReadback =
            decodeTiming.wallSeconds + deferredTokenReadbackSeconds;
        const double steadyWallPlusReadback =
            steadyWall + deferredTokenReadbackSeconds;
        std::cout << colorLabel("[Decoding wall incl deferred readback]") << " "
                  << formatFixed(decodeWallPlusReadback > 0.0
                                     ? aggregateDecodeTokens /
                                           decodeWallPlusReadback
                                     : 0.0)
                  << " aggregate tokens/s (" << aggregateDecodeTokens
                  << " tokens in " << formatFixed(decodeWallPlusReadback)
                  << "s, "
                  << formatFixed(decodeWallPlusReadback > 0.0
                                     ? decodeTiming.count /
                                           decodeWallPlusReadback
                                     : 0.0)
                  << " tok/s/user)\n";
        std::cout
            << colorLabel("[Decoding steady wall incl deferred readback]")
            << " "
            << formatFixed(steadyWallPlusReadback > 0.0
                               ? aggregateSteadyDecodeTokens /
                                     steadyWallPlusReadback
                               : 0.0)
            << " aggregate tokens/s ("
            << formatFixed(steadyWallPlusReadback > 0.0
                               ? steadyCount / steadyWallPlusReadback
                               : 0.0)
            << " tok/s/user, excluding first decode)\n";
      }
      std::cout << colorLabel("[Decoding steady wall]") << " "
                << formatFixed(steadyWall > 0.0
                                   ? aggregateSteadyDecodeTokens / steadyWall
                                   : 0.0)
                << " aggregate tokens/s ("
                << formatFixed(steadyWall > 0.0 ? steadyCount / steadyWall
                                                : 0.0)
                << " tok/s/user, excluding first decode)\n";
      std::cout << colorLabel("[Decoding device-only]") << " "
                << formatFixed(decodeTiming.submitSeconds > 0.0
                                   ? aggregateDecodeTokens /
                                         decodeTiming.submitSeconds
                                   : 0.0)
                << " aggregate tokens/s ("
                << formatFixed(decodeTiming.submitSeconds > 0.0
                                   ? decodeTiming.count /
                                         decodeTiming.submitSeconds
                                   : 0.0)
                << " tok/s/user)\n";
      std::cout << colorLabel("[Decoding steady device-only]") << " "
                << formatFixed(steadySubmit > 0.0
                                   ? aggregateSteadyDecodeTokens /
                                         steadySubmit
                                   : 0.0)
                << " aggregate tokens/s ("
                << formatFixed(steadySubmit > 0.0
                                   ? steadyCount / steadySubmit
                                   : 0.0)
                << " tok/s/user, excluding first decode)\n";
      std::cout << colorLabel("[Decoding steady breakdown]") << " "
                << "build="
                << formatFixed(
                       perSteadyTokenMs(decodeTiming
                                            .steadyBuildInputsSeconds()))
                << "ms, submit+wait="
                << formatFixed(perSteadyTokenMs(steadySubmit))
                << "ms, output="
                << formatFixed(
                       perSteadyTokenMs(decodeTiming.steadyOutputSeconds()))
                << "ms, logits_to_host="
                << formatFixed(perSteadyTokenMs(
                       decodeTiming.steadyLogitsToHostSeconds()))
                << "ms, kv_relayout="
                << formatFixed(perSteadyTokenMs(
                       decodeTiming.steadyKVRelayoutSeconds()))
                << "ms, bookkeeping="
                << formatFixed(perSteadyTokenMs(
                       decodeTiming.steadyBookkeepingSeconds()))
                << "ms\n";
      if (deferDecodeTokenReadback) {
        std::cout << colorLabel("[Deferred token readback]") << " "
                  << formatFixed(deferredTokenReadbackSeconds) << "s\n";
      }
      const bool printAllBatchOutputs = batchSize > 1 &&
          (cfg.printAllBatchOutputs ||
           std::getenv("BUDDY_LLAMA31_PRINT_ALL_BATCH") != nullptr);
      const int printedBatch =
          printAllBatchOutputs ? batchSize : std::min(batchSize, 1);
      for (int b = 0; b < printedBatch; ++b) {
        const std::string suffix =
            batchSize > 1 ? " user" + std::to_string(b) : "";
        std::cout << colorLabel("[Input" + suffix + "]") << " "
                  << promptBatch[static_cast<size_t>(b)] << "\n";
        std::cout << colorLabel("[Output" + suffix + "]") << " "
                  << (b < static_cast<int>(generatedTextBatch.size())
                          ? generatedTextBatch[static_cast<size_t>(b)]
                          : "")
                  << "\n";
      }
    }

    ::tt::runtime::closeMeshDevice(*device);
  } catch (...) {
    if (device) {
      try {
        ::tt::runtime::closeMeshDevice(*device);
      } catch (...) {
      }
    }
    throw;
  }
}

} // namespace runtime
} // namespace buddy

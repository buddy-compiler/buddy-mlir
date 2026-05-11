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

#include "buddy/runtime/core/ModelManifest.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace buddy {
namespace runtime {
namespace {

namespace fs = std::filesystem;

static std::string shellQuote(const std::string &value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

static std::string getEnvOr(const char *name, const std::string &fallback) {
  if (const char *value = std::getenv(name))
    if (value[0] != '\0')
      return value;
  return fallback;
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

static std::string defaultRunnerPath(const std::string &raxPath);

static std::string findRunnerArtifact(const ModelManifest &manifest,
                                      const std::string &raxPath) {
  for (const auto &codeObject : manifest.codeObjects) {
    if (codeObject.backend == "python" ||
        codeObject.name.find("runner") != std::string::npos) {
      if (!codeObject.path.empty())
        return codeObject.path;
    }
  }
  return lookupAttr(manifest, "runner_uri", defaultRunnerPath(raxPath));
}

static std::string defaultRunnerPath(const std::string &raxPath) {
  if (!raxPath.empty()) {
    fs::path besideRax =
        fs::path(raxPath).parent_path() / "llama31_chat_run.py";
    if (fs::exists(besideRax))
      return fs::absolute(besideRax).string();
  }
  if (const char *env = std::getenv("BUDDY_LLAMA31_TT_RUNNER"))
    if (env[0] != '\0')
      return env;
  return "llama31_chat_run.py";
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

  if (rest == "weights_npz")
    filename = "weights.npz";
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

} // namespace

void Llama31TTRunner::run(const RunConfig &cfg) {
  raiseMemoryLimitForLlama();

  if (cfg.raxPath.empty())
    throw std::runtime_error("llama31_tt requires --model <path.rax>");

  const ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
  const std::string prefillTTNN = findTTNNArtifact(manifest, "prefill");
  const std::string decodeTTNN = findTTNNArtifact(manifest, "decode");

  const fs::path raxDir = fs::absolute(fs::path(cfg.raxPath)).parent_path();
  const std::string runner = findRunnerArtifact(manifest, cfg.raxPath);
  std::string artifacts = materializeEmbeddedArtifacts(manifest, raxDir);
  if (artifacts.empty()) {
    artifacts = lookupAttr(manifest, "artifacts_uri",
                           (raxDir / "chat_artifacts").string());
  }
  const std::string tokenizer = lookupAttr(
      manifest, "tokenizer_uri",
      getEnvOr("LLAMA31_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"));
  const std::string maxCacheLen = lookupAttr(manifest, "max_cache_len", "1024");
  const bool ignoreSystemDesc =
      lookupBoolAttr(manifest, "ignore_system_desc", true);
  const bool ignoreEOS = lookupBoolAttr(manifest, "ignore_eos", false);
  const bool deviceTokenLoop =
      lookupBoolAttr(manifest, "device_token_loop", false);
  const std::string officialReference =
      lookupAttr(manifest, "official_reference_uri");
  const std::string officialTrace = lookupAttr(manifest, "official_trace_uri");

  const std::string python = getEnvOr("BUDDY_TT_PYTHON", "python3");

  std::string command;
  if (!cfg.interactive)
    command += "printf '%s\\n' " + shellQuote(cfg.prompt) + " | ";

  command += shellQuote(python) + " " + shellQuote(runner);
  command += " --prefill-ttnn " + shellQuote(prefillTTNN);
  command += " --decode-ttnn " + shellQuote(decodeTTNN);
  command += " --artifacts " + shellQuote(artifacts);
  command += " --model " + shellQuote(tokenizer);
  command += " --max-cache-len " + shellQuote(maxCacheLen);
  if (cfg.maxNewTokens > 0)
    command += " --max-new-tokens " + std::to_string(cfg.maxNewTokens);
  if (cfg.interactive && !cfg.prompt.empty())
    command += " --system-prompt " + shellQuote(cfg.prompt);
  if (ignoreSystemDesc)
    command += " --ignore-system-desc";
  if (ignoreEOS)
    command += " --ignore-eos";
  if (deviceTokenLoop)
    command += " --device-token-loop";
  if (!officialReference.empty())
    command += " --official-reference-npz " + shellQuote(officialReference);
  if (!officialTrace.empty())
    command += " --official-trace-out " + shellQuote(officialTrace);

  std::cout << "[buddy-cli] dispatch llama31_tt via Tenstorrent runner\n";
  std::cout << "[buddy-cli] prefill: " << prefillTTNN << "\n";
  std::cout << "[buddy-cli] decode:  " << decodeTTNN << "\n";

  const int rc = std::system(command.c_str());
  if (rc != 0)
    throw std::runtime_error("llama31_tt runner failed with exit code " +
                             std::to_string(rc));
}

} // namespace runtime
} // namespace buddy

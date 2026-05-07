//===- Llama31TTRunner.cpp - Tenstorrent Llama 3.1 runner -----------------===//
//
// Licensed under the Apache License, Version 2.0.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Llama31TTRunner.h"

#include "buddy/runtime/core/ModelManifest.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

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

} // namespace

void Llama31TTRunner::run(const RunConfig &cfg) {
  if (cfg.raxPath.empty())
    throw std::runtime_error("llama31_tt requires --model <path.rax>");

  const ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
  const std::string prefillTTNN = findTTNNArtifact(manifest, "prefill");
  const std::string decodeTTNN = findTTNNArtifact(manifest, "decode");

  const fs::path raxDir = fs::absolute(fs::path(cfg.raxPath)).parent_path();
  const std::string runner =
      lookupAttr(manifest, "runner_uri", defaultRunnerPath(cfg.raxPath));
  const std::string artifacts = lookupAttr(
      manifest, "artifacts_uri", (raxDir / "chat_artifacts").string());
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

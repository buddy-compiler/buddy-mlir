//===- buddy-server.cpp - Buddy model HTTP server -------------------------===//
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
#include "ResidentModelFactory.h"
#include "ResidentModelPluginHandle.h"
#include "SimpleHttpServer.h"

#include "buddy/runtime/core/ResidentModel.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using buddy::runtime::ModelLoadState;
using buddy::runtime::ModelStatus;
using buddy::runtime::ResidentModel;
using buddy::runtime::ResidentModelConfig;
using buddy::runtime::TokenizeRequest;
using buddy::server::HttpRequest;
using buddy::server::ResponseWriter;
using buddy::server::SimpleHttpServer;

namespace {

enum ServerLoadState {
  Loading = 0,
  Ready = 1,
  Error = 2,
};

void usage(const char *prog, std::ostream &os = std::cout) {
  const auto backends = buddy::server::availableResidentModelTypes();

  os << "Usage: " << prog << " [options]\n"
     << "\n"
     << "Model source (one required):\n"
     << "  --model      <path.rax>  Model manifest (recommended)\n"
     << "  --model-so   <path.so>   Model shared library (legacy mode)\n"
     << "  --weights    <path>      Weights file; repeatable in legacy mode\n"
     << "  --vocab      <path>      Vocabulary file (legacy mode)\n"
     << "  --model-type <name>      Resident model backend (default "
        "deepseek_r1)\n"
     << "  --serving-so <path.so>   Resident model plugin shared library\n"
     << "\n"
     << "Server:\n"
     << "  --host       <addr>      Bind address (default 127.0.0.1)\n"
     << "  --port       <port>      Bind port (default 8080)\n"
     << "\n"
     << "Chat:\n"
     << "  --chat-template <path>   Path to chat template JSON config\n"
     << "\n"
     << "Other:\n"
     << "  --help / -h\n";

  os << "\n"
     << "Built-in resident backends:";
  if (backends.empty()) {
    os << " none\n";
  } else {
    for (const auto &backend : backends)
      os << " " << backend;
    os << "\n";
  }
}

bool hasSuffix(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool isReady(const std::atomic<int> &loadState) {
  return loadState.load(std::memory_order_acquire) == Ready;
}

void sendError(ResponseWriter &writer, int status, const std::string &message,
               const std::string &type) {
  writer.sendResponse(buddy::server::jsonResponse(
      status, buddy::server::errorJson(message, type, status)));
}

template <typename Fn> void withJsonErrors(ResponseWriter &writer, Fn fn) {
  try {
    fn();
  } catch (const std::exception &ex) {
    sendError(writer, 400, ex.what(), "bad_request");
  }
}

void handleCompletion(ResidentModel &model, const std::atomic<int> &loadState,
                      const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseCompletionRequest(request.body);
    if (!decoded.stream) {
      auto result = model.complete(decoded.request);
      writer.sendResponse(
          buddy::server::jsonResponse(200, buddy::server::toJson(result)));
      return;
    }

    writer.startSse();
    auto result = model.completeStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toCompletionChunkJson(chunk)))
            return false;
          if (chunk.done)
            return writer.writeSseData("[DONE]");
          return true;
        });
    (void)result;
  });
}

void handleChat(ResidentModel &model, const std::atomic<int> &loadState,
                const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseChatCompletionRequest(request.body);
    if (!decoded.stream) {
      auto result = model.chat(decoded.request);
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toOpenAIChatJson(result)));
      return;
    }

    writer.startSse();
    auto result = model.chatStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toOpenAIChatChunkJson(chunk)))
            return false;
          if (chunk.done)
            return writer.writeSseData("[DONE]");
          return true;
        });
    (void)result;
  });
}

void handleTokenize(ResidentModel &model, const std::atomic<int> &loadState,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    TokenizeRequest tokenRequest =
        buddy::server::parseTokenizeRequest(request.body);
    auto result = model.tokenize(tokenRequest);
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(result, tokenRequest.countOnly)));
  });
}

} // namespace

int main(int argc, char **argv) {
  ResidentModelConfig modelConfig;
  std::string modelType;
  std::string servingSoPath;
  std::string host = "127.0.0.1";
  int port = 8080;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc)
      modelConfig.raxPath = argv[++i];
    else if (arg == "--model-so" && i + 1 < argc)
      modelConfig.modelSoPath = argv[++i];
    else if (arg == "--weights" && i + 1 < argc)
      modelConfig.weightPaths.push_back(argv[++i]);
    else if (arg == "--vocab" && i + 1 < argc)
      modelConfig.vocabPath = argv[++i];
    else if (arg == "--model-type" && i + 1 < argc)
      modelType = argv[++i];
    else if (arg == "--serving-so" && i + 1 < argc)
      servingSoPath = argv[++i];
    else if (arg == "--chat-template" && i + 1 < argc)
      modelConfig.chatTemplatePath = argv[++i];
    else if (arg == "--host" && i + 1 < argc)
      host = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      port = std::stoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0], std::cerr);
      return 2;
    }
  }

  if (modelConfig.raxPath.empty() && modelConfig.modelSoPath.empty()) {
    std::cerr << "Provide --model <path.rax> or --model-so <path.so>.\n";
    usage(argv[0], std::cerr);
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() &&
      hasSuffix(modelConfig.modelSoPath, ".rax")) {
    std::cerr << "--model-so expects a shared library. For .rax manifests, use "
                 "--model <path.rax>.\n";
    usage(argv[0], std::cerr);
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() && modelConfig.weightPaths.empty()) {
    std::cerr << "--model-so legacy mode requires at least one --weights "
                 "<path> argument.\n";
    usage(argv[0], std::cerr);
    return 2;
  }

  std::unique_ptr<buddy::server::ResidentModelPluginHandle> pluginHandle;
  buddy::server::ResidentModelPluginHandle::ModelPtr model(
      nullptr, [](ResidentModel *m) { delete m; });
  try {
    if (!servingSoPath.empty()) {
      pluginHandle = std::make_unique<buddy::server::ResidentModelPluginHandle>(
          servingSoPath);
      if (modelType.empty()) {
        modelType = pluginHandle->modelType().empty()
                        ? "plugin"
                        : pluginHandle->modelType();
      }
      model = pluginHandle->createModel();
    } else {
      if (modelType.empty())
        modelType = "deepseek_r1";
      std::unique_ptr<ResidentModel> builtInModel =
          buddy::server::createResidentModel(modelType);
      model = buddy::server::ResidentModelPluginHandle::ModelPtr(
          builtInModel.release(), [](ResidentModel *m) { delete m; });
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
  ResidentModel &residentModel = *model;

  std::atomic<int> loadState{Loading};
  std::mutex loadErrorMutex;
  std::string loadError;

  SimpleHttpServer server;
  server.get("/health", [&](const HttpRequest &, ResponseWriter &writer) {
    const int state = loadState.load(std::memory_order_acquire);
    if (state == Ready) {
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toJson(residentModel.status())));
      return;
    }

    ModelStatus status;
    status.modelName =
        modelConfig.modelName.empty() ? modelType : modelConfig.modelName;
    if (state == Error) {
      status.state = ModelLoadState::Error;
      std::lock_guard<std::mutex> lock(loadErrorMutex);
      status.message = loadError;
    } else {
      status.state = ModelLoadState::Loading;
      status.message = "model is loading";
    }
    writer.sendResponse(
        buddy::server::jsonResponse(200, buddy::server::toJson(status)));
  });
  server.post("/completion",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                handleCompletion(residentModel, loadState, request, writer);
              });
  server.post("/v1/chat/completions",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                handleChat(residentModel, loadState, request, writer);
              });
  server.post("/tokenize",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                handleTokenize(residentModel, loadState, request, writer);
              });

  std::thread([&residentModel, modelConfig, &loadState, &loadError,
               &loadErrorMutex] {
    try {
      std::cerr << "[buddy-server] loading model...\n";
      residentModel.load(modelConfig);
      loadState.store(Ready, std::memory_order_release);
      std::cerr << "[buddy-server] model loaded\n";
    } catch (const std::exception &ex) {
      {
        std::lock_guard<std::mutex> lock(loadErrorMutex);
        loadError = ex.what();
      }
      loadState.store(Error, std::memory_order_release);
      std::cerr << "[buddy-server] failed to load model: " << ex.what() << "\n";
    }
  }).detach();

  try {
    server.listen(host, port);
  } catch (const std::exception &ex) {
    std::cerr << "[buddy-server] " << ex.what() << "\n";
    return 1;
  }

  return 0;
}

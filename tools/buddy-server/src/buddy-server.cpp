//===- buddy-server.cpp - Buddy model HTTP server -----------------------===//
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

#include "BuddyServerApp.h"

#include <iostream>
#include <utility>

#include "CLI11.hpp"

int main(int argc, char **argv) {
  buddy::server::BuddyServerOptions options;

  CLI::App app{"buddy-server: HTTP inference server for Buddy MLIR models"};
  app.set_version_flag("--version", BUDDY_VERSION);
  app.add_option("--model", options.modelPath, "Model manifest")
      ->group("Model source");
  app.add_option("--model-type", options.modelType, "Model name override")
      ->group("Model source");
  app.add_option("--host", options.host, "Bind address")
      ->group("Server")
      ->capture_default_str();
  app.add_option("--port", options.port, "Bind port")
      ->group("Server")
      ->capture_default_str();
  app.add_option("--chat-template", options.chatTemplatePath,
                 "Chat template JSON")
      ->group("Server");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &error) {
    return app.exit(error);
  }

  if (options.modelPath.empty()) {
    std::cerr << "Provide --model <path.rax>.\n";
    return 2;
  }

  return buddy::server::BuddyServerApp(std::move(options)).run();
}

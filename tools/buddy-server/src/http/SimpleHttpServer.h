//===- SimpleHttpServer.h - Minimal HTTP/1.1 server -----------------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_SIMPLEHTTPSERVER_H
#define BUDDY_TOOLS_BUDDY_SERVER_SIMPLEHTTPSERVER_H

#include <functional>
#include <map>
#include <string>

namespace buddy {
namespace server {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string query;
  std::string body;
  std::map<std::string, std::string> headers;
};

struct HttpResponse {
  int status = 200;
  std::string contentType = "application/json";
  std::string body;
  std::map<std::string, std::string> headers;
};

class ResponseWriter {
public:
  explicit ResponseWriter(int fd);

  bool sendResponse(const HttpResponse &response);
  bool startSse(int status = 200);
  bool writeSseData(const std::string &data);
  bool writeRaw(const std::string &data);

private:
  int fd = -1;
};

using HttpHandler = std::function<void(const HttpRequest &, ResponseWriter &)>;

class SimpleHttpServer {
public:
  void get(const std::string &path, HttpHandler handler);
  void post(const std::string &path, HttpHandler handler);

  void listen(const std::string &host, int port);

private:
  std::map<std::string, HttpHandler> getHandlers;
  std::map<std::string, HttpHandler> postHandlers;

  void handleClient(int clientFd);
};

HttpResponse jsonResponse(int status, const std::string &body);
HttpResponse textResponse(int status, const std::string &body);

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_SIMPLEHTTPSERVER_H

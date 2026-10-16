//===- SimpleHttpServerTest.cpp - HTTP response lifecycle checks ---------===//
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

#include "SimpleHttpServer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using buddy::server::ResponseWriter;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Sockets {
  Sockets() {
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0)
      throw std::runtime_error("socketpair failed");
  }
  ~Sockets() {
    for (int socket : fd)
      if (socket >= 0)
        ::close(socket);
  }
  void closePeer() {
    ::close(fd[1]);
    fd[1] = -1;
  }
  std::string receive() {
    char buffer[4096];
    const auto count = ::recv(fd[1], buffer, sizeof(buffer), MSG_DONTWAIT);
    require(count > 0, "expected response bytes");
    return std::string(buffer, static_cast<std::size_t>(count));
  }
  int fd[2];
};

void checkResponses() {
  Sockets sockets;
  ResponseWriter writer(sockets.fd[0]);
  require(!writer.hasStarted() && !writer.isSse(),
          "response started too early");
  require(!writer.writeSseData("early"), "SSE data accepted before headers");
  require(writer.sendResponse(buddy::server::jsonResponse(400, "{}")),
          "JSON response failed");
  require(writer.hasStarted() && !writer.isSse(),
          "JSON response state changed");
  require(!writer.startSse(), "second response headers accepted");
  require(!writer.sendResponse(buddy::server::jsonResponse(500, "second")),
          "second HTTP response accepted");
  const auto bytes = sockets.receive();
  require(bytes.find("HTTP/1.1 400 Bad Request\r\n") == 0 &&
              bytes.find("Content-Length: 2\r\n") != std::string::npos &&
              bytes.substr(bytes.size() - 2) == "{}" &&
              bytes.find("second") == std::string::npos,
          "response wire headers/body changed");
}

void checkSseWrites() {
  Sockets sockets;
  ResponseWriter writer(sockets.fd[0]);
  require(writer.startSse(), "SSE headers failed");
  require(writer.isSse() && writer.hasStarted(), "SSE state missing");
  require(writer.writeSseData("{\"content\":\"A\"}"), "SSE event failed");
  const auto bytes = sockets.receive();
  require(bytes.find("Content-Type: text/event-stream\r\n") !=
                  std::string::npos &&
              bytes.find("\r\n\r\ndata: {\"content\":\"A\"}\n\n") !=
                  std::string::npos,
          "SSE framing changed");
  sockets.closePeer();
  require(!writer.writeSseData("after-close"), "closed peer write succeeded");
  require(!writer.sendResponse(buddy::server::jsonResponse(500, "second")),
          "stream failure allowed a second HTTP response");
}

void checkStartFailure() {
  Sockets sockets;
  sockets.closePeer();
  ResponseWriter writer(sockets.fd[0]);
  require(!writer.startSse(), "closed peer accepted SSE headers");
  require(writer.hasStarted() && writer.isSse(),
          "partial/failed header write did not retain response state");
  require(!writer.sendResponse(buddy::server::jsonResponse(500, "second")),
          "failed SSE startup allowed a second HTTP response");
}

} // namespace

int main() {
  try {
    checkResponses();
    checkSseWrites();
    checkStartFailure();
    std::cout << "HTTP response lifecycle checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

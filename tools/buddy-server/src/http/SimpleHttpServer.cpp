//===- SimpleHttpServer.cpp - Minimal HTTP/1.1 server ---------------------===//
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
#include "JsonCodec.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <strings.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace buddy {
namespace server {

namespace {

std::string trim(std::string value) {
  auto isSpace = [](unsigned char c) { return std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [&](unsigned char c) { return !isSpace(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char c) { return !isSpace(c); })
                  .base(),
              value.end());
  return value;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::string reasonPhrase(int status) {
  switch (status) {
  case 200:
    return "OK";
  case 400:
    return "Bad Request";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 500:
    return "Internal Server Error";
  case 503:
    return "Service Unavailable";
  default:
    return "OK";
  }
}

bool sendAll(int fd, const char *data, size_t size) {
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool sendAll(int fd, const std::string &data) {
  return sendAll(fd, data.data(), data.size());
}

bool readHttpRequest(int fd, HttpRequest &request) {
  std::string raw;
  char buf[4096];
  size_t headerEnd = std::string::npos;

  while ((headerEnd = raw.find("\r\n\r\n")) == std::string::npos) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return false;
    raw.append(buf, static_cast<size_t>(n));
    if (raw.size() > 1024 * 1024)
      return false;
  }

  std::string header = raw.substr(0, headerEnd);
  std::istringstream hs(header);
  std::string requestLine;
  if (!std::getline(hs, requestLine))
    return false;
  if (!requestLine.empty() && requestLine.back() == '\r')
    requestLine.pop_back();

  std::istringstream rl(requestLine);
  std::string target;
  std::string version;
  if (!(rl >> request.method >> target >> version))
    return false;

  auto queryPos = target.find('?');
  request.path = target.substr(0, queryPos);
  if (queryPos != std::string::npos)
    request.query = target.substr(queryPos + 1);

  std::string line;
  while (std::getline(hs, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    request.headers[lower(trim(line.substr(0, colon)))] =
        trim(line.substr(colon + 1));
  }

  if (request.headers.count("transfer-encoding"))
    throw std::invalid_argument("transfer-encoding is not supported");
  size_t contentLength = 0;
  auto it = request.headers.find("content-length");
  if (it != request.headers.end()) {
    const auto &value = it->second;
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return std::isdigit(c); }))
      throw std::invalid_argument(
          "content-length must be a nonnegative integer");
    try {
      contentLength = static_cast<size_t>(std::stoull(value));
    } catch (const std::out_of_range &) {
      throw std::invalid_argument("content-length is too large");
    }
    if (contentLength > 16 * 1024 * 1024)
      throw std::invalid_argument("request body exceeds 16 MiB");
  }

  request.body = raw.substr(headerEnd + 4);
  while (request.body.size() < contentLength) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return false;
    request.body.append(buf, static_cast<size_t>(n));
  }
  if (request.body.size() > contentLength)
    request.body.resize(contentLength);

  return true;
}

} // namespace

ResponseWriter::ResponseWriter(int fd) : fd(fd) {}

bool ResponseWriter::sendResponse(const HttpResponse &response) {
  if (started)
    return false;
  started = true;
  std::ostringstream os;
  os << "HTTP/1.1 " << response.status << " " << reasonPhrase(response.status)
     << "\r\n";
  os << "Content-Type: " << response.contentType << "\r\n";
  os << "Content-Length: " << response.body.size() << "\r\n";
  os << "Connection: close\r\n";
  for (const auto &kv : response.headers)
    os << kv.first << ": " << kv.second << "\r\n";
  os << "\r\n";
  os << response.body;
  return sendAll(fd, os.str());
}

bool ResponseWriter::startSse(int status) {
  if (started)
    return false;
  started = true;
  sse = true;
  std::ostringstream os;
  os << "HTTP/1.1 " << status << " " << reasonPhrase(status) << "\r\n";
  os << "Content-Type: text/event-stream\r\n";
  os << "Cache-Control: no-cache\r\n";
  os << "Connection: close\r\n";
  os << "X-Accel-Buffering: no\r\n";
  os << "\r\n";
  return sendAll(fd, os.str());
}

bool ResponseWriter::writeSseData(const std::string &data) {
  if (!sse)
    return false;
  return writeRaw("data: " + data + "\n\n");
}

bool ResponseWriter::writeRaw(const std::string &data) {
  return sendAll(fd, data);
}

void SimpleHttpServer::get(const std::string &path, HttpHandler handler) {
  getHandlers[path] = std::move(handler);
}

void SimpleHttpServer::post(const std::string &path, HttpHandler handler) {
  postHandlers[path] = std::move(handler);
}

void SimpleHttpServer::listen(const std::string &host, int port) {
  int serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (serverFd < 0)
    throw std::runtime_error("socket failed: " + std::string(strerror(errno)));

  int yes = 1;
  setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(serverFd);
    throw std::runtime_error("invalid bind address: " + host);
  }

  if (::bind(serverFd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::string err = strerror(errno);
    ::close(serverFd);
    throw std::runtime_error("bind failed: " + err);
  }

  if (::listen(serverFd, 64) < 0) {
    std::string err = strerror(errno);
    ::close(serverFd);
    throw std::runtime_error("listen failed: " + err);
  }

  std::cerr << "[buddy-server] listening on http://" << host << ":" << port
            << "\n";

  while (true) {
    int clientFd = ::accept(serverFd, nullptr, nullptr);
    if (clientFd < 0)
      continue;
    std::thread([this, clientFd] { handleClient(clientFd); }).detach();
  }
}

void SimpleHttpServer::handleClient(int clientFd) {
  HttpRequest request;
  ResponseWriter writer(clientFd);

  try {
    if (!readHttpRequest(clientFd, request)) {
      writer.sendResponse(
          jsonResponse(400, errorJson("bad request", "bad_request", 400)));
      ::close(clientFd);
      return;
    }

    const bool knownPath =
        getHandlers.count(request.path) || postHandlers.count(request.path);
    if (!knownPath) {
      writer.sendResponse(
          jsonResponse(404, errorJson("not found", "not_found", 404)));
      ::close(clientFd);
      return;
    }

    const auto &handlers = request.method == "GET" ? getHandlers : postHandlers;
    auto it = handlers.find(request.path);
    if ((request.method != "GET" && request.method != "POST") ||
        it == handlers.end()) {
      auto response = jsonResponse(
          405, errorJson("method not allowed", "method_not_allowed", 405));
      response.headers["Allow"] =
          getHandlers.count(request.path) ? "GET" : "POST";
      writer.sendResponse(response);
      ::close(clientFd);
      return;
    }

    it->second(request, writer);
  } catch (const std::invalid_argument &ex) {
    const auto body = errorJson(ex.what(), "bad_request", 400);
    if (writer.isSse())
      writer.writeSseData(body);
    else
      writer.sendResponse(jsonResponse(400, body));
  } catch (const std::exception &ex) {
    const auto body = errorJson(ex.what(), "internal_error", 500);
    if (writer.isSse())
      writer.writeSseData(body);
    else
      writer.sendResponse(jsonResponse(500, body));
  }

  ::close(clientFd);
}

HttpResponse jsonResponse(int status, const std::string &body) {
  HttpResponse response;
  response.status = status;
  response.contentType = "application/json";
  response.body = body;
  return response;
}

HttpResponse textResponse(int status, const std::string &body) {
  HttpResponse response;
  response.status = status;
  response.contentType = "text/plain";
  response.body = body;
  return response;
}

} // namespace server
} // namespace buddy

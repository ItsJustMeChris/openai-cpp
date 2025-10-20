#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace openai {

struct HttpRequest {
  std::string method;
  std::string url;
  std::map<std::string, std::string> headers;
  std::string body;
  std::chrono::milliseconds timeout{60000};
  std::function<void(const char*, std::size_t)> on_chunk;
  bool collect_body = true;
};

struct HttpResponse {
  long status_code = 0;
  std::map<std::string, std::string> headers;
  std::string body;
};

class HttpClient {
public:
  virtual ~HttpClient() = default;
  virtual HttpResponse request(const HttpRequest& request) = 0;
};

std::unique_ptr<HttpClient> make_default_http_client();

}  // namespace openai

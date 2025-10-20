#include "openai/http_client.hpp"

#include "openai/error.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace openai {
namespace {

struct WriteContext {
  std::string* body;
  std::function<void(const char*, std::size_t)>* on_chunk;
  bool error = false;
  std::string error_message;
};

size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* context = static_cast<WriteContext*>(userdata);
  const size_t total = size * nmemb;
  if (context->on_chunk && *context->on_chunk) {
    try {
      (*context->on_chunk)(ptr, total);
    } catch (const std::exception& ex) {
      context->error = true;
      context->error_message = ex.what();
      return 0;
    }
  }
  if (context->body) {
    context->body->append(ptr, total);
  }
  return total;
}

size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
  std::size_t total_size = size * nitems;
  std::string line(buffer, total_size);

  auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);
  auto colon_pos = line.find(':');
  if (colon_pos != std::string::npos) {
    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);

    auto trim = [](std::string& s) {
      auto not_space = [](unsigned char ch) { return !std::isspace(static_cast<unsigned char>(ch)); };
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
      s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    };

    trim(key);
    trim(value);
    if (!key.empty()) {
      (*headers)[key] = value;
    }
  }

  return total_size;
}

class CurlHttpClient : public HttpClient {
public:
  CurlHttpClient() = default;

  HttpResponse request(const HttpRequest& request) override {
    CURL* curl = curl_easy_init();
    if (!curl) {
      throw OpenAIError("Failed to initialize libcurl");
    }

    struct curl_slist* header_list = nullptr;
    for (const auto& [key, value] : request.headers) {
      std::string header = key + ": " + value;
      header_list = curl_slist_append(header_list, header.c_str());
    }

    std::string response_body;
    std::map<std::string, std::string> response_headers;

    curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, request.method.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
  std::function<void(const char*, std::size_t)> on_chunk = request.on_chunk;
  WriteContext context{
      request.collect_body ? &response_body : nullptr,
      on_chunk ? &on_chunk : nullptr,
  };

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response_headers);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(request.timeout.count()));
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "openai-cpp/0.1");

    if (!request.body.empty()) {
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.body.c_str());
      curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request.body.size());
    }

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
      std::string message = std::string("libcurl error: ") + curl_easy_strerror(res);
      curl_slist_free_all(header_list);
      curl_easy_cleanup(curl);
    throw OpenAIError(message);
  }

  long status_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);

  if (context.error) {
    throw OpenAIError(context.error_message.empty() ? "Streaming callback failed" : context.error_message);
  }

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    return HttpResponse{status_code, response_headers, response_body};
  }
};

struct CurlGlobalState {
  CurlGlobalState() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobalState() { curl_global_cleanup(); }
};

CurlGlobalState& curl_state() {
  static CurlGlobalState state;
  return state;
}

}  // namespace

std::unique_ptr<HttpClient> make_default_http_client() {
  (void)curl_state();
  return std::make_unique<CurlHttpClient>();
}

}  // namespace openai

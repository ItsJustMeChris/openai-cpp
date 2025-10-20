#include "openai/client.hpp"

#include "openai/error.hpp"
#include "openai/http_client.hpp"

#include <nlohmann/json.hpp>

#include <utility>
#include <stdexcept>

namespace openai {
namespace {

using json = nlohmann::json;

json completion_request_to_json(const CompletionRequest& request) {
  json body;
  body["model"] = request.model;
  body["prompt"] = request.prompt;
  if (request.max_tokens) {
    body["max_tokens"] = *request.max_tokens;
  }
  if (request.temperature) {
    body["temperature"] = *request.temperature;
  }
  if (request.top_p) {
    body["top_p"] = *request.top_p;
  }
  if (request.n) {
    body["n"] = *request.n;
  }
  if (request.stop) {
    body["stop"] = *request.stop;
  }
  if (request.stream) {
    body["stream"] = *request.stream;
  }
  return body;
}

Completion parse_completion(const json& payload) {
  Completion completion;
  completion.id = payload.value("id", "");
  completion.object = payload.value("object", "");
  completion.created = payload.value("created", 0);
  completion.model = payload.value("model", "");

  if (payload.contains("choices")) {
    for (const auto& choice_json : payload.at("choices")) {
      CompletionChoice choice;
      choice.index = choice_json.value("index", 0);
      choice.text = choice_json.value("text", "");
      choice.finish_reason = choice_json.value("finish_reason", "");
      completion.choices.push_back(std::move(choice));
    }
  }

  if (payload.contains("usage")) {
    CompletionUsage usage;
    const json& usage_json = payload.at("usage");
    usage.prompt_tokens = usage_json.value("prompt_tokens", 0);
    usage.completion_tokens = usage_json.value("completion_tokens", 0);
    usage.total_tokens = usage_json.value("total_tokens", 0);
    completion.usage = usage;
  }

  return completion;
}

std::string build_url(const std::string& base_url, const std::string& path) {
  if (path.empty()) {
    return base_url;
  }
  if (path.find("http://") == 0 || path.find("https://") == 0) {
    return path;
  }
  std::string url = base_url;
  if (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  if (!path.empty() && path.front() != '/') {
    url.push_back('/');
  }
  url += path;
  return url;
}

std::string extract_error_message(const json& payload) {
  if (payload.contains("error")) {
    const auto& err = payload.at("error");
    if (err.is_object()) {
      return err.value("message", "");
    }
    if (err.is_string()) {
      return err.get<std::string>();
    }
  }
  return {};
}

}  // namespace

OpenAIClient::OpenAIClient(ClientOptions options,
                           std::unique_ptr<HttpClient> http_client)
    : options_(std::move(options)),
      http_client_(http_client ? std::move(http_client) : make_default_http_client()),
      completions_(*this) {
  if (options_.api_key.empty()) {
    throw OpenAIError("ClientOptions.api_key must be set");
  }
}

Completion OpenAIClient::create_completion(const CompletionRequest& request,
                                           const RequestOptions& options) {
  return completions_.create(request, options);
}

HttpResponse OpenAIClient::perform_request(const std::string& method,
                                           const std::string& path,
                                           const std::string& body,
                                           const RequestOptions& options) const {
  HttpRequest http_request;
  http_request.method = method;
  http_request.url = build_url(options_.base_url, path);
  http_request.body = body;
  http_request.timeout = options.timeout.value_or(options_.timeout);

  http_request.headers.clear();
  // The API expects JSON by default.
  if (!body.empty()) {
    http_request.headers["Content-Type"] = "application/json";
  }
  http_request.headers["Accept"] = "application/json";
  http_request.headers["Authorization"] = std::string("Bearer ") + options_.api_key;

  if (options_.organization) {
    http_request.headers["OpenAI-Organization"] = *options_.organization;
  }
  if (options_.project) {
    http_request.headers["OpenAI-Project"] = *options_.project;
  }
  if (options.idempotency_key) {
    http_request.headers["Idempotency-Key"] = *options.idempotency_key;
  }

  for (const auto& [key, value] : options.headers) {
    http_request.headers[key] = value;
  }

  auto response = http_client_->request(http_request);

  if (response.status_code >= 400) {
    std::string message;
    try {
      auto payload = json::parse(response.body);
      message = extract_error_message(payload);
    } catch (const std::exception&) {
      // Ignore parsing errors.
    }
    if (message.empty()) {
      message = "HTTP " + std::to_string(response.status_code) + " error";
    }
    throw HttpError(response.status_code, message);
  }

  return response;
}

Completion CompletionsResource::create(const CompletionRequest& request,
                                       const RequestOptions& options) const {
  auto body = completion_request_to_json(request).dump();
  auto response = client_.perform_request("POST", "/completions", body, options);
  try {
    auto payload = json::parse(response.body);
    return parse_completion(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse completion response: ") + ex.what());
  }
}

}  // namespace openai

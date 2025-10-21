#include "openai/client.hpp"

#include "openai/error.hpp"
#include "openai/http_client.hpp"

#include <nlohmann/json.hpp>

#include <utility>
#include <stdexcept>
#include <variant>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cctype>

#include "openai/utils/base64.hpp"

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

json embedding_input_to_json(const EmbeddingRequest::Input& input) {
  return std::visit(
      [](const auto& value) -> json {
        return json(value);
      },
      input);
}

json embedding_request_to_json(const EmbeddingRequest& request) {
  json body;
  body["model"] = request.model;
  body["input"] = embedding_input_to_json(request.input);
  if (request.dimensions) {
    body["dimensions"] = *request.dimensions;
  }
  if (request.encoding_format) {
    body["encoding_format"] = *request.encoding_format;
  }
  if (request.user) {
    body["user"] = *request.user;
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

bool is_unreserved(char c) {
  if (std::isalnum(static_cast<unsigned char>(c))) {
    return true;
  }
  switch (c) {
    case '-':
    case '_':
    case '.':
    case '~':
      return true;
    default:
      return false;
  }
}

std::string url_encode(const std::string& value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << std::hex << std::uppercase;

  for (unsigned char c : value) {
    if (is_unreserved(static_cast<char>(c))) {
      escaped << static_cast<char>(c);
    } else {
      escaped << '%'
              << std::setw(2)
              << std::setfill('0')
              << static_cast<int>(c);
    }
  }

  return escaped.str();
}

std::string append_query_params(const std::string& url,
                                const std::map<std::string, std::string>& query_params) {
  if (query_params.empty()) {
    return url;
  }

  std::ostringstream query;
  bool first = true;
  for (const auto& [key, value] : query_params) {
    if (key.empty()) {
      continue;
    }
    if (!first) {
      query << '&';
    }
    query << url_encode(key) << '=';
    query << url_encode(value);
    first = false;
  }

  std::string query_string = query.str();
  if (query_string.empty()) {
    return url;
  }

  std::string result = url;
  result += (url.find('?') == std::string::npos) ? '?' : '&';
  result += query_string;
  return result;
}

std::vector<float> bytes_to_float32(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() % 4 != 0) {
    throw OpenAIError("Embedding bytes length must be a multiple of 4");
  }
  std::vector<float> values(bytes.size() / 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint32_t word = static_cast<std::uint32_t>(bytes[i * 4]) |
                         (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8) |
                         (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16) |
                         (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24);
    float value;
    std::memcpy(&value, &word, sizeof(float));
    values[i] = value;
  }
  return values;
}

Embedding parse_embedding(const json& payload, bool decode_base64) {
  Embedding embedding;
  embedding.index = payload.value("index", 0);
  embedding.object = payload.value("object", "");

  const auto& data = payload.at("embedding");
  if (decode_base64) {
    if (!data.is_string()) {
      throw OpenAIError("Expected base64 string for embedding data");
    }
    auto bytes = utils::decode_base64(data.get<std::string>());
    embedding.embedding = bytes_to_float32(bytes);
  } else {
    if (data.is_string()) {
      embedding.embedding = data.get<std::string>();
    } else {
      std::vector<float> values;
      values.reserve(data.size());
      for (const auto& item : data) {
        values.push_back(static_cast<float>(item.get<double>()));
      }
      embedding.embedding = std::move(values);
    }
  }

  return embedding;
}

CreateEmbeddingResponse parse_embedding_response(const json& payload, bool decode_base64) {
  CreateEmbeddingResponse response;
  response.model = payload.value("model", "");
  response.object = payload.value("object", "");

  if (payload.contains("data")) {
    for (const auto& embedding_json : payload.at("data")) {
      response.data.push_back(parse_embedding(embedding_json, decode_base64));
    }
  }

  if (payload.contains("usage")) {
    EmbeddingUsage usage;
    const auto& usage_json = payload.at("usage");
    usage.prompt_tokens = usage_json.value("prompt_tokens", 0);
    usage.total_tokens = usage_json.value("total_tokens", 0);
    response.usage = usage;
  }

  return response;
}

Model parse_model(const json& payload) {
  Model model;
  model.id = payload.value("id", "");
  model.created = payload.value("created", 0);
  model.object = payload.value("object", "");
  model.owned_by = payload.value("owned_by", "");
  return model;
}

ModelList parse_model_list(const json& payload) {
  ModelList list;
  list.object = payload.value("object", "");
  if (payload.contains("data")) {
    for (const auto& model_json : payload.at("data")) {
      list.data.push_back(parse_model(model_json));
    }
  }
  return list;
}

ModelDeleted parse_model_deleted(const json& payload) {
  ModelDeleted deleted;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
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
      completions_(*this),
      models_(*this),
      embeddings_(*this),
      moderations_(*this),
      responses_(*this),
      files_(*this),
      images_(*this),
      audio_(*this),
      vector_stores_(*this),
      assistants_(*this),
      threads_(*this),
      chat_(*this) {
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
  std::string url = build_url(options_.base_url, path);
  url = append_query_params(url, options.query_params);
  http_request.url = std::move(url);
  http_request.body = body;
  http_request.timeout = options.timeout.value_or(options_.timeout);

  http_request.headers.clear();
  http_request.on_chunk = options.on_chunk;
  http_request.collect_body = options.collect_body;
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

HttpResponse OpenAIClient::perform_request(const PageRequestOptions& options) const {
  RequestOptions request_options;
  request_options.headers = options.headers;
  request_options.query_params = options.query;
  return perform_request(options.method, options.path, options.body, request_options);
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

Model ModelsResource::retrieve(const std::string& model_id, const RequestOptions& options) const {
  auto path = std::string("/models/") + model_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_model(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse model response: ") + ex.what());
  }
}

ModelList ModelsResource::list(const RequestOptions& options) const {
  auto response = client_.perform_request("GET", "/models", "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_model_list(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse models list: ") + ex.what());
  }
}

ModelDeleted ModelsResource::Delete(const std::string& model_id, const RequestOptions& options) const {
  auto path = std::string("/models/") + model_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_model_deleted(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse model deletion response: ") + ex.what());
  }
}

CreateEmbeddingResponse EmbeddingsResource::create(const EmbeddingRequest& request,
                                                   const RequestOptions& options) const {
  const bool has_user_encoding = request.encoding_format.has_value();
  EmbeddingRequest request_body = request;
  if (!request_body.encoding_format) {
    request_body.encoding_format = std::string("base64");
  }

  auto body = embedding_request_to_json(request_body).dump();
  auto response = client_.perform_request("POST", "/embeddings", body, options);
  try {
    auto payload = json::parse(response.body);
    const bool decode_base64 =
        !has_user_encoding && request_body.encoding_format && *request_body.encoding_format == "base64";
    return parse_embedding_response(payload, decode_base64);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse embedding response: ") + ex.what());
  }
}

}  // namespace openai

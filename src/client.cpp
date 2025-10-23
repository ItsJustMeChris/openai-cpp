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
#include <random>
#include <array>
#include <algorithm>
#include <locale>
#include <cmath>
#include <ctime>
#include <string_view>
#include <cstdlib>
#include <set>
#include <optional>

#include "openai/logging.hpp"
#include "openai/utils/base64.hpp"
#include "openai/utils/env.hpp"
#include "openai/utils/platform.hpp"
#include "openai/utils/qs.hpp"
#include "openai/utils/time.hpp"
#include "openai/utils/uuid.hpp"
#include "openai/utils/values.hpp"

namespace openai {
namespace {

using json = nlohmann::json;

constexpr std::chrono::milliseconds kMaxRetryAfter = std::chrono::milliseconds(60'000);
constexpr const char* kDefaultBaseUrl = "https://api.openai.com/v1";

const std::set<std::string> kAzureDeploymentEndpoints = {
    "/completions",
    "/chat/completions",
    "/embeddings",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/images/generations",
    "/batches",
    "/images/edits"};

bool iequals(std::string_view lhs, std::string_view rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [](char a, char b) {
           return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
         });
}

std::optional<std::string> get_header_value(const std::map<std::string, std::string>& headers,
                                            std::string_view key) {
  for (const auto& [name, value] : headers) {
    if (iequals(name, key)) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<std::chrono::milliseconds> parse_numeric_retry_after_ms(const std::string& value) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end != value.c_str() && !std::isnan(parsed)) {
    if (parsed < 0) {
      return std::chrono::milliseconds(0);
    }
    return std::chrono::milliseconds(static_cast<long>(parsed));
  }
  return std::nullopt;
}

std::optional<std::chrono::milliseconds> parse_retry_after_seconds(const std::string& value) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end != value.c_str() && !std::isnan(parsed)) {
    if (parsed < 0) {
      return std::chrono::milliseconds(0);
    }
    return std::chrono::milliseconds(static_cast<long>(parsed * 1000.0));
  }
  return std::nullopt;
}

std::optional<std::chrono::milliseconds> parse_retry_after_http_date(const std::string& value) {
  std::tm tm{};
  std::istringstream stream(value);
  stream.imbue(std::locale::classic());
  stream >> std::get_time(&tm, "%a, %d %b %Y %H:%M:%S GMT");
  if (stream.fail()) {
    return std::nullopt;
  }
#if defined(_WIN32)
  std::time_t utc_time = _mkgmtime(&tm);
#else
  std::time_t utc_time = timegm(&tm);
#endif
  if (utc_time == static_cast<std::time_t>(-1)) {
    return std::nullopt;
  }
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  auto delta = std::difftime(utc_time, now_time);
  if (delta <= 0) {
    return std::chrono::milliseconds(0);
  }
  return std::chrono::milliseconds(static_cast<long>(delta * 1000.0));
}

std::optional<std::chrono::milliseconds> parse_retry_after(const std::map<std::string, std::string>& headers) {
  if (auto retry_after_ms = get_header_value(headers, "retry-after-ms")) {
    if (auto parsed = parse_numeric_retry_after_ms(*retry_after_ms)) {
      return parsed;
    }
  }
  if (auto retry_after = get_header_value(headers, "retry-after")) {
    if (auto parsed_seconds = parse_retry_after_seconds(*retry_after)) {
      return parsed_seconds;
    }
    if (auto parsed_date = parse_retry_after_http_date(*retry_after)) {
      return parsed_date;
    }
  }
  return std::nullopt;
}

bool should_retry_status(long status) {
  if (status == 408 || status == 409 || status == 429) {
    return true;
  }
  return status >= 500;
}

bool should_retry_response(const HttpResponse& response, std::size_t retries_remaining) {
  if (retries_remaining == 0) {
    return false;
  }
  if (auto explicit_retry = get_header_value(response.headers, "x-should-retry")) {
    std::string lowered;
    lowered.reserve(explicit_retry->size());
    for (char ch : *explicit_retry) {
      lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lowered == "true") {
      return true;
    }
    if (lowered == "false") {
      return false;
    }
  }
  return should_retry_status(response.status_code);
}

std::chrono::milliseconds compute_retry_delay(const HttpResponse* response,
                                              std::size_t retries_remaining,
                                              std::size_t max_retries) {
  if (response) {
    if (auto header_delay = parse_retry_after(response->headers)) {
      if (header_delay->count() >= 0 && *header_delay < kMaxRetryAfter) {
        return *header_delay;
      }
    }
  }
  auto delay = utils::calculate_default_retry_delay(retries_remaining, max_retries);
  if (delay < std::chrono::milliseconds(0)) {
    return std::chrono::milliseconds(0);
  }
  if (delay > kMaxRetryAfter) {
    return kMaxRetryAfter;
  }
  return delay;
}

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

void apply_optional_entries(std::map<std::string, std::string>& target,
                            const std::map<std::string, std::optional<std::string>>& overrides) {
  for (const auto& [key, value] : overrides) {
    if (value.has_value()) {
      target[key] = *value;
    } else {
      target.erase(key);
    }
  }
}

json build_query_object(const std::map<std::string, std::string>& default_query,
                        const RequestOptions& options) {
  json query = json::object();
  for (const auto& [key, value] : default_query) {
    query[key] = value;
  }
  for (const auto& [key, value] : options.query_params) {
    if (value.has_value()) {
      query[key] = *value;
    } else {
      query.erase(key);
    }
  }
  if (options.query) {
    if (!options.query->is_object()) {
      throw OpenAIError("RequestOptions::query must be a JSON object");
    }
    for (const auto& item : options.query->items()) {
      query[item.key()] = item.value();
    }
  }
  return query;
}

std::string append_query_string(const std::string& url, const std::string& query_string) {
  if (query_string.empty()) {
    return url;
  }
  std::string result = url;
  if (query_string.front() == '?') {
    if (query_string.size() == 1) {
      return result;
    }
    if (url.find('?') == std::string::npos) {
      result += query_string;
    } else {
      result += '&';
      result += query_string.substr(1);
    }
    return result;
  }
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
  if (utils::is_absolute_url(path)) {
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

nlohmann::json extract_error_payload(const nlohmann::json& payload) {
  if (payload.contains("error")) {
    const auto& err = payload.at("error");
    if (err.is_object()) {
      return err;
    }
  }
  return payload;
}

std::map<std::string, std::string> sanitize_headers(const std::map<std::string, std::string>& headers) {
  static const std::set<std::string> kSensitive = {"authorization", "cookie", "set-cookie"};
  std::map<std::string, std::string> sanitized;
  for (const auto& [key, value] : headers) {
    std::string lowered; lowered.reserve(key.size());
    std::transform(key.begin(), key.end(), std::back_inserter(lowered), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (kSensitive.count(lowered)) {
      sanitized[key] = "***";
    } else {
      sanitized[key] = value;
    }
  }
  return sanitized;
}

nlohmann::json build_request_log_details(const HttpRequest& request, std::size_t retry_count) {
  nlohmann::json details;
  details["method"] = request.method;
  details["url"] = request.url;
  details["retry_count"] = static_cast<int>(retry_count);
  details["headers"] = sanitize_headers(request.headers);
  return details;
}

nlohmann::json build_response_log_details(const HttpRequest& request,
                                          const HttpResponse& response,
                                          std::chrono::steady_clock::duration duration,
                                          std::size_t retry_count) {
  nlohmann::json details = build_request_log_details(request, retry_count);
  details["status"] = response.status_code;
  details["duration_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  details["response_headers"] = sanitize_headers(response.headers);
  return details;
}

[[noreturn]] void throw_api_error(long status,
                                  const std::string& fallback_message,
                                  const nlohmann::json& error_payload,
                                  const std::map<std::string, std::string>& headers) {
  const std::string message = fallback_message.empty() ? ("HTTP " + std::to_string(status) + " error") : fallback_message;
  const nlohmann::json& body = error_payload;

  switch (status) {
    case 400:
      throw BadRequestError(message, status, body, headers);
    case 401:
      throw AuthenticationError(message, status, body, headers);
    case 403:
      throw PermissionDeniedError(message, status, body, headers);
    case 404:
      throw NotFoundError(message, status, body, headers);
    case 409:
      throw ConflictError(message, status, body, headers);
    case 422:
      throw UnprocessableEntityError(message, status, body, headers);
    case 429:
      throw RateLimitError(message, status, body, headers);
    case 500:
    case 502:
    case 503:
    case 504:
      throw InternalServerError(message, status, body, headers);
    default:
      throw APIError(message, status, body, headers);
  }
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
      graders_(*this),
      assistants_(*this),
      threads_(*this),
      thread_messages_(*this),
      runs_(*this),
      run_steps_(*this),
      chat_(*this),
      containers_(*this),
      videos_(*this),
      fine_tuning_(*this),
      webhooks_(*this),
      conversations_(*this),
      beta_(*this),
      batches_(*this),
      uploads_(*this) {
  if (options_.api_key.empty() && !options_.api_key_provider) {
    if (auto env_api = utils::read_env("OPENAI_API_KEY")) {
      options_.api_key = *env_api;
    }
  }

  if (auto env_base = utils::read_env("OPENAI_BASE_URL")) {
    if (!env_base->empty()) {
      if (options_.base_url == kDefaultBaseUrl) {
        options_.base_url = *env_base;
      }
    } else if (options_.base_url.empty()) {
      options_.base_url = kDefaultBaseUrl;
    }
  }

  if (!options_.organization) {
    if (auto env_org = utils::read_env("OPENAI_ORG_ID")) {
      options_.organization = *env_org;
    }
  }

  if (!options_.project) {
    if (auto env_project = utils::read_env("OPENAI_PROJECT_ID")) {
      options_.project = *env_project;
    }
  }

  if (!options_.webhook_secret) {
    if (auto env_webhook = utils::read_env("OPENAI_WEBHOOK_SECRET")) {
      options_.webhook_secret = *env_webhook;
    }
  }

  if (options_.log_level == LogLevel::Off) {
    if (auto env_log = utils::read_env("OPENAI_LOG")) {
      if (!env_log->empty()) {
        options_.log_level = parse_log_level(*env_log, options_.log_level);
      }
    }
  }

  if (options_.api_key.empty() && !options_.api_key_provider) {
    throw OpenAIError("Missing API key. Provide ClientOptions.api_key or set the OPENAI_API_KEY environment variable.");
  }

  utils::validate_positive_integer("ClientOptions.timeout", options_.timeout.count());
}

Completion OpenAIClient::create_completion(const CompletionRequest& request,
                                           const RequestOptions& options) {
  return completions_.create(request, options);
}

void OpenAIClient::log(LogLevel level, const std::string& message, const nlohmann::json& details) const {
  if (!options_.logger) {
    return;
  }
  if (static_cast<int>(level) > static_cast<int>(options_.log_level)) {
    return;
  }
  options_.logger(level, message, details);
}

HttpResponse OpenAIClient::perform_request(const std::string& method,
                                           const std::string& path,
                                           const std::string& body,
                                           const RequestOptions& options) const {
  if (options.timeout) {
    utils::validate_positive_integer("RequestOptions.timeout", options.timeout->count());
  }
  if (options.max_retries) {
    utils::validate_positive_integer("RequestOptions.max_retries",
                                     static_cast<long long>(*options.max_retries));
  }
  const std::size_t max_retries = options.max_retries.value_or(options_.max_retries);
  std::size_t retries_remaining = max_retries;
  std::optional<std::string> idempotency_key = options.idempotency_key;
  if (!idempotency_key && !iequals(method, "GET")) {
    idempotency_key = utils::uuid4();
  }

  std::string request_path = path;
  if (options_.azure_deployment_routing && iequals(method, "POST")) {
    if (kAzureDeploymentEndpoints.count(path) > 0 && options_.base_url.find("/deployments/") == std::string::npos) {
      std::string deployment = options_.azure_deployment_name.value_or("");
      if (deployment.empty() && !body.empty()) {
        if (auto parsed = utils::safe_json(body)) {
          if (parsed->contains("model") && (*parsed)["model"].is_string()) {
            deployment = (*parsed)["model"].get<std::string>();
          }
        }
      }
      if (!deployment.empty()) {
        request_path = std::string("/deployments/") + deployment + path;
      }
    }
  }

  auto build_request = [&](std::size_t retry_count) {
    HttpRequest http_request;
    http_request.method = method;
    std::string url = build_url(options_.base_url, request_path);
    json query_json = build_query_object(options_.default_query, options);
    utils::qs::StringifyOptions qs_options;
    qs_options.array_format = utils::qs::ArrayFormat::Brackets;
    std::string query_string = utils::qs::stringify(query_json, qs_options);
    url = append_query_string(url, query_string);
    http_request.url = std::move(url);
    http_request.body = body;
    http_request.timeout = options.timeout.value_or(options_.timeout);
    http_request.on_chunk = options.on_chunk;
    http_request.collect_body = options.collect_body;

    std::map<std::string, std::string> headers;

    if (idempotency_key) {
      headers["Idempotency-Key"] = *idempotency_key;
    }

    headers["Accept"] = "application/json";
    headers["User-Agent"] = utils::user_agent();
    headers["X-Stainless-Retry-Count"] = std::to_string(retry_count);
    auto timeout_seconds = std::chrono::duration_cast<std::chrono::seconds>(http_request.timeout).count();
    if (timeout_seconds > 0) {
      headers["X-Stainless-Timeout"] = std::to_string(timeout_seconds);
    }
    for (const auto& [key, value] : utils::platform_headers()) {
      headers[key] = value;
    }
    std::string auth_token;
    if (options_.api_key_provider) {
      auth_token = options_.api_key_provider();
    } else {
      auth_token = options_.api_key;
    }
    if (options_.use_bearer_auth) {
      if (!auth_token.empty()) {
        headers["Authorization"] = std::string("Bearer ") + auth_token;
      }
    } else if (options_.alternative_auth_header && !auth_token.empty()) {
      headers[*options_.alternative_auth_header] = options_.alternative_auth_prefix + auth_token;
    }

    if (options_.organization) {
      headers["OpenAI-Organization"] = *options_.organization;
    }
    if (options_.project) {
      headers["OpenAI-Project"] = *options_.project;
    }

    for (const auto& [key, value] : options_.default_headers) {
      headers[key] = value;
    }

    // The API expects JSON by default.
    if (!body.empty()) {
      headers["Content-Type"] = "application/json";
    }

    apply_optional_entries(headers, options.headers);
    http_request.headers = std::move(headers);
    return http_request;
  };

  while (true) {
    const std::size_t retry_count = max_retries - retries_remaining;
    HttpRequest http_request = build_request(retry_count);
    log(LogLevel::Debug, "sending request", build_request_log_details(http_request, retry_count));
    auto start_time = std::chrono::steady_clock::now();
    HttpResponse response;
    try {
      response = http_client_->request(http_request);
    } catch (const OpenAIError& error) {
      if (retries_remaining == 0) {
        throw APIConnectionError(error.what());
      }
      log(LogLevel::Warn, "request failed, retrying", build_request_log_details(http_request, retry_count));
      auto delay = compute_retry_delay(nullptr, retries_remaining, max_retries);
      utils::sleep_for(delay);
      --retries_remaining;
      continue;
    } catch (const std::exception& error) {
      if (retries_remaining == 0) {
        throw APIConnectionError(error.what());
      }
      log(LogLevel::Warn, "request failed, retrying", build_request_log_details(http_request, retry_count));
      auto delay = compute_retry_delay(nullptr, retries_remaining, max_retries);
      utils::sleep_for(delay);
      --retries_remaining;
      continue;
    }

    if (response.status_code < 400) {
      auto duration = std::chrono::steady_clock::now() - start_time;
      log(LogLevel::Info, "request succeeded", build_response_log_details(http_request, response, duration, retry_count));
      return response;
    }

    if (should_retry_response(response, retries_remaining)) {
      auto delay = compute_retry_delay(&response, retries_remaining, max_retries);
      auto duration = std::chrono::steady_clock::now() - start_time;
      auto details = build_response_log_details(http_request, response, duration, retry_count);
      details["retry_delay_ms"] = delay.count();
      log(LogLevel::Warn, "retrying request after error", details);
      utils::sleep_for(delay);
      --retries_remaining;
      continue;
    }

    nlohmann::json error_payload = nlohmann::json::object();
    std::string message;
    if (auto payload = utils::safe_json(response.body)) {
      message = extract_error_message(*payload);
      error_payload = extract_error_payload(*payload);
    }
    auto duration = std::chrono::steady_clock::now() - start_time;
    log(LogLevel::Error, "request failed", build_response_log_details(http_request, response, duration, retry_count));
    throw_api_error(response.status_code, message, error_payload, response.headers);
  }

  throw OpenAIError("Retry loop exited unexpectedly");
}

HttpResponse OpenAIClient::perform_request(const PageRequestOptions& options) const {
  RequestOptions request_options;
  for (const auto& [key, value] : options.headers) {
    request_options.headers[key] = value;
  }
  request_options.query = options.query;
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

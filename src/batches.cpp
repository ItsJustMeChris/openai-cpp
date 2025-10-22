#include "openai/batches.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <nlohmann/json.hpp>

#include <functional>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kBatchesPath = "/batches";

void apply_list_params(const BatchListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) options.query_params["after"] = *params.after;
}

BatchError parse_batch_error(const json& payload) {
  BatchError error;
  if (payload.contains("code") && payload.at("code").is_string()) {
    error.code = payload.at("code").get<std::string>();
  }
  if (payload.contains("line")) {
    if (payload.at("line").is_number_integer()) {
      error.line = payload.at("line").get<int>();
    } else if (payload.at("line").is_number_float()) {
      error.line = static_cast<int>(payload.at("line").get<double>());
    } else if (payload.at("line").is_null()) {
      error.line = std::nullopt;
    }
  }
  if (payload.contains("message") && payload.at("message").is_string()) {
    error.message = payload.at("message").get<std::string>();
  }
  if (payload.contains("param") && payload.at("param").is_string()) {
    error.param = payload.at("param").get<std::string>();
  } else if (payload.contains("param") && payload.at("param").is_null()) {
    error.param = std::nullopt;
  }
  return error;
}

std::optional<BatchErrors> parse_batch_errors(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  BatchErrors errors;
  if (payload.contains("object") && payload.at("object").is_string()) {
    errors.object = payload.at("object").get<std::string>();
  }
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      errors.data.push_back(parse_batch_error(item));
    }
  }
  return errors;
}

std::optional<BatchRequestCounts> parse_batch_request_counts(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  BatchRequestCounts counts;
  counts.completed = payload.value("completed", 0);
  counts.failed = payload.value("failed", 0);
  counts.total = payload.value("total", 0);
  return counts;
}

std::optional<BatchUsage> parse_batch_usage(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  BatchUsage usage;
  usage.input_tokens = payload.value("input_tokens", 0);
  if (payload.contains("input_tokens_details") && payload.at("input_tokens_details").is_object()) {
    usage.input_tokens_details.cached_tokens =
        payload.at("input_tokens_details").value("cached_tokens", 0);
  }
  usage.output_tokens = payload.value("output_tokens", 0);
  if (payload.contains("output_tokens_details") && payload.at("output_tokens_details").is_object()) {
    usage.output_tokens_details.reasoning_tokens =
        payload.at("output_tokens_details").value("reasoning_tokens", 0);
  }
  usage.total_tokens = payload.value("total_tokens", 0);
  return usage;
}

std::optional<std::map<std::string, std::string>> parse_metadata(const json& payload) {
  if (payload.is_null()) {
    return std::nullopt;
  }
  if (!payload.is_object()) {
    return std::nullopt;
  }
  std::map<std::string, std::string> metadata;
  for (const auto& item : payload.items()) {
    if (item.value().is_string()) {
      metadata[item.key()] = item.value().get<std::string>();
    } else if (!item.value().is_null()) {
      metadata[item.key()] = item.value().dump();
    }
  }
  return metadata;
}

Batch parse_batch(const json& payload) {
  Batch batch;
  batch.id = payload.value("id", "");
  batch.completion_window = payload.value("completion_window", "");
  batch.created_at = payload.value("created_at", 0);
  batch.endpoint = payload.value("endpoint", "");
  batch.input_file_id = payload.value("input_file_id", "");
  batch.object = payload.value("object", "");
  batch.status = payload.value("status", "");
  if (payload.contains("cancelled_at") && payload.at("cancelled_at").is_number()) {
    batch.cancelled_at = payload.at("cancelled_at").get<int>();
  }
  if (payload.contains("cancelling_at") && payload.at("cancelling_at").is_number()) {
    batch.cancelling_at = payload.at("cancelling_at").get<int>();
  }
  if (payload.contains("completed_at") && payload.at("completed_at").is_number()) {
    batch.completed_at = payload.at("completed_at").get<int>();
  }
  if (payload.contains("error_file_id") && payload.at("error_file_id").is_string()) {
    batch.error_file_id = payload.at("error_file_id").get<std::string>();
  }
  if (payload.contains("errors")) {
    batch.errors = parse_batch_errors(payload.at("errors"));
  }
  if (payload.contains("expired_at") && payload.at("expired_at").is_number()) {
    batch.expired_at = payload.at("expired_at").get<int>();
  }
  if (payload.contains("expires_at") && payload.at("expires_at").is_number()) {
    batch.expires_at = payload.at("expires_at").get<int>();
  }
  if (payload.contains("failed_at") && payload.at("failed_at").is_number()) {
    batch.failed_at = payload.at("failed_at").get<int>();
  }
  if (payload.contains("finalizing_at") && payload.at("finalizing_at").is_number()) {
    batch.finalizing_at = payload.at("finalizing_at").get<int>();
  }
  if (payload.contains("in_progress_at") && payload.at("in_progress_at").is_number()) {
    batch.in_progress_at = payload.at("in_progress_at").get<int>();
  }
  if (payload.contains("metadata")) {
    batch.metadata = parse_metadata(payload.at("metadata"));
  }
  if (payload.contains("model") && payload.at("model").is_string()) {
    batch.model = payload.at("model").get<std::string>();
  }
  if (payload.contains("output_file_id") && payload.at("output_file_id").is_string()) {
    batch.output_file_id = payload.at("output_file_id").get<std::string>();
  }
  if (payload.contains("request_counts")) {
    batch.request_counts = parse_batch_request_counts(payload.at("request_counts"));
  }
  if (payload.contains("usage")) {
    batch.usage = parse_batch_usage(payload.at("usage"));
  }
  batch.raw = payload;
  return batch;
}

BatchList parse_batch_list(const json& payload) {
  BatchList list;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_batch(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.next_cursor = payload.at("last_id").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  list.raw = payload;
  return list;
}

json batch_create_request_to_json(const BatchCreateRequest& request) {
  json body = json::object();
  body["completion_window"] = request.completion_window;
  body["endpoint"] = request.endpoint;
  body["input_file_id"] = request.input_file_id;
  if (request.metadata) {
    json metadata = json::object();
    for (const auto& [key, value] : *request.metadata) {
      metadata[key] = value;
    }
    body["metadata"] = std::move(metadata);
  }
  if (request.output_expires_after) {
    json expires_after = json::object();
    expires_after["anchor"] = request.output_expires_after->anchor;
    expires_after["seconds"] = request.output_expires_after->seconds;
    body["output_expires_after"] = std::move(expires_after);
  }
  return body;
}

Batch parse_batch_response(const std::string& body) {
  try {
    return parse_batch(json::parse(body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse batch response: ") + ex.what());
  }
}

BatchList parse_batch_list_response(const std::string& body) {
  try {
    return parse_batch_list(json::parse(body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse batch list: ") + ex.what());
  }
}

CursorPage<Batch> make_batch_cursor_page(const PageRequestOptions& request_options,
                                         std::function<CursorPage<Batch>(const PageRequestOptions&)> fetch,
                                         BatchList list) {
  std::optional<std::string> cursor = list.next_cursor;
  if (!cursor && !list.data.empty()) {
    cursor = list.data.back().id;
  }
  return CursorPage<Batch>(
      std::move(list.data),
      list.has_more,
      std::move(cursor),
      request_options,
      std::move(fetch),
      "after",
      std::move(list.raw));
}

}  // namespace

Batch BatchesResource::create(const BatchCreateRequest& request) const {
  return create(request, RequestOptions{});
}

Batch BatchesResource::create(const BatchCreateRequest& request, const RequestOptions& options) const {
  auto body = batch_create_request_to_json(request);
  auto response = client_.perform_request("POST", kBatchesPath, body.dump(), options);
  return parse_batch_response(response.body);
}

Batch BatchesResource::retrieve(const std::string& batch_id) const {
  return retrieve(batch_id, RequestOptions{});
}

Batch BatchesResource::retrieve(const std::string& batch_id, const RequestOptions& options) const {
  auto path = std::string(kBatchesPath) + "/" + batch_id;
  auto response = client_.perform_request("GET", path, "", options);
  return parse_batch_response(response.body);
}

Batch BatchesResource::cancel(const std::string& batch_id) const {
  return cancel(batch_id, RequestOptions{});
}

Batch BatchesResource::cancel(const std::string& batch_id, const RequestOptions& options) const {
  auto path = std::string(kBatchesPath) + "/" + batch_id + "/cancel";
  auto response = client_.perform_request("POST", path, json::object().dump(), options);
  return parse_batch_response(response.body);
}

BatchList BatchesResource::list() const {
  return list(BatchListParams{}, RequestOptions{});
}

BatchList BatchesResource::list(const BatchListParams& params) const {
  return list(params, RequestOptions{});
}

BatchList BatchesResource::list(const RequestOptions& options) const {
  return list(BatchListParams{}, options);
}

BatchList BatchesResource::list(const BatchListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_list_params(params, request_options);
  auto response = client_.perform_request("GET", kBatchesPath, "", request_options);
  return parse_batch_list_response(response.body);
}

CursorPage<Batch> BatchesResource::list_page() const {
  return list_page(BatchListParams{}, RequestOptions{});
}

CursorPage<Batch> BatchesResource::list_page(const BatchListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<Batch> BatchesResource::list_page(const RequestOptions& options) const {
  auto fetch_impl = std::make_shared<std::function<CursorPage<Batch>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<Batch> {
    RequestOptions next = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next);
    auto list = parse_batch_list_response(response.body);
    return make_batch_cursor_page(request_options, *fetch_impl, std::move(list));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kBatchesPath;
  initial.headers = materialize_headers(options);
  initial.query = materialize_query(options);

  return (*fetch_impl)(initial);
}

CursorPage<Batch> BatchesResource::list_page(const BatchListParams& params,
                                             const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_list_params(params, request_options);
  return list_page(request_options);
}

}  // namespace openai

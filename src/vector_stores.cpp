#include "openai/vector_stores.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>
#include <type_traits>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kVectorStoresPath = "/vector_stores";
constexpr const char* kVectorStoreFilesSuffix = "/files";
constexpr const char* kVectorStoreFileBatchesSuffix = "/file_batches";
constexpr const char* kVectorStoreFileBatchCancelSuffix = "/cancel";
constexpr const char* kVectorStoreSearchSuffix = "/search";
constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

std::string operator_to_string(VectorStoreFilter::Comparison::Operator op) {
  using Operator = VectorStoreFilter::Comparison::Operator;
  switch (op) {
    case Operator::Eq:
      return "eq";
    case Operator::Ne:
      return "ne";
    case Operator::Gt:
      return "gt";
    case Operator::Gte:
      return "gte";
    case Operator::Lt:
      return "lt";
    case Operator::Lte:
      return "lte";
    case Operator::In:
      return "in";
    case Operator::Nin:
      return "nin";
  }
  return "eq";
}

std::string compound_operator_to_string(VectorStoreFilter::Compound::Operator op) {
  using Operator = VectorStoreFilter::Compound::Operator;
  return op == Operator::And ? "and" : "or";
}

json comparison_value_to_json(const VectorStoreFilter::Comparison::Value& value) {
  return std::visit(
      [](const auto& v) -> json {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, double> || std::is_same_v<T, bool>) {
          return json(v);
        } else {
          json arr = json::array();
          for (const auto& item : v) {
            std::visit([&arr](const auto& inner) { arr.push_back(inner); }, item);
          }
          return arr;
        }
      },
      value);
}

json filter_to_json(const VectorStoreFilter& filter) {
  return std::visit(
      [](const auto& node) -> json {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, VectorStoreFilter::Comparison>) {
          json payload = json::object();
          payload["type"] = operator_to_string(node.op);
          payload["key"] = node.key;
          payload["value"] = comparison_value_to_json(node.value);
          return payload;
        } else {
          json payload = json::object();
          payload["type"] = compound_operator_to_string(node.op);
          json filters = json::array();
          for (const auto& child : node.filters) {
            filters.push_back(filter_to_json(child));
          }
          payload["filters"] = std::move(filters);
          return payload;
        }
      },
      filter.expression);
}

VectorStoreChunkingStrategy parse_chunking_strategy(const json& payload) {
  VectorStoreChunkingStrategy strategy;
  if (!payload.is_object()) return strategy;

  const auto type = payload.value("type", std::string{});
  if (type == "static") {
    strategy.type = VectorStoreChunkingStrategy::Type::Static;
    if (payload.contains("static") && payload.at("static").is_object()) {
      const auto& params = payload.at("static");
      if (params.contains("chunk_overlap_tokens") && !params.at("chunk_overlap_tokens").is_null()) {
        strategy.chunk_overlap_tokens = params.at("chunk_overlap_tokens").get<int>();
      }
      if (params.contains("max_chunk_size_tokens") && !params.at("max_chunk_size_tokens").is_null()) {
        strategy.max_chunk_size_tokens = params.at("max_chunk_size_tokens").get<int>();
      }
    }
  } else if (type == "other") {
    strategy.type = VectorStoreChunkingStrategy::Type::Other;
  } else {
    strategy.type = VectorStoreChunkingStrategy::Type::Auto;
  }
  return strategy;
}

json chunking_strategy_to_json(const VectorStoreChunkingStrategy& strategy) {
  json payload = json::object();
  switch (strategy.type) {
    case VectorStoreChunkingStrategy::Type::Auto:
      payload["type"] = "auto";
      break;
    case VectorStoreChunkingStrategy::Type::Other:
      payload["type"] = "other";
      break;
    case VectorStoreChunkingStrategy::Type::Static: {
      payload["type"] = "static";
      json params = json::object();
      if (strategy.chunk_overlap_tokens) {
        params["chunk_overlap_tokens"] = *strategy.chunk_overlap_tokens;
      }
      if (strategy.max_chunk_size_tokens) {
        params["max_chunk_size_tokens"] = *strategy.max_chunk_size_tokens;
      }
      payload["static"] = std::move(params);
      break;
    }
  }
  return payload;
}

std::optional<AttributeMap> attributes_from_json(const json& payload) {
  if (payload.is_null()) return std::nullopt;
  if (!payload.is_object()) return AttributeMap{};

  AttributeMap attributes;
  for (const auto& [key, value] : payload.items()) {
    if (value.is_string()) {
      attributes[key] = value.get<std::string>();
    } else if (value.is_boolean()) {
      attributes[key] = value.get<bool>();
    } else if (value.is_number()) {
      attributes[key] = value.get<double>();
    }
  }
  return attributes;
}

json attributes_to_json(const AttributeMap& attributes) {
  json object = json::object();
  for (const auto& [key, value] : attributes) {
    const auto& map_key = key;
    std::visit([&](const auto& entry) { object[map_key] = entry; }, value);
  }
  return object;
}

VectorStore parse_vector_store(const json& payload) {
  VectorStore store;
  store.raw = payload;
  store.id = payload.value("id", "");
  store.created_at = payload.value("created_at", 0);
  store.object = payload.value("object", "");
  store.status = payload.value("status", "");
  store.usage_bytes = payload.value("usage_bytes", static_cast<std::int64_t>(0));

  if (payload.contains("name") && !payload.at("name").is_null()) {
    store.name = payload.at("name").get<std::string>();
  }

  if (payload.contains("last_active_at") && !payload.at("last_active_at").is_null()) {
    store.last_active_at = payload.at("last_active_at").get<int>();
  }

  if (payload.contains("metadata")) {
    auto meta = attributes_from_json(payload.at("metadata"));
    if (meta) {
      Metadata metadata;
      for (const auto& [key, value] : *meta) {
        if (std::holds_alternative<std::string>(value)) {
          metadata[key] = std::get<std::string>(value);
        }
      }
      store.metadata = metadata;
    } else if (payload.at("metadata").is_null()) {
      store.metadata = std::nullopt;
    }
  }

  if (payload.contains("expires_after") && payload.at("expires_after").is_object()) {
    VectorStoreExpiresAfter expires_after;
    expires_after.anchor = payload.at("expires_after").value("anchor", "");
    expires_after.days = payload.at("expires_after").value("days", 0);
    store.expires_after = expires_after;
  }

  if (payload.contains("expires_at") && !payload.at("expires_at").is_null()) {
    store.expires_at = payload.at("expires_at").get<int>();
  }

  if (payload.contains("file_counts") && payload.at("file_counts").is_object()) {
    const auto& counts = payload.at("file_counts");
    store.file_counts.cancelled = counts.value("cancelled", 0);
    store.file_counts.completed = counts.value("completed", 0);
    store.file_counts.failed = counts.value("failed", 0);
    store.file_counts.in_progress = counts.value("in_progress", 0);
    store.file_counts.total = counts.value("total", 0);
  }

  return store;
}

VectorStoreList parse_vector_store_list(const json& payload) {
  VectorStoreList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_vector_store(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("object") && payload.at("object").is_string()) {
    list.object = payload.at("object").get<std::string>();
  }
  return list;
}

VectorStoreDeleted parse_vector_store_deleted(const json& payload) {
  VectorStoreDeleted deleted;
  deleted.raw = payload;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
}

VectorStoreFileLastError parse_last_error(const json& payload) {
  VectorStoreFileLastError error;
  error.code = payload.value("code", "");
  error.message = payload.value("message", "");
  return error;
}

VectorStoreFile parse_vector_store_file(const json& payload) {
  VectorStoreFile file;
  file.raw = payload;
  file.id = payload.value("id", "");
  file.created_at = payload.value("created_at", 0);
  file.object = payload.value("object", "");
  file.status = payload.value("status", "");
  file.usage_bytes = payload.value("usage_bytes", static_cast<std::int64_t>(0));
  file.vector_store_id = payload.value("vector_store_id", "");

  if (payload.contains("last_error") && !payload.at("last_error").is_null()) {
    file.last_error = parse_last_error(payload.at("last_error"));
  }

  if (payload.contains("attributes")) {
    file.attributes = attributes_from_json(payload.at("attributes"));
  }

  if (payload.contains("chunking_strategy") && payload.at("chunking_strategy").is_object()) {
    file.chunking_strategy = parse_chunking_strategy(payload.at("chunking_strategy"));
  }

  return file;
}

VectorStoreFileList parse_vector_store_file_list(const json& payload) {
  VectorStoreFileList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_vector_store_file(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("object") && payload.at("object").is_string()) {
    list.object = payload.at("object").get<std::string>();
  }
  return list;
}

VectorStoreFileDeleted parse_vector_store_file_deleted(const json& payload) {
  VectorStoreFileDeleted deleted;
  deleted.raw = payload;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
}

VectorStoreFileBatchCounts parse_batch_counts(const json& payload) {
  VectorStoreFileBatchCounts counts;
  counts.cancelled = payload.value("cancelled", 0);
  counts.completed = payload.value("completed", 0);
  counts.failed = payload.value("failed", 0);
  counts.in_progress = payload.value("in_progress", 0);
  counts.total = payload.value("total", 0);
  return counts;
}

VectorStoreFileBatch parse_vector_store_file_batch(const json& payload) {
  VectorStoreFileBatch batch;
  batch.raw = payload;
  batch.id = payload.value("id", "");
  batch.created_at = payload.value("created_at", 0);
  batch.object = payload.value("object", "");
  batch.status = payload.value("status", "");
  batch.vector_store_id = payload.value("vector_store_id", "");
  if (payload.contains("file_counts") && payload.at("file_counts").is_object()) {
    batch.file_counts = parse_batch_counts(payload.at("file_counts"));
  }
  return batch;
}

VectorStoreSearchResult parse_search_result(const json& payload) {
  VectorStoreSearchResult result;
  result.raw = payload;
  if (payload.contains("attributes")) {
    result.attributes = attributes_from_json(payload.at("attributes"));
  }
  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& entry : payload.at("content")) {
      VectorStoreSearchResultContent content;
      content.type = entry.value("type", "");
      content.text = entry.value("text", "");
      result.content.push_back(std::move(content));
    }
  }
  result.file_id = payload.value("file_id", "");
  result.filename = payload.value("filename", "");
  result.score = payload.value("score", 0.0);
  return result;
}

VectorStoreSearchResults parse_search_results(const json& payload) {
  VectorStoreSearchResults results;
  results.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      results.data.push_back(parse_search_result(item));
    }
  }
  if (payload.contains("object") && payload.at("object").is_string()) {
    results.object = payload.at("object").get<std::string>();
  }
  return results;
}

json metadata_to_json(const Metadata& metadata) {
  json object = json::object();
  for (const auto& [key, value] : metadata) {
    object[key] = value;
  }
  return object;
}

void apply_vector_store_list_params(const VectorStoreListParams& params, RequestOptions& options) {
  if (params.after) {
    options.query_params["after"] = *params.after;
  }
  if (params.before) {
    options.query_params["before"] = *params.before;
  }
  if (params.limit) {
    options.query_params["limit"] = std::to_string(*params.limit);
  }
  if (params.order) {
    options.query_params["order"] = *params.order;
  }
}

void apply_vector_store_file_list_params(const VectorStoreFileListParams& params, RequestOptions& options) {
  if (params.after) {
    options.query_params["after"] = *params.after;
  }
  if (params.before) {
    options.query_params["before"] = *params.before;
  }
  if (params.limit) {
    options.query_params["limit"] = std::to_string(*params.limit);
  }
  if (params.order) {
    options.query_params["order"] = *params.order;
  }
  if (params.filter) {
    options.query_params["filter"] = *params.filter;
  }
}

void apply_vector_store_file_batch_list_params(const VectorStoreFileBatchListParams& params, RequestOptions& options) {
  if (params.after) {
    options.query_params["after"] = *params.after;
  }
  if (params.before) {
    options.query_params["before"] = *params.before;
  }
  if (params.limit) {
    options.query_params["limit"] = std::to_string(*params.limit);
  }
  if (params.order) {
    options.query_params["order"] = *params.order;
  }
  if (params.filter) {
    options.query_params["filter"] = *params.filter;
  }
}

json build_create_body(const VectorStoreCreateRequest& request) {
  json body = json::object();

  if (request.chunking_strategy) {
    body["chunking_strategy"] = chunking_strategy_to_json(*request.chunking_strategy);
  }
  if (request.description) {
    body["description"] = *request.description;
  }
  if (request.expires_after) {
    body["expires_after"] = {{"anchor", request.expires_after->anchor}, {"days", request.expires_after->days}};
  }
  if (request.file_ids) {
    body["file_ids"] = *request.file_ids;
  }
  if (request.metadata_null) {
    body["metadata"] = nullptr;
  } else if (request.metadata) {
    body["metadata"] = metadata_to_json(*request.metadata);
  }
  if (request.name) {
    body["name"] = *request.name;
  }

  return body;
}

json build_update_body(const VectorStoreUpdateRequest& request) {
  json body = json::object();

  if (request.expires_after) {
    body["expires_after"] = {{"anchor", request.expires_after->anchor}, {"days", request.expires_after->days}};
  }
  if (request.metadata_null) {
    body["metadata"] = nullptr;
  } else if (request.metadata) {
    body["metadata"] = metadata_to_json(*request.metadata);
  }
  if (request.name_null) {
    body["name"] = nullptr;
  } else if (request.name) {
    body["name"] = *request.name;
  }

  return body;
}

json build_file_create_body(const VectorStoreFileCreateRequest& request) {
  json body = json::object();
  body["file_id"] = request.file_id;
  if (request.attributes_null) {
    body["attributes"] = nullptr;
  } else if (request.attributes) {
    body["attributes"] = attributes_to_json(*request.attributes);
  }
  if (request.chunking_strategy) {
    body["chunking_strategy"] = chunking_strategy_to_json(*request.chunking_strategy);
  }
  return body;
}

json build_file_batch_create_body(const VectorStoreFileBatchCreateRequest& request) {
  json body = json::object();
  body["file_ids"] = request.file_ids;
  if (request.attributes_null) {
    body["attributes"] = nullptr;
  } else if (request.attributes) {
    body["attributes"] = attributes_to_json(*request.attributes);
  }
  if (request.chunking_strategy) {
    body["chunking_strategy"] = chunking_strategy_to_json(*request.chunking_strategy);
  }
  return body;
}

json build_search_body(const VectorStoreSearchRequest& request) {
  json body = json::object();
  std::visit([&](const auto& value) { body["query"] = value; }, request.query);
  if (request.filters) {
    body["filters"] = filter_to_json(*request.filters);
  }
  if (request.max_num_results) {
    body["max_num_results"] = *request.max_num_results;
  }
  if (request.ranking_options) {
    json ranking = json::object();
    ranking["ranker"] = request.ranking_options->ranker;
    if (request.ranking_options->score_threshold) {
      ranking["score_threshold"] = *request.ranking_options->score_threshold;
    }
    body["ranking_options"] = std::move(ranking);
  }
  if (request.rewrite_query) {
    body["rewrite_query"] = *request.rewrite_query;
  }
  return body;
}

}  // namespace

VectorStore VectorStoresResource::create(const VectorStoreCreateRequest& request) const {
  return create(request, RequestOptions{});
}

VectorStore VectorStoresResource::create(const VectorStoreCreateRequest& request, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response =
      client_.perform_request("POST", kVectorStoresPath, build_create_body(request).dump(), request_options);
  try {
    return parse_vector_store(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStore VectorStoresResource::retrieve(const std::string& id) const {
  return retrieve(id, RequestOptions{});
}

VectorStore VectorStoresResource::retrieve(const std::string& id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", std::string(kVectorStoresPath) + "/" + id, "", request_options);
  try {
    return parse_vector_store(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStore VectorStoresResource::update(const std::string& id, const VectorStoreUpdateRequest& request) const {
  return update(id, request, RequestOptions{});
}

VectorStore VectorStoresResource::update(const std::string& id,
                                         const VectorStoreUpdateRequest& request,
                                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request(
      "POST", std::string(kVectorStoresPath) + "/" + id, build_update_body(request).dump(), request_options);
  try {
    return parse_vector_store(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStoreList VectorStoresResource::list() const {
  return list(RequestOptions{});
}

VectorStoreList VectorStoresResource::list(const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", kVectorStoresPath, "", request_options);
  try {
    return parse_vector_store_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store list: ") + ex.what());
  }
}

VectorStoreList VectorStoresResource::list(const VectorStoreListParams& params) const {
  return list(params, RequestOptions{});
}

VectorStoreList VectorStoresResource::list(const VectorStoreListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  apply_vector_store_list_params(params, request_options);
  auto response = client_.perform_request("GET", kVectorStoresPath, "", request_options);
  try {
    return parse_vector_store_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store list: ") + ex.what());
  }
}

VectorStoreDeleted VectorStoresResource::remove(const std::string& id) const {
  return remove(id, RequestOptions{});
}

VectorStoreDeleted VectorStoresResource::remove(const std::string& id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("DELETE", std::string(kVectorStoresPath) + "/" + id, "", request_options);
  try {
    return parse_vector_store_deleted(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store delete response: ") + ex.what());
  }
}

VectorStoreFile VectorStoresResource::attach_file(const std::string& vector_store_id,
                                                  const VectorStoreFileCreateRequest& request) const {
  return attach_file(vector_store_id, request, RequestOptions{});
}

VectorStoreFile VectorStoresResource::attach_file(const std::string& vector_store_id,
                                                  const VectorStoreFileCreateRequest& request,
                                                  const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("POST", path, build_file_create_body(request).dump(), request_options);
  try {
    return parse_vector_store_file(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file: ") + ex.what());
  }
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id) const {
  return list_files(vector_store_id, RequestOptions{});
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_vector_store_file_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file list: ") + ex.what());
  }
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id,
                                                     const VectorStoreFileListParams& params) const {
  return list_files(vector_store_id, params, RequestOptions{});
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id,
                                                     const VectorStoreFileListParams& params,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  apply_vector_store_file_list_params(params, request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_vector_store_file_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file list: ") + ex.what());
  }
}

VectorStoreFileDeleted VectorStoresResource::remove_file(const std::string& vector_store_id,
                                                         const std::string& file_id) const {
  return remove_file(vector_store_id, file_id, RequestOptions{});
}

VectorStoreFileDeleted VectorStoresResource::remove_file(const std::string& vector_store_id,
                                                         const std::string& file_id,
                                                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix + "/" + file_id;
  auto response = client_.perform_request("DELETE", path, "", request_options);
  try {
    return parse_vector_store_file_deleted(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file delete response: ") + ex.what());
  }
}

VectorStoreFileBatch VectorStoresResource::create_file_batch(const std::string& vector_store_id,
                                                             const VectorStoreFileBatchCreateRequest& request) const {
  return create_file_batch(vector_store_id, request, RequestOptions{});
}

VectorStoreFileBatch VectorStoresResource::create_file_batch(const std::string& vector_store_id,
                                                             const VectorStoreFileBatchCreateRequest& request,
                                                             const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFileBatchesSuffix;
  auto response =
      client_.perform_request("POST", path, build_file_batch_create_body(request).dump(), request_options);
  try {
    return parse_vector_store_file_batch(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file batch: ") + ex.what());
  }
}

VectorStoreFileBatch VectorStoresResource::retrieve_file_batch(const std::string& vector_store_id,
                                                               const std::string& batch_id) const {
  return retrieve_file_batch(vector_store_id, batch_id, RequestOptions{});
}

VectorStoreFileBatch VectorStoresResource::retrieve_file_batch(const std::string& vector_store_id,
                                                               const std::string& batch_id,
                                                               const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path =
      std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFileBatchesSuffix + "/" + batch_id;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_vector_store_file_batch(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file batch: ") + ex.what());
  }
}

VectorStoreFileBatch VectorStoresResource::cancel_file_batch(const std::string& vector_store_id,
                                                             const std::string& batch_id) const {
  return cancel_file_batch(vector_store_id, batch_id, RequestOptions{});
}

VectorStoreFileBatch VectorStoresResource::cancel_file_batch(const std::string& vector_store_id,
                                                             const std::string& batch_id,
                                                             const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFileBatchesSuffix + "/" +
              batch_id + kVectorStoreFileBatchCancelSuffix;
  auto response = client_.perform_request("POST", path, json::object().dump(), request_options);
  try {
    return parse_vector_store_file_batch(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file batch cancel response: ") + ex.what());
  }
}

VectorStoreFileList VectorStoresResource::list_file_batch_files(const std::string& vector_store_id,
                                                                const std::string& batch_id,
                                                                const VectorStoreFileBatchListParams& params) const {
  return list_file_batch_files(vector_store_id, batch_id, params, RequestOptions{});
}

VectorStoreFileList VectorStoresResource::list_file_batch_files(const std::string& vector_store_id,
                                                                const std::string& batch_id,
                                                                const VectorStoreFileBatchListParams& params,
                                                                const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  apply_vector_store_file_batch_list_params(params, request_options);
  auto path =
      std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFileBatchesSuffix + "/" + batch_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_vector_store_file_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file batch file list: ") + ex.what());
  }
}

VectorStoreSearchResults VectorStoresResource::search(const std::string& vector_store_id,
                                                      const VectorStoreSearchRequest& request) const {
  return search(vector_store_id, request, RequestOptions{});
}

VectorStoreSearchResults VectorStoresResource::search(const std::string& vector_store_id,
                                                      const VectorStoreSearchRequest& request,
                                                      const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreSearchSuffix;
  auto response = client_.perform_request("POST", path, build_search_body(request).dump(), request_options);
  try {
    return parse_search_results(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store search response: ") + ex.what());
  }
}

}  // namespace openai

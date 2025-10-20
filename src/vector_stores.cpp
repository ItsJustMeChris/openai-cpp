#include "openai/vector_stores.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kVectorStoresPath = "/vector_stores";
constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

VectorStore parse_vector_store(const json& payload) {
  VectorStore store;
  store.raw = payload;
  store.id = payload.value("id", "");
  store.name = payload.value("name", "");
  store.object = payload.value("object", "");
  store.created_at = payload.value("created_at", 0);
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) {
        store.metadata[it.key()] = it.value().get<std::string>();
      }
    }
  }
  return store;
}

VectorStoreList parse_vector_store_list(const json& payload) {
  VectorStoreList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_vector_store(item));
    }
  }
  return list;
}

VectorStoreDeleteResponse parse_delete_response(const json& payload) {
  VectorStoreDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  return response;
}

json build_create_body(const VectorStoreCreateRequest& request) {
  json body = request.extra.is_null() ? json::object() : request.extra;
  if (!body.is_object()) {
    throw OpenAIError("VectorStoreCreateRequest.extra must be an object");
  }
  body["name"] = request.name;
  if (!request.metadata.empty()) {
    body["metadata"] = request.metadata;
  }
  return body;
}

json build_update_body(const VectorStoreUpdateRequest& request) {
  json body = request.extra.is_null() ? json::object() : request.extra;
  if (!body.is_object()) {
    throw OpenAIError("VectorStoreUpdateRequest.extra must be an object");
  }
  if (request.name) {
    body["name"] = *request.name;
  }
  if (request.metadata) {
    body["metadata"] = *request.metadata;
  }
  return body;
}

}  // namespace

VectorStore VectorStoresResource::create(const VectorStoreCreateRequest& request,
                                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("POST", kVectorStoresPath, build_create_body(request).dump(), request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStore VectorStoresResource::create(const VectorStoreCreateRequest& request) const {
  return create(request, RequestOptions{});
}

VectorStore VectorStoresResource::retrieve(const std::string& id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", std::string(kVectorStoresPath) + "/" + id, "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStore VectorStoresResource::retrieve(const std::string& id) const {
  return retrieve(id, RequestOptions{});
}

VectorStore VectorStoresResource::update(const std::string& id, const VectorStoreUpdateRequest& request,
                                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response =
      client_.perform_request("POST", std::string(kVectorStoresPath) + "/" + id, build_update_body(request).dump(), request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store: ") + ex.what());
  }
}

VectorStore VectorStoresResource::update(const std::string& id, const VectorStoreUpdateRequest& request) const {
  return update(id, request, RequestOptions{});
}

VectorStoreList VectorStoresResource::list(const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", kVectorStoresPath, "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store_list(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store list: ") + ex.what());
  }
}

VectorStoreList VectorStoresResource::list() const {
  return list(RequestOptions{});
}

VectorStoreDeleteResponse VectorStoresResource::remove(const std::string& id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("DELETE", std::string(kVectorStoresPath) + "/" + id, "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_delete_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store delete response: ") + ex.what());
  }
}

VectorStoreDeleteResponse VectorStoresResource::remove(const std::string& id) const {
  return remove(id, RequestOptions{});
}

}  // namespace openai


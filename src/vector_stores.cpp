#include "openai/vector_stores.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kVectorStoresPath = "/vector_stores";
constexpr const char* kVectorStoreFilesSuffix = "/files";
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

VectorStoreFile parse_vector_store_file(const json& payload) {
  VectorStoreFile file;
  file.raw = payload;
  file.id = payload.value("id", "");
  file.object = payload.value("object", "");
  file.status = payload.value("status", "");
  file.file_id = payload.value("file_id", "");
  return file;
}

VectorStoreFileList parse_vector_store_file_list(const json& payload) {
  VectorStoreFileList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_vector_store_file(item));
    }
  }
  return list;
}

VectorStoreFileDeleteResponse parse_vector_store_file_delete(const json& payload) {
  VectorStoreFileDeleteResponse response;
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

VectorStoreFile VectorStoresResource::attach_file(const std::string& vector_store_id,
                                                  const VectorStoreFileCreateRequest& request,
                                                  const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);

  json body = request.extra.is_null() ? json::object() : request.extra;
  if (!body.is_object()) {
    throw OpenAIError("VectorStoreFileCreateRequest.extra must be an object");
  }
  body["file_id"] = request.file_id;

  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("POST", path, body.dump(), request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store_file(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file: ") + ex.what());
  }
}

VectorStoreFile VectorStoresResource::attach_file(const std::string& vector_store_id,
                                                  const VectorStoreFileCreateRequest& request) const {
  return attach_file(vector_store_id, request, RequestOptions{});
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store_file_list(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file list: ") + ex.what());
  }
}

VectorStoreFileList VectorStoresResource::list_files(const std::string& vector_store_id) const {
  return list_files(vector_store_id, RequestOptions{});
}

VectorStoreFileDeleteResponse VectorStoresResource::remove_file(const std::string& vector_store_id,
                                                               const std::string& file_id,
                                                               const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto path = std::string(kVectorStoresPath) + "/" + vector_store_id + kVectorStoreFilesSuffix + "/" + file_id;
  auto response = client_.perform_request("DELETE", path, "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_vector_store_file_delete(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse vector store file delete response: ") + ex.what());
  }
}

VectorStoreFileDeleteResponse VectorStoresResource::remove_file(const std::string& vector_store_id,
                                                               const std::string& file_id) const {
  return remove_file(vector_store_id, file_id, RequestOptions{});
}

}  // namespace openai

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct VectorStore {
  std::string id;
  std::string name;
  std::string object;
  int created_at = 0;
  std::map<std::string, std::string> metadata;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreList {
  std::vector<VectorStore> data;
  bool has_more = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreCreateRequest {
  std::string name;
  std::map<std::string, std::string> metadata;
  nlohmann::json extra = nlohmann::json::object();
};

struct VectorStoreUpdateRequest {
  std::optional<std::string> name;
  std::optional<std::map<std::string, std::string>> metadata;
  nlohmann::json extra = nlohmann::json::object();
};

struct VectorStoreDeleteResponse {
  std::string id;
  bool deleted = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct RequestOptions;
class OpenAIClient;

class VectorStoresResource {
public:
  explicit VectorStoresResource(OpenAIClient& client) : client_(client) {}

  VectorStore create(const VectorStoreCreateRequest& request) const;
  VectorStore create(const VectorStoreCreateRequest& request, const RequestOptions& options) const;

  VectorStore retrieve(const std::string& id) const;
  VectorStore retrieve(const std::string& id, const RequestOptions& options) const;

  VectorStore update(const std::string& id, const VectorStoreUpdateRequest& request) const;
  VectorStore update(const std::string& id, const VectorStoreUpdateRequest& request,
                     const RequestOptions& options) const;

  VectorStoreList list() const;
  VectorStoreList list(const RequestOptions& options) const;

  VectorStoreDeleteResponse remove(const std::string& id) const;
  VectorStoreDeleteResponse remove(const std::string& id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai


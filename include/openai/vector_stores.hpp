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

struct VectorStoreFile {
  std::string id;
  std::string object;
  std::string status;
  std::string file_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileList {
  std::vector<VectorStoreFile> data;
  bool has_more = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileCreateRequest {
  std::string file_id;
  nlohmann::json extra = nlohmann::json::object();
};

struct VectorStoreFileBatch {
  std::string id;
  std::string object;
  std::string status;
  nlohmann::json file_counts = nlohmann::json::object();
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileBatchCreateRequest {
  std::vector<std::string> file_ids;
  nlohmann::json extra = nlohmann::json::object();
};

struct VectorStoreFileDeleteResponse {
  std::string id;
  bool deleted = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreSearchRequest {
  std::vector<std::string> query;
  std::optional<std::map<std::string, std::string>> metadata_filter;
  std::optional<int> max_num_results;
  struct RankingOptions {
    std::string ranker = "auto";
    std::optional<double> score_threshold;
  };
  std::optional<RankingOptions> ranking_options;
  std::optional<bool> rewrite_query;
};

struct VectorStoreSearchResult {
  std::string file_id;
  std::string filename;
  double score = 0.0;
  std::vector<std::string> content;
  nlohmann::json attributes = nlohmann::json::object();
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreSearchResults {
  std::vector<VectorStoreSearchResult> data;
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

  VectorStoreFile attach_file(const std::string& vector_store_id, const VectorStoreFileCreateRequest& request) const;
  VectorStoreFile attach_file(const std::string& vector_store_id, const VectorStoreFileCreateRequest& request,
                              const RequestOptions& options) const;

  VectorStoreFileList list_files(const std::string& vector_store_id) const;
  VectorStoreFileList list_files(const std::string& vector_store_id, const RequestOptions& options) const;

  VectorStoreFileDeleteResponse remove_file(const std::string& vector_store_id, const std::string& file_id) const;
  VectorStoreFileDeleteResponse remove_file(const std::string& vector_store_id, const std::string& file_id,
                                            const RequestOptions& options) const;

  VectorStoreFileBatch create_file_batch(const std::string& vector_store_id,
                                         const VectorStoreFileBatchCreateRequest& request) const;
  VectorStoreFileBatch create_file_batch(const std::string& vector_store_id,
                                         const VectorStoreFileBatchCreateRequest& request,
                                         const RequestOptions& options) const;

  VectorStoreFileBatch retrieve_file_batch(const std::string& vector_store_id, const std::string& batch_id) const;
  VectorStoreFileBatch retrieve_file_batch(const std::string& vector_store_id, const std::string& batch_id,
                                           const RequestOptions& options) const;

  VectorStoreFileBatch cancel_file_batch(const std::string& vector_store_id, const std::string& batch_id) const;
  VectorStoreFileBatch cancel_file_batch(const std::string& vector_store_id, const std::string& batch_id,
                                         const RequestOptions& options) const;

  VectorStoreSearchResults search(const std::string& vector_store_id, const VectorStoreSearchRequest& request) const;
  VectorStoreSearchResults search(const std::string& vector_store_id,
                                  const VectorStoreSearchRequest& request,
                                  const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

using Metadata = std::map<std::string, std::string>;
using AttributeValue = std::variant<std::string, double, bool>;
using AttributeMap = std::map<std::string, AttributeValue>;
using ComparisonArrayValue = std::variant<std::string, double>;
using ComparisonArray = std::vector<ComparisonArrayValue>;

struct VectorStoreFileCounts {
  int cancelled = 0;
  int completed = 0;
  int failed = 0;
  int in_progress = 0;
  int total = 0;
};

struct VectorStoreExpiresAfter {
  std::string anchor;
  int days = 0;
};

struct VectorStoreChunkingStrategy {
  enum class Type { Auto, Static, Other };
  Type type = Type::Auto;
  std::optional<int> chunk_overlap_tokens;
  std::optional<int> max_chunk_size_tokens;
};

struct VectorStoreFilter {
  struct Comparison {
    enum class Operator { Eq, Ne, Gt, Gte, Lt, Lte, In, Nin };
    std::string key;
    Operator op = Operator::Eq;
    using Value = std::variant<std::string, double, bool, ComparisonArray>;
    Value value;
  };

  struct Compound {
    enum class Operator { And, Or };
    Operator op = Operator::And;
    std::vector<VectorStoreFilter> filters;
  };

  std::variant<Comparison, Compound> expression;
};

struct VectorStore {
  std::string id;
  int created_at = 0;
  VectorStoreFileCounts file_counts;
  std::optional<int> last_active_at;
  std::optional<Metadata> metadata;
  std::optional<std::string> name;
  std::string object;
  std::string status;
  std::int64_t usage_bytes = 0;
  std::optional<VectorStoreExpiresAfter> expires_after;
  std::optional<int> expires_at;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreList {
  std::vector<VectorStore> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  std::optional<std::string> object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileLastError {
  std::string code;
  std::string message;
};

struct VectorStoreFile {
  std::string id;
  int created_at = 0;
  std::optional<VectorStoreFileLastError> last_error;
  std::string object;
  std::string status;
  std::int64_t usage_bytes = 0;
  std::string vector_store_id;
  std::optional<AttributeMap> attributes;
  std::optional<VectorStoreChunkingStrategy> chunking_strategy;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileList {
  std::vector<VectorStoreFile> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  std::optional<std::string> object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreFileBatchCounts {
  int cancelled = 0;
  int completed = 0;
  int failed = 0;
  int in_progress = 0;
  int total = 0;
};

struct VectorStoreFileBatch {
  std::string id;
  int created_at = 0;
  VectorStoreFileBatchCounts file_counts;
  std::string object;
  std::string status;
  std::string vector_store_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreSearchResultContent {
  std::string type;
  std::string text;
};

struct VectorStoreSearchResult {
  std::optional<AttributeMap> attributes;
  std::vector<VectorStoreSearchResultContent> content;
  std::string file_id;
  std::string filename;
  double score = 0.0;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreSearchResults {
  std::vector<VectorStoreSearchResult> data;
  std::optional<std::string> object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VectorStoreCreateRequest {
  std::optional<VectorStoreChunkingStrategy> chunking_strategy;
  std::optional<std::string> description;
  std::optional<VectorStoreExpiresAfter> expires_after;
  std::optional<std::vector<std::string>> file_ids;
  std::optional<Metadata> metadata;
  bool metadata_null = false;
  std::optional<std::string> name;
};

struct VectorStoreUpdateRequest {
  std::optional<VectorStoreExpiresAfter> expires_after;
  std::optional<Metadata> metadata;
  bool metadata_null = false;
  std::optional<std::string> name;
  bool name_null = false;
};

struct VectorStoreFileCreateRequest {
  std::string file_id;
  std::optional<AttributeMap> attributes;
  bool attributes_null = false;
  std::optional<VectorStoreChunkingStrategy> chunking_strategy;
};

struct VectorStoreFileBatchCreateRequest {
  std::vector<std::string> file_ids;
  std::optional<AttributeMap> attributes;
  bool attributes_null = false;
  std::optional<VectorStoreChunkingStrategy> chunking_strategy;
};

struct VectorStoreSearchRequest {
  std::variant<std::string, std::vector<std::string>> query;
  std::optional<VectorStoreFilter> filters;
  std::optional<int> max_num_results;
  struct RankingOptions {
    std::string ranker = "auto";
    std::optional<double> score_threshold;
  };
  std::optional<RankingOptions> ranking_options;
  std::optional<bool> rewrite_query;
};

struct VectorStoreListParams {
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<int> limit;
  std::optional<std::string> order;
};

struct VectorStoreFileListParams {
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> filter;
  std::optional<int> limit;
  std::optional<std::string> order;
};

struct VectorStoreFileBatchListParams {
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> filter;
  std::optional<int> limit;
  std::optional<std::string> order;
};

class VectorStoresResource {
public:
  explicit VectorStoresResource(OpenAIClient& client) : client_(client) {}

  VectorStore create(const VectorStoreCreateRequest& request) const;
  VectorStore create(const VectorStoreCreateRequest& request, const RequestOptions& options) const;

  VectorStore retrieve(const std::string& id) const;
  VectorStore retrieve(const std::string& id, const RequestOptions& options) const;

  VectorStore update(const std::string& id, const VectorStoreUpdateRequest& request) const;
  VectorStore update(const std::string& id, const VectorStoreUpdateRequest& request, const RequestOptions& options) const;

  VectorStoreList list() const;
  VectorStoreList list(const RequestOptions& options) const;
  VectorStoreList list(const VectorStoreListParams& params) const;
  VectorStoreList list(const VectorStoreListParams& params, const RequestOptions& options) const;

  VectorStoreDeleted remove(const std::string& id) const;
  VectorStoreDeleted remove(const std::string& id, const RequestOptions& options) const;

  VectorStoreFile attach_file(const std::string& vector_store_id, const VectorStoreFileCreateRequest& request) const;
  VectorStoreFile attach_file(const std::string& vector_store_id,
                              const VectorStoreFileCreateRequest& request,
                              const RequestOptions& options) const;

  VectorStoreFileList list_files(const std::string& vector_store_id) const;
  VectorStoreFileList list_files(const std::string& vector_store_id, const RequestOptions& options) const;
  VectorStoreFileList list_files(const std::string& vector_store_id, const VectorStoreFileListParams& params) const;
  VectorStoreFileList list_files(const std::string& vector_store_id,
                                 const VectorStoreFileListParams& params,
                                 const RequestOptions& options) const;

  VectorStoreFileDeleted remove_file(const std::string& vector_store_id, const std::string& file_id) const;
  VectorStoreFileDeleted remove_file(const std::string& vector_store_id,
                                     const std::string& file_id,
                                     const RequestOptions& options) const;

  VectorStoreFileBatch create_file_batch(const std::string& vector_store_id,
                                         const VectorStoreFileBatchCreateRequest& request) const;
  VectorStoreFileBatch create_file_batch(const std::string& vector_store_id,
                                         const VectorStoreFileBatchCreateRequest& request,
                                         const RequestOptions& options) const;

  VectorStoreFileBatch retrieve_file_batch(const std::string& vector_store_id, const std::string& batch_id) const;
  VectorStoreFileBatch retrieve_file_batch(const std::string& vector_store_id,
                                           const std::string& batch_id,
                                           const RequestOptions& options) const;

  VectorStoreFileBatch cancel_file_batch(const std::string& vector_store_id, const std::string& batch_id) const;
  VectorStoreFileBatch cancel_file_batch(const std::string& vector_store_id,
                                         const std::string& batch_id,
                                         const RequestOptions& options) const;

  VectorStoreFileList list_file_batch_files(const std::string& vector_store_id,
                                            const std::string& batch_id,
                                            const VectorStoreFileBatchListParams& params) const;
  VectorStoreFileList list_file_batch_files(const std::string& vector_store_id,
                                            const std::string& batch_id,
                                            const VectorStoreFileBatchListParams& params,
                                            const RequestOptions& options) const;

  VectorStoreSearchResults search(const std::string& vector_store_id, const VectorStoreSearchRequest& request) const;
  VectorStoreSearchResults search(const std::string& vector_store_id,
                                  const VectorStoreSearchRequest& request,
                                  const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai


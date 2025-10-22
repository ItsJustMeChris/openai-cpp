#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

struct BatchRequestCounts {
  int completed = 0;
  int failed = 0;
  int total = 0;
};

struct BatchUsageInputTokensDetails {
  int cached_tokens = 0;
};

struct BatchUsageOutputTokensDetails {
  int reasoning_tokens = 0;
};

struct BatchUsage {
  int input_tokens = 0;
  BatchUsageInputTokensDetails input_tokens_details;
  int output_tokens = 0;
  BatchUsageOutputTokensDetails output_tokens_details;
  int total_tokens = 0;
};

struct BatchError {
  std::optional<std::string> code;
  std::optional<int> line;
  std::optional<std::string> message;
  std::optional<std::string> param;
};

struct BatchErrors {
  std::vector<BatchError> data;
  std::optional<std::string> object;
};

struct Batch {
  std::string id;
  std::string completion_window;
  int created_at = 0;
  std::string endpoint;
  std::string input_file_id;
  std::string object;
  std::string status;
  std::optional<int> cancelled_at;
  std::optional<int> cancelling_at;
  std::optional<int> completed_at;
  std::optional<std::string> error_file_id;
  std::optional<BatchErrors> errors;
  std::optional<int> expired_at;
  std::optional<int> expires_at;
  std::optional<int> failed_at;
  std::optional<int> finalizing_at;
  std::optional<int> in_progress_at;
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<std::string> model;
  std::optional<std::string> output_file_id;
  std::optional<BatchRequestCounts> request_counts;
  std::optional<BatchUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct BatchList {
  std::vector<Batch> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct BatchCreateRequest {
  struct OutputExpiresAfter {
    std::string anchor;
    int seconds = 0;
  };

  std::string completion_window;
  std::string endpoint;
  std::string input_file_id;
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<OutputExpiresAfter> output_expires_after;
};

struct BatchListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
};

class BatchesResource {
public:
  explicit BatchesResource(OpenAIClient& client) : client_(client) {}

  Batch create(const BatchCreateRequest& request) const;
  Batch create(const BatchCreateRequest& request, const RequestOptions& options) const;

  Batch retrieve(const std::string& batch_id) const;
  Batch retrieve(const std::string& batch_id, const RequestOptions& options) const;

  Batch cancel(const std::string& batch_id) const;
  Batch cancel(const std::string& batch_id, const RequestOptions& options) const;

  BatchList list() const;
  BatchList list(const BatchListParams& params) const;
  BatchList list(const RequestOptions& options) const;
  BatchList list(const BatchListParams& params, const RequestOptions& options) const;

  CursorPage<Batch> list_page() const;
  CursorPage<Batch> list_page(const BatchListParams& params) const;
  CursorPage<Batch> list_page(const RequestOptions& options) const;
  CursorPage<Batch> list_page(const BatchListParams& params, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai


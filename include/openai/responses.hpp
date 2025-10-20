#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/streaming.hpp"

namespace openai {

struct ResponseUsage {
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseOutputTextSegment {
  std::string text;
};

struct ResponseOutputMessage {
  std::string role;
  std::vector<ResponseOutputTextSegment> text_segments;
};

struct Response {
  std::string id;
  std::string object;
  int created = 0;
  std::string model;
  std::vector<ResponseOutputMessage> messages;
  std::string output_text;
  std::optional<ResponseUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputContent {
  std::string text;
};

struct ResponseInput {
  std::string role;
  std::vector<ResponseInputContent> content;
};

struct ResponseRequest {
  std::string model;
  std::vector<ResponseInput> input;
  std::map<std::string, std::string> metadata;
};

struct ResponseRetrieveOptions {
  bool stream = false;
};

struct ResponseList {
  std::vector<Response> data;
  bool has_more = false;
  nlohmann::json raw = nlohmann::json::object();
};

class OpenAIClient;

class ResponsesResource {
public:
  explicit ResponsesResource(OpenAIClient& client) : client_(client) {}

  Response create(const ResponseRequest& request) const;
  Response create(const ResponseRequest& request, const struct RequestOptions& options) const;

  Response retrieve(const std::string& response_id) const;
  Response retrieve(const std::string& response_id,
                    const ResponseRetrieveOptions& retrieve_options,
                    const struct RequestOptions& options) const;

  void remove(const std::string& response_id) const;
  void remove(const std::string& response_id, const struct RequestOptions& options) const;

  Response cancel(const std::string& response_id) const;
  Response cancel(const std::string& response_id, const struct RequestOptions& options) const;

  ResponseList list() const;
  ResponseList list(const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> create_stream(const ResponseRequest& request) const;
  std::vector<ServerSentEvent> create_stream(const ResponseRequest& request,
                                             const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> retrieve_stream(const std::string& response_id) const;
  std::vector<ServerSentEvent> retrieve_stream(const std::string& response_id,
                                               const ResponseRetrieveOptions& retrieve_options,
                                               const struct RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

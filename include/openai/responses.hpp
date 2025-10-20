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
  enum class Type { Text, Image, File, Audio, Raw };

  Type type = Type::Text;
  std::string text;
  std::string image_url;
  std::string image_detail;
  std::string file_id;
  std::string file_url;
  std::string filename;
  std::string audio_data;
  std::string audio_format;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInput {
  std::string role;
  std::vector<ResponseInputContent> content;
  std::map<std::string, std::string> metadata;
};

struct ResponsePrompt {
  std::string id;
  std::map<std::string, std::string> variables;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseReasoningConfig {
  std::optional<std::string> effort;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseStreamOptions {
  std::optional<bool> include_usage;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseRequest {
  std::string model;
  std::vector<ResponseInput> input;
  std::map<std::string, std::string> metadata;
  std::optional<bool> background;
  std::optional<std::string> conversation_id;
  std::vector<std::string> include;
  std::optional<std::string> instructions;
  std::optional<int> max_output_tokens;
  std::optional<bool> parallel_tool_calls;
  std::optional<std::string> previous_response_id;
  std::optional<ResponsePrompt> prompt;
  std::optional<std::string> prompt_cache_key;
  std::optional<ResponseReasoningConfig> reasoning;
  std::optional<std::string> safety_identifier;
  std::optional<std::string> service_tier;
  std::optional<bool> store;
  std::optional<bool> stream;
  std::optional<ResponseStreamOptions> stream_options;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::vector<nlohmann::json> tools;
  std::optional<nlohmann::json> tool_choice;
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

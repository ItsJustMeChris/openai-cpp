#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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

struct ResponseRequest {
  nlohmann::json body = nlohmann::json::object();
};

struct ResponseRetrieveOptions {
  bool stream = false;
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

private:
  OpenAIClient& client_;
};

}  // namespace openai

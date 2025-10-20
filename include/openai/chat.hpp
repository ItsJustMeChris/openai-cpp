#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/streaming.hpp"

namespace openai {

struct ChatMessageContent {
  enum class Type {
    Text,
    Image,
    File,
    Audio,
    Raw
  };

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

struct ChatCompletionToolCall {
  std::string id;
  std::string type;
  nlohmann::json function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatMessage {
  std::string role;
  std::vector<ChatMessageContent> content;
  std::optional<std::string> name;
  std::vector<ChatCompletionToolCall> tool_calls;
  std::map<std::string, std::string> metadata;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  nlohmann::json extra = nlohmann::json::object();
};

struct ChatCompletionChoice {
  int index = 0;
  std::optional<ChatMessage> message;
  std::optional<std::string> finish_reason;
  nlohmann::json logprobs = nlohmann::json();
  nlohmann::json extra = nlohmann::json::object();
};

struct ChatCompletion {
  std::string id;
  std::string object;
  int created = 0;
  std::string model;
  std::optional<std::string> system_fingerprint;
  std::vector<ChatCompletionChoice> choices;
  std::optional<ChatCompletionUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatResponseFormat {
  std::string type;
  nlohmann::json json_schema = nlohmann::json::object();
};

struct ChatToolFunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  nlohmann::json parameters = nlohmann::json::object();
};

struct ChatCompletionToolDefinition {
  std::string type;
  std::optional<ChatToolFunctionDefinition> function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolChoice {
  std::string type;
  std::optional<std::string> function_name;
  nlohmann::json raw = nlohmann::json::object();
};

struct RequestOptions;

struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::map<std::string, std::string> metadata;
  std::optional<int> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<double> frequency_penalty;
  std::optional<double> presence_penalty;
  std::map<std::string, double> logit_bias;
  std::optional<bool> logprobs;
  std::optional<int> top_logprobs;
  std::optional<std::vector<std::string>> stop;
  std::optional<int64_t> seed;
  std::optional<ChatResponseFormat> response_format;
  std::vector<ChatCompletionToolDefinition> tools;
  std::optional<ChatToolChoice> tool_choice;
  std::optional<bool> parallel_tool_calls;
  std::optional<std::string> user;
  std::optional<bool> stream;
};

class OpenAIClient;

class ChatCompletionsResource {
public:
  explicit ChatCompletionsResource(OpenAIClient& client) : client_(client) {}

  ChatCompletion create(const ChatCompletionRequest& request) const;
  ChatCompletion create(const ChatCompletionRequest& request,
                        const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> create_stream(const ChatCompletionRequest& request) const;
  std::vector<ServerSentEvent> create_stream(const ChatCompletionRequest& request,
                                             const struct RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class ChatResource {
public:
  explicit ChatResource(OpenAIClient& client) : client_(client), completions_(client) {}

  ChatCompletionsResource& completions() { return completions_; }
  const ChatCompletionsResource& completions() const { return completions_; }

private:
  OpenAIClient& client_;
  ChatCompletionsResource completions_;
};

}  // namespace openai


#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct ChatMessageContent {
  std::string type;
  nlohmann::json data;
  std::optional<std::string> text;
};

struct ChatCompletionToolCall {
  std::string id;
  std::string type;
  nlohmann::json function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatMessage {
  std::string role;
  std::variant<std::monostate, std::string, std::vector<ChatMessageContent>> content;
  std::optional<std::string> name;
  std::vector<ChatCompletionToolCall> tool_calls;
  nlohmann::json metadata = nlohmann::json::object();
  nlohmann::json extra_fields = nlohmann::json::object();
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

struct RequestOptions;

struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::optional<bool> stream;
  nlohmann::json extra_params = nlohmann::json::object();
};

class OpenAIClient;

class ChatCompletionsResource {
public:
  explicit ChatCompletionsResource(OpenAIClient& client) : client_(client) {}

  ChatCompletion create(const ChatCompletionRequest& request) const;
  ChatCompletion create(const ChatCompletionRequest& request,
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

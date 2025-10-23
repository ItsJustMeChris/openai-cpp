#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/streaming.hpp"

namespace openai {

template <typename Item>
class CursorPage;

struct ChatMessageContent {
  enum class Type {
    Text,
    Image,
    File,
    Audio,
    Refusal,
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
  std::string refusal_text;
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
  std::optional<std::string> id;
  std::optional<std::string> tool_call_id;
  std::vector<ChatMessageContent> content;
  std::optional<std::string> name;
  std::vector<ChatCompletionToolCall> tool_calls;
  std::map<std::string, std::string> metadata;
  std::optional<std::string> refusal;
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
  std::optional<std::string> service_tier;
  std::map<std::string, std::string> metadata;
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

struct ChatCompletionStreamOptions {
  std::optional<bool> include_obfuscation;
  std::optional<bool> include_usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::map<std::string, std::string> metadata;
  std::optional<int> max_tokens;
  std::optional<int> max_completion_tokens;
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
  std::optional<bool> store;
  std::optional<ChatCompletionStreamOptions> stream_options;
  std::vector<std::string> modalities;
  std::optional<std::string> service_tier;
};

class OpenAIClient;

struct ChatCompletionUpdateRequest {
  std::optional<std::map<std::string, std::string>> metadata;
  bool clear_metadata = false;
};

struct ChatCompletionListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> model;
  std::optional<std::map<std::string, std::string>> metadata;
};

struct ChatCompletionList {
  std::vector<ChatCompletion> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionStoreMessage {
  ChatMessage message;
  std::string id;
  std::vector<ChatMessageContent> content_parts;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionMessageListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
};

struct ChatCompletionStoreMessageList {
  std::vector<ChatCompletionStoreMessage> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolFunctionHandler {
  ChatCompletionToolDefinition definition;
  std::function<nlohmann::json(const nlohmann::json&)> callback;
};

struct ChatCompletionToolRunParams {
  ChatCompletionRequest request;
  std::vector<ChatToolFunctionHandler> functions;
  std::size_t max_iterations = 10;
};

struct ChatCompletionToolRunResult {
  ChatCompletion final_completion;
  std::vector<ChatCompletion> completions;
  std::vector<ChatMessage> transcript;
};

class ChatCompletionsResource {
public:
  class MessagesResource {
  public:
    explicit MessagesResource(OpenAIClient& client) : client_(client) {}

    ChatCompletionStoreMessageList list(const std::string& completion_id) const;
    ChatCompletionStoreMessageList list(const std::string& completion_id,
                                        const ChatCompletionMessageListParams& params) const;
    ChatCompletionStoreMessageList list(const std::string& completion_id,
                                        const ChatCompletionMessageListParams& params,
                                        const struct RequestOptions& options) const;

    CursorPage<ChatCompletionStoreMessage> list_page(const std::string& completion_id) const;
    CursorPage<ChatCompletionStoreMessage> list_page(const std::string& completion_id,
                                                     const ChatCompletionMessageListParams& params) const;
    CursorPage<ChatCompletionStoreMessage> list_page(const std::string& completion_id,
                                                     const ChatCompletionMessageListParams& params,
                                                     const struct RequestOptions& options) const;

  private:
    OpenAIClient& client_;
  };

  explicit ChatCompletionsResource(OpenAIClient& client) : client_(client), messages_(client) {}

  ChatCompletion create(const ChatCompletionRequest& request) const;
  ChatCompletion create(const ChatCompletionRequest& request,
                        const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> create_stream(const ChatCompletionRequest& request) const;
  std::vector<ServerSentEvent> create_stream(const ChatCompletionRequest& request,
                                             const struct RequestOptions& options) const;
  void create_stream(const ChatCompletionRequest& request,
                     const std::function<bool(const ServerSentEvent&)>& on_event) const;
  void create_stream(const ChatCompletionRequest& request,
                     const std::function<bool(const ServerSentEvent&)>& on_event,
                     const struct RequestOptions& options) const;

  ChatCompletion retrieve(const std::string& completion_id) const;
  ChatCompletion retrieve(const std::string& completion_id, const struct RequestOptions& options) const;

  ChatCompletion update(const std::string& completion_id, const ChatCompletionUpdateRequest& request) const;
  ChatCompletion update(const std::string& completion_id,
                        const ChatCompletionUpdateRequest& request,
                        const struct RequestOptions& options) const;

  ChatCompletionList list() const;
  ChatCompletionList list(const ChatCompletionListParams& params) const;
  ChatCompletionList list(const ChatCompletionListParams& params, const struct RequestOptions& options) const;
  CursorPage<ChatCompletion> list_page() const;
  CursorPage<ChatCompletion> list_page(const ChatCompletionListParams& params) const;
  CursorPage<ChatCompletion> list_page(const ChatCompletionListParams& params,
                                       const struct RequestOptions& options) const;

  ChatCompletionDeleted remove(const std::string& completion_id) const;
  ChatCompletionDeleted remove(const std::string& completion_id, const struct RequestOptions& options) const;

  ChatCompletionToolRunResult run_tools(const ChatCompletionToolRunParams& params) const;
  ChatCompletionToolRunResult run_tools(const ChatCompletionToolRunParams& params,
                                        const struct RequestOptions& options) const;

  MessagesResource& messages() { return messages_; }
  const MessagesResource& messages() const { return messages_; }

private:
  OpenAIClient& client_;
  MessagesResource messages_;
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

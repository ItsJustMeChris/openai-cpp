#pragma once

#include <cstdint>
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

using json = nlohmann::json;

struct ChatMessageContent {
  struct FilePayload {
    std::optional<std::string> file_data;
    std::optional<std::string> file_id;
    std::optional<std::string> filename;
  };

  struct InputAudioPayload {
    std::string data;
    std::string format;
  };

  enum class Type {
    Text,
    Image,
    File,
    InputAudio,
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
  std::optional<std::string> detail;
  std::optional<std::string> format;
  std::optional<FilePayload> file;
  std::optional<InputAudioPayload> input_audio;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatMessageAnnotationURLCitation {
  int start_index = 0;
  int end_index = 0;
  std::string title;
  std::string url;
};

struct ChatMessageAnnotation {
  std::string type;
  std::optional<ChatMessageAnnotationURLCitation> url_citation;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionAudio {
  std::string id;
  std::string data;
  std::int64_t expires_at = 0;
  std::string transcript;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionAudioParam {
  std::string format;
  std::string voice;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatFunctionCall {
  std::string name;
  std::string arguments;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionMessageCustomToolCallPayload {
  std::string name;
  std::string input;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionMessageCustomToolCall {
  std::string id;
  ChatCompletionMessageCustomToolCallPayload custom;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionMessageFunctionToolCall {
  std::string id;
  ChatFunctionCall function;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionMessageToolCall {
  enum class Type { Function, Custom };

  Type type = Type::Function;
  std::optional<ChatCompletionMessageFunctionToolCall> function_call;
  std::optional<ChatCompletionMessageCustomToolCall> custom_call;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionToolCall {
  std::string id;
  std::string type;
  nlohmann::json function;
  std::optional<ChatFunctionCall> parsed_function;
  std::optional<ChatCompletionMessageCustomToolCallPayload> custom;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatMessage {
  std::string role;
  std::optional<std::string> id;
  std::optional<std::string> tool_call_id;
  std::vector<ChatMessageContent> content;
  std::optional<std::string> name;
  std::vector<ChatMessageAnnotation> annotations;
  std::optional<ChatCompletionAudio> audio;
  std::optional<ChatFunctionCall> function_call;
  std::vector<ChatCompletionToolCall> tool_calls;
  std::vector<ChatCompletionMessageToolCall> structured_tool_calls;
  std::map<std::string, std::string> metadata;
  std::optional<std::string> refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionTokenLogprob {
  std::string token;
  std::vector<int> bytes;
  double logprob = 0.0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionLogprobs {
  std::vector<ChatCompletionTokenLogprob> content;
  std::vector<ChatCompletionTokenLogprob> refusal;
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
  std::optional<ChatCompletionLogprobs> logprobs;
  nlohmann::json raw_logprobs = nlohmann::json();
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

struct ChatCompletionPredictionContent {
  std::variant<std::monostate, std::string, std::vector<ChatMessageContent>> content;
  std::optional<std::string> type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolFunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  json parameters;
};

struct ChatCompletionFunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  json parameters;
};

struct ChatCompletionFunctionCallOption {
  std::string name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionFunctionCallDirective {
  enum class Type { None, Auto, Function };

  Type type = Type::Auto;
  std::optional<ChatCompletionFunctionCallOption> function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolCustomGrammarContent {
  std::string definition;
  std::string syntax;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolCustomGrammarDefinition {
  std::string type;
  ChatToolCustomGrammarContent grammar;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolCustomFormat {
  std::string type;
  std::optional<ChatToolCustomGrammarDefinition> grammar;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolCustomDefinition {
  std::string name;
  std::optional<std::string> description;
  std::optional<ChatToolCustomFormat> format;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionToolDefinition {
  std::string type;
  std::optional<ChatToolFunctionDefinition> function;
  std::optional<ChatToolCustomDefinition> custom;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionChunkToolCallFunctionDelta {
  std::optional<std::string> name;
  std::optional<std::string> arguments;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionChunkToolCallDelta {
  int index = 0;
  std::optional<std::string> id;
  std::optional<std::string> type;
  std::optional<ChatCompletionChunkToolCallFunctionDelta> function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionChunkDelta {
  std::optional<std::string> content;
  std::optional<ChatFunctionCall> function_call;
  std::optional<std::string> refusal;
  std::optional<std::string> role;
  std::vector<ChatCompletionChunkToolCallDelta> tool_calls;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionChunkChoiceLogprobs {
  std::vector<ChatCompletionTokenLogprob> content;
  std::vector<ChatCompletionTokenLogprob> refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionChunkChoice {
  ChatCompletionChunkDelta delta;
  std::optional<std::string> finish_reason;
  int index = 0;
  std::optional<ChatCompletionChunkChoiceLogprobs> logprobs;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionAllowedTools {
  std::string mode;
  std::vector<nlohmann::json> tools;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionAllowedToolChoice {
  ChatCompletionAllowedTools allowed_tools;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionNamedToolChoiceFunction {
  std::string name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionNamedToolChoice {
  std::string type;
  ChatCompletionNamedToolChoiceFunction function;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionNamedToolChoiceCustomValue {
  std::string name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionNamedToolChoiceCustom {
  std::string type;
  ChatCompletionNamedToolChoiceCustomValue custom;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionWebSearchApproximateLocation {
  std::optional<std::string> city;
  std::optional<std::string> country;
  std::optional<std::string> region;
  std::optional<std::string> timezone;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionWebSearchUserLocation {
  ChatCompletionWebSearchApproximateLocation approximate;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatCompletionWebSearchOptions {
  std::optional<std::string> search_context_size;
  std::optional<ChatCompletionWebSearchUserLocation> user_location;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatToolChoice {
  enum class Type { None, Auto, Required, AllowedTools, NamedFunction, NamedCustom };

  Type type = Type::Auto;
  std::optional<ChatCompletionAllowedToolChoice> allowed_tools;
  std::optional<ChatCompletionNamedToolChoice> named_function;
  std::optional<ChatCompletionNamedToolChoiceCustom> named_custom;
  std::optional<std::string> literal;
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
  std::optional<ChatCompletionAudioParam> audio;
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
  std::optional<ChatCompletionFunctionCallDirective> function_call;
  std::vector<ChatCompletionFunctionDefinition> functions;
  std::optional<ChatResponseFormat> response_format;
  std::vector<ChatCompletionToolDefinition> tools;
  std::optional<ChatToolChoice> tool_choice;
  std::optional<std::string> prompt_cache_key;
  std::optional<std::string> reasoning_effort;
  std::optional<bool> parallel_tool_calls;
  std::optional<ChatCompletionPredictionContent> prediction;
  std::optional<std::string> user;
  std::optional<std::string> safety_identifier;
  std::optional<int> n;
  std::optional<bool> stream;
  std::optional<bool> store;
  std::optional<ChatCompletionStreamOptions> stream_options;
  std::vector<std::string> modalities;
  std::optional<std::string> verbosity;
  std::optional<ChatCompletionWebSearchOptions> web_search_options;
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

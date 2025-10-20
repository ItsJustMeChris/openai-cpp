#include "openai/chat.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/streaming.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

json message_content_to_json(const ChatMessageContent& content) {
  json block = content.data.is_null() ? json::object() : content.data;
  if (!block.is_object()) {
    block = json::object();
  }
  block["type"] = content.type;
  if (content.text) {
    block["text"] = *content.text;
  }
  return block;
}

json message_to_json(const ChatMessage& message) {
  json result = message.extra_fields.is_null() ? json::object() : message.extra_fields;
  if (!result.is_object()) {
    throw OpenAIError("ChatMessage.extra_fields must be an object");
  }

  result["role"] = message.role;

  if (std::holds_alternative<std::string>(message.content)) {
    result["content"] = std::get<std::string>(message.content);
  } else if (std::holds_alternative<std::vector<ChatMessageContent>>(message.content)) {
    json content_array = json::array();
    for (const auto& block : std::get<std::vector<ChatMessageContent>>(message.content)) {
      content_array.push_back(message_content_to_json(block));
    }
    result["content"] = std::move(content_array);
  }

  if (message.name) {
    result["name"] = *message.name;
  }

  if (!message.tool_calls.empty()) {
    json tool_calls = json::array();
    for (const auto& call : message.tool_calls) {
      json tool_call = call.raw.is_null() ? json::object() : call.raw;
      if (!tool_call.is_object()) {
        tool_call = json::object();
      }
      if (!call.id.empty()) {
        tool_call["id"] = call.id;
      }
      if (!call.type.empty()) {
        tool_call["type"] = call.type;
      }
      if (!call.function.is_null()) {
        tool_call["function"] = call.function;
      }
      tool_calls.push_back(std::move(tool_call));
    }
    result["tool_calls"] = std::move(tool_calls);
  }

  if (!message.metadata.is_null() && !message.metadata.empty()) {
    result["metadata"] = message.metadata;
  }

  return result;
}

ChatCompletionToolCall parse_tool_call(const json& payload) {
  ChatCompletionToolCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.type = payload.value("type", "");
  if (payload.contains("function")) {
    call.function = payload.at("function");
  }
  return call;
}

ChatMessageContent parse_message_content(const json& payload) {
  ChatMessageContent content;
  content.data = payload;
  content.type = payload.value("type", "");
  if (payload.contains("text") && payload["text"].is_string()) {
    content.text = payload["text"].get<std::string>();
  }
  return content;
}

ChatMessage parse_message(const json& payload) {
  ChatMessage message;
  message.raw = payload;
  message.extra_fields = json::object();

  message.role = payload.value("role", "");
  if (payload.contains("name") && payload["name"].is_string()) {
    message.name = payload["name"].get<std::string>();
  }
  if (payload.contains("metadata")) {
    message.metadata = payload.at("metadata");
  }

  if (payload.contains("content")) {
    const auto& content = payload.at("content");
    if (content.is_string()) {
      message.content = content.get<std::string>();
    } else if (content.is_array()) {
      std::vector<ChatMessageContent> blocks;
      for (const auto& block_json : content) {
        blocks.push_back(parse_message_content(block_json));
      }
      message.content = std::move(blocks);
    }
  }

  if (payload.contains("tool_calls") && payload["tool_calls"].is_array()) {
    for (const auto& call_json : payload.at("tool_calls")) {
      message.tool_calls.push_back(parse_tool_call(call_json));
    }
  }

  return message;
}

ChatCompletionUsage parse_usage(const json& payload) {
  ChatCompletionUsage usage;
  usage.extra = payload;
  usage.prompt_tokens = payload.value("prompt_tokens", 0);
  usage.completion_tokens = payload.value("completion_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  return usage;
}

ChatCompletionChoice parse_choice(const json& payload) {
  ChatCompletionChoice choice;
  choice.extra = payload;
  choice.index = payload.value("index", 0);
  if (payload.contains("message")) {
    choice.message = parse_message(payload.at("message"));
  }
  if (payload.contains("finish_reason") && !payload["finish_reason"].is_null()) {
    choice.finish_reason = payload["finish_reason"].get<std::string>();
  }
  if (payload.contains("logprobs")) {
    choice.logprobs = payload.at("logprobs");
  }
  return choice;
}

ChatCompletion parse_chat_completion(const json& payload) {
  ChatCompletion completion;
  completion.raw = payload;
  completion.id = payload.value("id", "");
  completion.object = payload.value("object", "");
  completion.created = payload.value("created", 0);
  completion.model = payload.value("model", "");
  if (payload.contains("system_fingerprint") && payload["system_fingerprint"].is_string()) {
    completion.system_fingerprint = payload["system_fingerprint"].get<std::string>();
  }

  if (payload.contains("choices")) {
    for (const auto& choice_json : payload.at("choices")) {
      completion.choices.push_back(parse_choice(choice_json));
    }
  }

  if (payload.contains("usage")) {
    completion.usage = parse_usage(payload.at("usage"));
  }

  return completion;
}

}  // namespace

ChatCompletion ChatCompletionsResource::create(const ChatCompletionRequest& request) const {
  return create(request, RequestOptions{});
}

ChatCompletion ChatCompletionsResource::create(const ChatCompletionRequest& request,
                                               const RequestOptions& options) const {
  json body = request.extra_params.is_null() ? json::object() : request.extra_params;
  if (!body.is_object()) {
    throw OpenAIError("ChatCompletionRequest.extra_params must be an object");
  }

  body["model"] = request.model;

  json messages = json::array();
  for (const auto& message : request.messages) {
    messages.push_back(message_to_json(message));
  }
  body["messages"] = std::move(messages);

  if (request.stream.has_value()) {
    body["stream"] = *request.stream;
  }

  auto response = client_.perform_request("POST", "/chat/completions", body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_chat_completion(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion response: ") + ex.what());
  }
}

std::vector<ServerSentEvent> ChatCompletionsResource::create_stream(const ChatCompletionRequest& request,
                                                                    const RequestOptions& options) const {
  json body = request.extra_params.is_null() ? json::object() : request.extra_params;
  if (!body.is_object()) {
    throw OpenAIError("ChatCompletionRequest.extra_params must be an object");
  }

  body["model"] = request.model;

  json messages = json::array();
  for (const auto& message : request.messages) {
    messages.push_back(message_to_json(message));
  }
  body["messages"] = std::move(messages);
  body["stream"] = true;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;

  SSEParser parser;
  std::vector<ServerSentEvent> events;
  request_options.on_chunk = [&](const char* data, std::size_t size) {
    auto chunk_events = parser.feed(data, size);
    events.insert(events.end(), chunk_events.begin(), chunk_events.end());
  };

  client_.perform_request("POST", "/chat/completions", body.dump(), request_options);

  auto remaining = parser.finalize();
  events.insert(events.end(), remaining.begin(), remaining.end());
  return events;
}

std::vector<ServerSentEvent> ChatCompletionsResource::create_stream(const ChatCompletionRequest& request) const {
  return create_stream(request, RequestOptions{});
}

}  // namespace openai

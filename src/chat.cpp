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
  json block = json::object();
  switch (content.type) {
    case ChatMessageContent::Type::Text:
      block["type"] = "text";
      block["text"] = content.text;
      break;
    case ChatMessageContent::Type::Image:
      block["type"] = "input_image";
      if (!content.image_url.empty()) block["image_url"] = content.image_url;
      if (!content.image_detail.empty()) block["detail"] = content.image_detail;
      if (!content.file_id.empty()) block["file_id"] = content.file_id;
      break;
    case ChatMessageContent::Type::File:
      block["type"] = "input_file";
      if (!content.file_id.empty()) block["file_id"] = content.file_id;
      if (!content.file_url.empty()) block["file_url"] = content.file_url;
      if (!content.filename.empty()) block["filename"] = content.filename;
      break;
    case ChatMessageContent::Type::Audio:
      block["type"] = "input_audio";
      if (!content.audio_data.empty()) block["audio"] = { {"data", content.audio_data}, {"format", content.audio_format} };
      break;
    case ChatMessageContent::Type::Raw:
      block = content.raw.is_null() ? json::object() : content.raw;
      break;
  }
  if (!content.raw.is_null() && content.type != ChatMessageContent::Type::Raw) {
    for (auto it = content.raw.begin(); it != content.raw.end(); ++it) {
      block[it.key()] = it.value();
    }
  }
  return block;
}

json message_to_json(const ChatMessage& message) {
  json result;
  result["role"] = message.role;
  if (message.name) {
    result["name"] = *message.name;
  }
  if (!message.metadata.empty()) {
    result["metadata"] = message.metadata;
  }

  if (message.content.size() == 1 && message.content.front().type == ChatMessageContent::Type::Text &&
      message.metadata.empty() && message.tool_calls.empty()) {
    result["content"] = message.content.front().text;
  } else if (!message.content.empty()) {
    json content_array = json::array();
    for (const auto& block : message.content) {
      content_array.push_back(message_content_to_json(block));
    }
    result["content"] = std::move(content_array);
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
  content.raw = payload;
  std::string type = payload.value("type", "text");
  if (type == "text" && payload.contains("text")) {
    content.type = ChatMessageContent::Type::Text;
    if (payload["text"].is_string()) {
      content.text = payload["text"].get<std::string>();
    } else if (payload["text"].is_object()) {
      content.text = payload["text"].value("content", payload["text"].value("value", ""));
    }
  } else if (type == "input_image" || type == "image_url") {
    content.type = ChatMessageContent::Type::Image;
    content.image_url = payload.value("image_url", "");
    content.image_detail = payload.value("detail", "");
    content.file_id = payload.value("file_id", "");
  } else if (type == "input_file") {
    content.type = ChatMessageContent::Type::File;
    content.file_id = payload.value("file_id", "");
    content.file_url = payload.value("file_url", "");
    content.filename = payload.value("filename", "");
  } else if (type == "input_audio") {
    content.type = ChatMessageContent::Type::Audio;
    if (payload.contains("audio") && payload["audio"].is_object()) {
      const auto& audio = payload.at("audio");
      content.audio_data = audio.value("data", "");
      content.audio_format = audio.value("format", "");
    }
  } else {
    content.type = ChatMessageContent::Type::Raw;
  }
  return content;
}

ChatMessage parse_message(const json& payload) {
  ChatMessage message;
  message.raw = payload;

  message.role = payload.value("role", "");
  if (payload.contains("name") && payload["name"].is_string()) {
    message.name = payload["name"].get<std::string>();
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) {
        message.metadata[it.key()] = it.value().get<std::string>();
      }
    }
  }

  if (payload.contains("content")) {
    const auto& content = payload.at("content");
    if (content.is_string()) {
      ChatMessageContent item;
      item.type = ChatMessageContent::Type::Text;
      item.text = content.get<std::string>();
      message.content.push_back(std::move(item));
    } else if (content.is_array()) {
      for (const auto& block_json : content) {
        message.content.push_back(parse_message_content(block_json));
      }
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
  json body;
  body["model"] = request.model;

  json messages = json::array();
  for (const auto& message : request.messages) {
    messages.push_back(message_to_json(message));
  }
  body["messages"] = std::move(messages);

  if (!request.metadata.empty()) body["metadata"] = request.metadata;
  if (request.max_tokens) body["max_tokens"] = *request.max_tokens;
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.frequency_penalty) body["frequency_penalty"] = *request.frequency_penalty;
  if (request.presence_penalty) body["presence_penalty"] = *request.presence_penalty;
  if (!request.logit_bias.empty()) body["logit_bias"] = request.logit_bias;
  if (request.logprobs) body["logprobs"] = *request.logprobs;
  if (request.top_logprobs) body["top_logprobs"] = *request.top_logprobs;
  if (request.stop && !request.stop->empty()) body["stop"] = *request.stop;
  if (request.seed) body["seed"] = *request.seed;
  if (request.response_format) {
    json format;
    format["type"] = request.response_format->type;
    if (!request.response_format->json_schema.is_null() && !request.response_format->json_schema.empty()) {
      format["json_schema"] = request.response_format->json_schema;
    }
    body["response_format"] = std::move(format);
  }
  if (!request.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : request.tools) {
      json tool_json;
      tool_json["type"] = tool.type;
      if (tool.function) {
        json fn;
        fn["name"] = tool.function->name;
        if (tool.function->description) fn["description"] = *tool.function->description;
        if (!tool.function->parameters.is_null() && !tool.function->parameters.empty()) {
          fn["parameters"] = tool.function->parameters;
        }
        tool_json["function"] = std::move(fn);
      }
      for (auto it = tool.raw.begin(); it != tool.raw.end(); ++it) {
        tool_json[it.key()] = it.value();
      }
      tools.push_back(std::move(tool_json));
    }
    body["tools"] = std::move(tools);
  }
  if (request.tool_choice) {
    json choice;
    choice["type"] = request.tool_choice->type;
    if (request.tool_choice->function_name) {
      choice["function"] = { {"name", *request.tool_choice->function_name} };
    }
    for (auto it = request.tool_choice->raw.begin(); it != request.tool_choice->raw.end(); ++it) {
      choice[it.key()] = it.value();
    }
    body["tool_choice"] = std::move(choice);
  }
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.user) body["user"] = *request.user;
  if (request.stream) body["stream"] = *request.stream;

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
  json body;
  body["model"] = request.model;

  json messages = json::array();
  for (const auto& message : request.messages) {
    messages.push_back(message_to_json(message));
  }
  body["messages"] = std::move(messages);
  body["stream"] = true;

  if (!request.metadata.empty()) body["metadata"] = request.metadata;
  if (request.max_tokens) body["max_tokens"] = *request.max_tokens;
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.frequency_penalty) body["frequency_penalty"] = *request.frequency_penalty;
  if (request.presence_penalty) body["presence_penalty"] = *request.presence_penalty;
  if (!request.logit_bias.empty()) body["logit_bias"] = request.logit_bias;
  if (request.logprobs) body["logprobs"] = *request.logprobs;
  if (request.top_logprobs) body["top_logprobs"] = *request.top_logprobs;
  if (request.stop && !request.stop->empty()) body["stop"] = *request.stop;
  if (request.seed) body["seed"] = *request.seed;
  if (request.response_format) {
    json format;
    format["type"] = request.response_format->type;
    if (!request.response_format->json_schema.is_null() && !request.response_format->json_schema.empty()) {
      format["json_schema"] = request.response_format->json_schema;
    }
    body["response_format"] = std::move(format);
  }
  if (!request.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : request.tools) {
      json tool_json;
      tool_json["type"] = tool.type;
      if (tool.function) {
        json fn;
        fn["name"] = tool.function->name;
        if (tool.function->description) fn["description"] = *tool.function->description;
        if (!tool.function->parameters.is_null() && !tool.function->parameters.empty()) {
          fn["parameters"] = tool.function->parameters;
        }
        tool_json["function"] = std::move(fn);
      }
      for (auto it = tool.raw.begin(); it != tool.raw.end(); ++it) {
        tool_json[it.key()] = it.value();
      }
      tools.push_back(std::move(tool_json));
    }
    body["tools"] = std::move(tools);
  }
  if (request.tool_choice) {
    json choice;
    choice["type"] = request.tool_choice->type;
    if (request.tool_choice->function_name) {
      choice["function"] = { {"name", *request.tool_choice->function_name} };
    }
    for (auto it = request.tool_choice->raw.begin(); it != request.tool_choice->raw.end(); ++it) {
      choice[it.key()] = it.value();
    }
    body["tool_choice"] = std::move(choice);
  }
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.user) body["user"] = *request.user;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;

  SSEEventStream stream;
  request_options.on_chunk = [&](const char* data, std::size_t size) {
    stream.feed(data, size);
  };

  client_.perform_request("POST", "/chat/completions", body.dump(), request_options);

  stream.finalize();
  return stream.events();
}

std::vector<ServerSentEvent> ChatCompletionsResource::create_stream(const ChatCompletionRequest& request) const {
  return create_stream(request, RequestOptions{});
}

}  // namespace openai

#include "openai/chat.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"
#include "openai/streaming.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kChatCompletionsPath = "/chat/completions";

json message_content_to_json(const ChatMessageContent& content);
json message_to_json(const ChatMessage& message);

ChatCompletionTokenLogprob parse_token_logprob(const json& payload) {
  ChatCompletionTokenLogprob token;
  token.raw = payload;
  token.token = payload.value("token", "");
  if (payload.contains("bytes") && payload.at("bytes").is_array()) {
    for (const auto& value : payload.at("bytes")) {
      if (value.is_number_integer()) {
        token.bytes.push_back(value.get<int>());
      }
    }
  }
  token.logprob = payload.value("logprob", 0.0);
  return token;
}

std::vector<ChatCompletionTokenLogprob> parse_token_logprob_list(const json& payload) {
  std::vector<ChatCompletionTokenLogprob> items;
  if (payload.is_array()) {
    for (const auto& item : payload) {
      items.push_back(parse_token_logprob(item));
    }
  }
  return items;
}

std::optional<ChatCompletionLogprobs> parse_logprobs(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  ChatCompletionLogprobs logprobs;
  logprobs.raw = payload;
  if (payload.contains("content")) {
    logprobs.content = parse_token_logprob_list(payload.at("content"));
  }
  if (payload.contains("refusal")) {
    logprobs.refusal = parse_token_logprob_list(payload.at("refusal"));
  }
  return logprobs;
}

json build_chat_request_body(const ChatCompletionRequest& request, std::optional<bool> stream_override) {
  json body;
  body["model"] = request.model;

  json messages = json::array();
  for (const auto& message : request.messages) {
    messages.push_back(message_to_json(message));
  }
  body["messages"] = std::move(messages);

  if (request.audio) {
    const auto& audio = *request.audio;
    json audio_json = audio.raw.is_null() ? json::object() : audio.raw;
    if (!audio_json.is_object()) audio_json = json::object();
    if (!audio.format.empty()) audio_json["format"] = audio.format;
    if (!audio.voice.empty()) audio_json["voice"] = audio.voice;
    body["audio"] = std::move(audio_json);
  }

  if (!request.metadata.empty()) body["metadata"] = request.metadata;
  if (request.max_tokens) body["max_tokens"] = *request.max_tokens;
  if (request.max_completion_tokens) body["max_completion_tokens"] = *request.max_completion_tokens;
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.frequency_penalty) body["frequency_penalty"] = *request.frequency_penalty;
  if (request.presence_penalty) body["presence_penalty"] = *request.presence_penalty;
  if (!request.logit_bias.empty()) body["logit_bias"] = request.logit_bias;
  if (request.logprobs) body["logprobs"] = *request.logprobs;
  if (request.top_logprobs) body["top_logprobs"] = *request.top_logprobs;
  if (request.stop && !request.stop->empty()) body["stop"] = *request.stop;
  if (request.seed) body["seed"] = *request.seed;
  if (!request.functions.empty()) {
    json functions = json::array();
    for (const auto& fn : request.functions) {
      json fn_json = json::object();
      fn_json["name"] = fn.name;
      if (fn.description) fn_json["description"] = *fn.description;
      if (!fn.parameters.is_null() && !fn.parameters.empty()) fn_json["parameters"] = fn.parameters;
      functions.push_back(std::move(fn_json));
    }
    body["functions"] = std::move(functions);
  }
  if (request.function_call) {
    const auto& directive = *request.function_call;
    switch (directive.type) {
      case ChatCompletionFunctionCallDirective::Type::None:
        body["function_call"] = "none";
        break;
      case ChatCompletionFunctionCallDirective::Type::Auto:
        body["function_call"] = "auto";
        break;
      case ChatCompletionFunctionCallDirective::Type::Function: {
        if (directive.function) {
          json fn_json = directive.function->raw.is_null() ? json::object() : directive.function->raw;
          if (!fn_json.is_object()) fn_json = json::object();
          if (!directive.function->name.empty()) fn_json["name"] = directive.function->name;
          body["function_call"] = std::move(fn_json);
        }
        break;
      }
    }
  }
  if (request.response_format) {
    json format;
    format["type"] = request.response_format->type;
    if (!request.response_format->json_schema.is_null() && !request.response_format->json_schema.empty()) {
      format["json_schema"] = request.response_format->json_schema;
    }
    body["response_format"] = std::move(format);
  }
  if (request.prompt_cache_key) body["prompt_cache_key"] = *request.prompt_cache_key;
  if (request.reasoning_effort) body["reasoning_effort"] = *request.reasoning_effort;
  if (request.prediction) {
    const auto& prediction = *request.prediction;
    json prediction_json = prediction.raw.is_null() ? json::object() : prediction.raw;
    if (!prediction_json.is_object()) prediction_json = json::object();
    if (prediction.type && !prediction.type->empty()) prediction_json["type"] = *prediction.type;
    if (const auto* text = std::get_if<std::string>(&prediction.content)) {
      prediction_json["content"] = *text;
    } else if (const auto* parts = std::get_if<std::vector<ChatMessageContent>>(&prediction.content)) {
      json array = json::array();
      for (const auto& part : *parts) {
        array.push_back(message_content_to_json(part));
      }
      prediction_json["content"] = std::move(array);
    }
    body["prediction"] = std::move(prediction_json);
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
      if (tool.custom) {
        json custom_json = tool.custom->raw.is_null() ? json::object() : tool.custom->raw;
        if (!custom_json.is_object()) custom_json = json::object();
        custom_json["name"] = tool.custom->name;
        if (tool.custom->description) custom_json["description"] = *tool.custom->description;
        if (tool.custom->format) {
          const auto& format = *tool.custom->format;
          json format_json = format.raw.is_null() ? json::object() : format.raw;
          if (!format_json.is_object()) format_json = json::object();
          if (!format.type.empty()) format_json["type"] = format.type;
          if (format.grammar) {
            const auto& grammar = *format.grammar;
            json grammar_json = grammar.raw.is_null() ? json::object() : grammar.raw;
            if (!grammar_json.is_object()) grammar_json = json::object();
            if (!grammar.type.empty()) grammar_json["type"] = grammar.type;
            json grammar_body = grammar.grammar.raw.is_null() ? json::object() : grammar.grammar.raw;
            if (!grammar_body.is_object()) grammar_body = json::object();
            grammar_body["definition"] = grammar.grammar.definition;
            grammar_body["syntax"] = grammar.grammar.syntax;
            grammar_json["grammar"] = std::move(grammar_body);
            format_json = std::move(grammar_json);
          }
          custom_json["format"] = std::move(format_json);
        }
        tool_json["custom"] = std::move(custom_json);
      }
      for (auto it = tool.raw.begin(); it != tool.raw.end(); ++it) {
        tool_json[it.key()] = it.value();
      }
      tools.push_back(std::move(tool_json));
    }
    body["tools"] = std::move(tools);
  }
  if (request.tool_choice) {
    const auto& choice = *request.tool_choice;
    if (choice.literal) {
      body["tool_choice"] = *choice.literal;
    } else {
      switch (choice.type) {
        case ChatToolChoice::Type::None:
          body["tool_choice"] = "none";
          break;
        case ChatToolChoice::Type::Auto:
          body["tool_choice"] = "auto";
          break;
        case ChatToolChoice::Type::Required:
          body["tool_choice"] = "required";
          break;
        case ChatToolChoice::Type::AllowedTools: {
          json choice_json = choice.raw.is_null() ? json::object() : choice.raw;
          if (!choice_json.is_object()) choice_json = json::object();
          choice_json["type"] = "allowed_tools";
          if (choice.allowed_tools) {
            const auto& allowed_choice = *choice.allowed_tools;
            json allowed_json = allowed_choice.raw.is_null() ? json::object() : allowed_choice.raw;
            if (!allowed_json.is_object()) allowed_json = json::object();
            allowed_json["mode"] = allowed_choice.allowed_tools.mode;
            if (!allowed_choice.allowed_tools.tools.empty()) {
              allowed_json["tools"] = allowed_choice.allowed_tools.tools;
            }
            choice_json["allowed_tools"] = std::move(allowed_json);
          }
          body["tool_choice"] = std::move(choice_json);
          break;
        }
        case ChatToolChoice::Type::NamedFunction: {
          json choice_json = choice.raw.is_null() ? json::object() : choice.raw;
          if (!choice_json.is_object()) choice_json = json::object();
          choice_json["type"] = "function";
          if (choice.named_function) {
            json fn_json = choice.named_function->raw.is_null() ? json::object() : choice.named_function->raw;
            if (!fn_json.is_object()) fn_json = json::object();
            fn_json["name"] = choice.named_function->function.name;
            choice_json["function"] = std::move(fn_json);
          }
          body["tool_choice"] = std::move(choice_json);
          break;
        }
        case ChatToolChoice::Type::NamedCustom: {
          json choice_json = choice.raw.is_null() ? json::object() : choice.raw;
          if (!choice_json.is_object()) choice_json = json::object();
          choice_json["type"] = "custom";
          if (choice.named_custom) {
            json custom_json = choice.named_custom->raw.is_null() ? json::object() : choice.named_custom->raw;
            if (!custom_json.is_object()) custom_json = json::object();
            custom_json["name"] = choice.named_custom->custom.name;
            choice_json["custom"] = std::move(custom_json);
          }
          body["tool_choice"] = std::move(choice_json);
          break;
        }
      }
    }
  }
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.user) body["user"] = *request.user;
  if (request.safety_identifier) body["safety_identifier"] = *request.safety_identifier;
  if (request.n) body["n"] = *request.n;

  if (stream_override.has_value()) {
    body["stream"] = *stream_override;
  } else if (request.stream) {
    body["stream"] = *request.stream;
  }

  if (request.store) body["store"] = *request.store;
  if (!request.modalities.empty()) {
    json modalities = json::array();
    for (const auto& modality : request.modalities) {
      modalities.push_back(modality);
    }
    body["modalities"] = std::move(modalities);
  }
  if (request.verbosity) body["verbosity"] = *request.verbosity;
  if (request.web_search_options) {
    const auto& options = *request.web_search_options;
    json options_json = options.raw.is_null() ? json::object() : options.raw;
    if (!options_json.is_object()) options_json = json::object();
    if (options.search_context_size) options_json["search_context_size"] = *options.search_context_size;
    if (options.user_location) {
      const auto& location = *options.user_location;
      json location_json = location.raw.is_null() ? json::object() : location.raw;
      if (!location_json.is_object()) location_json = json::object();
      location_json["type"] = location.type;
      json approximate_json = location.approximate.raw.is_null() ? json::object() : location.approximate.raw;
      if (!approximate_json.is_object()) approximate_json = json::object();
      if (location.approximate.city) approximate_json["city"] = *location.approximate.city;
      if (location.approximate.country) approximate_json["country"] = *location.approximate.country;
      if (location.approximate.region) approximate_json["region"] = *location.approximate.region;
      if (location.approximate.timezone) approximate_json["timezone"] = *location.approximate.timezone;
      location_json["approximate"] = std::move(approximate_json);
      options_json["user_location"] = std::move(location_json);
    }
    body["web_search_options"] = std::move(options_json);
  }
  if (request.service_tier) body["service_tier"] = *request.service_tier;
  if (request.stream_options) {
    json stream_options = request.stream_options->raw.is_object() ? request.stream_options->raw : json::object();
    if (request.stream_options->include_obfuscation.has_value()) {
      stream_options["include_obfuscation"] = *request.stream_options->include_obfuscation;
    }
    if (request.stream_options->include_usage.has_value()) {
      stream_options["include_usage"] = *request.stream_options->include_usage;
    }
    body["stream_options"] = std::move(stream_options);
  }

  return body;
}

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
      if (content.detail) block["detail"] = *content.detail;
      if (!content.file_id.empty()) block["file_id"] = content.file_id;
      break;
    case ChatMessageContent::Type::File:
      block["type"] = "input_file";
      if (!content.file_id.empty()) block["file_id"] = content.file_id;
      if (!content.file_url.empty()) block["file_url"] = content.file_url;
      if (!content.filename.empty()) block["filename"] = content.filename;
      if (content.file) {
        json file_json = json::object();
        if (content.file->file_id) file_json["file_id"] = *content.file->file_id;
        if (content.file->file_data) file_json["file_data"] = *content.file->file_data;
        if (content.file->filename) file_json["filename"] = *content.file->filename;
        if (!file_json.empty()) block["file"] = std::move(file_json);
      }
      break;
    case ChatMessageContent::Type::InputAudio:
      block["type"] = "input_audio";
      if (content.input_audio) {
        block["input_audio"] = {
          {"data", content.input_audio->data},
          {"format", content.input_audio->format},
        };
      } else if (!content.audio_data.empty()) {
        block["input_audio"] = { {"data", content.audio_data}, {"format", content.audio_format} };
      }
      break;
    case ChatMessageContent::Type::Audio:
      block = content.raw.is_null() ? json::object() : content.raw;
      if (block.empty()) {
        block["type"] = "audio";
      }
      break;
    case ChatMessageContent::Type::Refusal:
      block["type"] = "refusal";
      if (!content.refusal_text.empty()) {
        block["refusal"] = content.refusal_text;
      }
      break;
    case ChatMessageContent::Type::Raw:
      block = content.raw.is_null() ? json::object() : content.raw;
      break;
  }
  if (content.format) {
    block["format"] = *content.format;
  }
  if (!content.raw.is_null() && content.type != ChatMessageContent::Type::Raw) {
    for (auto it = content.raw.begin(); it != content.raw.end(); ++it) {
      block[it.key()] = it.value();
    }
  }
  return block;
}

json metadata_to_json(const std::map<std::string, std::string>& metadata) {
  json object = json::object();
  for (const auto& item : metadata) {
    object[item.first] = item.second;
  }
  return object;
}

void apply_list_params(const ChatCompletionListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) options.query_params["order"] = *params.order;
  if (params.after) options.query_params["after"] = *params.after;
  if (params.before) options.query_params["before"] = *params.before;
  if (params.model) options.query_params["model"] = *params.model;
  if (params.metadata) {
    json query = options.query.value_or(json::object());
    if (!query.is_object()) {
      query = json::object();
    }
    query["metadata"] = metadata_to_json(*params.metadata);
    options.query = std::move(query);
  }
}

void apply_message_list_params(const ChatCompletionMessageListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) options.query_params["order"] = *params.order;
  if (params.after) options.query_params["after"] = *params.after;
  if (params.before) options.query_params["before"] = *params.before;
}

json message_to_json(const ChatMessage& message) {
  json result;
  result["role"] = message.role;
  if (message.name) {
    result["name"] = *message.name;
  }
  if (message.tool_call_id) {
    result["tool_call_id"] = *message.tool_call_id;
  }
  if (!message.metadata.empty()) {
    result["metadata"] = message.metadata;
  }
  if (message.refusal) {
    result["refusal"] = *message.refusal;
  }

  if (!message.annotations.empty()) {
    json annotations = json::array();
    for (const auto& annotation : message.annotations) {
      json annotation_json = annotation.raw.is_null() ? json::object() : annotation.raw;
      if (!annotation_json.is_object()) annotation_json = json::object();
      if (!annotation.type.empty()) annotation_json["type"] = annotation.type;
      if (annotation.url_citation) {
        const auto& citation = *annotation.url_citation;
        json citation_json = citation.title.empty() && citation.url.empty() && citation.start_index == 0 && citation.end_index == 0
                                 ? json::object()
                                 : json::object();
        citation_json["start_index"] = citation.start_index;
        citation_json["end_index"] = citation.end_index;
        citation_json["title"] = citation.title;
        citation_json["url"] = citation.url;
        annotation_json["url_citation"] = std::move(citation_json);
      }
      annotations.push_back(std::move(annotation_json));
    }
    result["annotations"] = std::move(annotations);
  }

  if (message.audio) {
    const auto& audio = *message.audio;
    json audio_json = audio.raw.is_null() ? json::object() : audio.raw;
    if (!audio_json.is_object()) audio_json = json::object();
    if (!audio.id.empty()) audio_json["id"] = audio.id;
    if (!audio.data.empty()) audio_json["data"] = audio.data;
    audio_json["expires_at"] = audio.expires_at;
    if (!audio.transcript.empty()) audio_json["transcript"] = audio.transcript;
    result["audio"] = std::move(audio_json);
  }

  if (message.function_call) {
    const auto& fn = *message.function_call;
    json fn_json = fn.raw.is_null() ? json::object() : fn.raw;
    if (!fn_json.is_object()) fn_json = json::object();
    if (!fn.name.empty()) fn_json["name"] = fn.name;
    if (!fn.arguments.empty()) fn_json["arguments"] = fn.arguments;
    result["function_call"] = std::move(fn_json);
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
      } else if (call.parsed_function) {
        const auto& fn = *call.parsed_function;
        json fn_json = fn.raw.is_null() ? json::object() : fn.raw;
        if (!fn_json.is_object()) fn_json = json::object();
        if (!fn.name.empty()) fn_json["name"] = fn.name;
        if (!fn.arguments.empty()) fn_json["arguments"] = fn.arguments;
        tool_call["function"] = std::move(fn_json);
      }
      if (call.custom) {
        const auto& custom = *call.custom;
        json custom_json = custom.raw.is_null() ? json::object() : custom.raw;
        if (!custom_json.is_object()) custom_json = json::object();
        if (!custom.name.empty()) custom_json["name"] = custom.name;
        custom_json["input"] = custom.input;
        tool_call["custom"] = std::move(custom_json);
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
    if (payload.at("function").is_object()) {
      ChatFunctionCall fn;
      fn.raw = payload.at("function");
      fn.name = payload.at("function").value("name", "");
      fn.arguments = payload.at("function").value("arguments", "");
      call.parsed_function = std::move(fn);
    }
  }
  if (payload.contains("custom") && payload.at("custom").is_object()) {
    const auto& custom = payload.at("custom");
    ChatCompletionMessageCustomToolCallPayload custom_payload;
    custom_payload.raw = custom;
    custom_payload.name = custom.value("name", "");
    custom_payload.input = custom.value("input", "");
    call.custom = std::move(custom_payload);
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
  } else if (type == "refusal") {
    content.type = ChatMessageContent::Type::Refusal;
    content.refusal_text = payload.value("refusal", "");
  } else if (type == "input_image" || type == "image_url") {
    content.type = ChatMessageContent::Type::Image;
    content.image_url = payload.value("image_url", "");
    content.image_detail = payload.value("detail", "");
    if (payload.contains("detail") && payload["detail"].is_string()) {
      content.detail = payload["detail"].get<std::string>();
    }
    content.file_id = payload.value("file_id", "");
  } else if (type == "input_file") {
    content.type = ChatMessageContent::Type::File;
    content.file_id = payload.value("file_id", "");
    content.file_url = payload.value("file_url", "");
    content.filename = payload.value("filename", "");
    if (payload.contains("file") && payload["file"].is_object()) {
      const auto& file = payload.at("file");
      ChatMessageContent::FilePayload payload_file;
      if (file.contains("file_id") && file["file_id"].is_string()) payload_file.file_id = file["file_id"].get<std::string>();
      if (file.contains("file_data") && file["file_data"].is_string()) payload_file.file_data = file["file_data"].get<std::string>();
      if (file.contains("filename") && file["filename"].is_string()) payload_file.filename = file["filename"].get<std::string>();
      content.file = std::move(payload_file);
      if (content.file->file_id && content.file_id.empty()) content.file_id = *content.file->file_id;
      if (content.file->filename && content.filename.empty()) content.filename = *content.file->filename;
    }
  } else if (type == "file" && payload.contains("file") && payload["file"].is_object()) {
    content.type = ChatMessageContent::Type::File;
    const auto& file = payload.at("file");
    ChatMessageContent::FilePayload payload_file;
    if (file.contains("file_id") && file["file_id"].is_string()) payload_file.file_id = file["file_id"].get<std::string>();
    if (file.contains("file_data") && file["file_data"].is_string()) payload_file.file_data = file["file_data"].get<std::string>();
    if (file.contains("filename") && file["filename"].is_string()) payload_file.filename = file["filename"].get<std::string>();
    content.file = std::move(payload_file);
    if (content.file->file_id) content.file_id = *content.file->file_id;
    if (content.file->filename) content.filename = *content.file->filename;
  } else if (type == "input_audio") {
    content.type = ChatMessageContent::Type::InputAudio;
    if (payload.contains("input_audio") && payload["input_audio"].is_object()) {
      const auto& audio = payload.at("input_audio");
      ChatMessageContent::InputAudioPayload audio_payload;
      audio_payload.data = audio.value("data", "");
      audio_payload.format = audio.value("format", "");
      content.input_audio = std::move(audio_payload);
    } else if (payload.contains("audio") && payload["audio"].is_object()) {
      const auto& audio = payload.at("audio");
      ChatMessageContent::InputAudioPayload audio_payload;
      audio_payload.data = audio.value("data", "");
      audio_payload.format = audio.value("format", "");
      content.input_audio = std::move(audio_payload);
    }
    if (content.input_audio) {
      content.audio_data = content.input_audio->data;
      content.audio_format = content.input_audio->format;
      content.format = content.input_audio->format;
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
  if (payload.contains("id") && payload["id"].is_string()) {
    message.id = payload["id"].get<std::string>();
  }
  if (payload.contains("tool_call_id") && payload["tool_call_id"].is_string()) {
    message.tool_call_id = payload["tool_call_id"].get<std::string>();
  }
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

  if (payload.contains("annotations") && payload["annotations"].is_array()) {
    for (const auto& annotation_json : payload.at("annotations")) {
      ChatMessageAnnotation annotation;
      annotation.raw = annotation_json;
      annotation.type = annotation_json.value("type", "");
      if (annotation_json.contains("url_citation") && annotation_json.at("url_citation").is_object()) {
        const auto& citation = annotation_json.at("url_citation");
        ChatMessageAnnotationURLCitation url_citation;
        url_citation.start_index = citation.value("start_index", 0);
        url_citation.end_index = citation.value("end_index", 0);
        url_citation.title = citation.value("title", "");
        url_citation.url = citation.value("url", "");
        annotation.url_citation = std::move(url_citation);
      }
      message.annotations.push_back(std::move(annotation));
    }
  }

  if (payload.contains("audio") && payload["audio"].is_object()) {
    const auto& audio = payload.at("audio");
    ChatCompletionAudio message_audio;
    message_audio.raw = audio;
    message_audio.id = audio.value("id", "");
    message_audio.data = audio.value("data", "");
    if (audio.contains("expires_at") && audio["expires_at"].is_number_integer()) {
      message_audio.expires_at = audio["expires_at"].get<std::int64_t>();
    }
    message_audio.transcript = audio.value("transcript", "");
    message.audio = std::move(message_audio);
  }

  if (payload.contains("function_call") && payload["function_call"].is_object()) {
    const auto& function_call = payload.at("function_call");
    ChatFunctionCall fn;
    fn.raw = function_call;
    fn.name = function_call.value("name", "");
    fn.arguments = function_call.value("arguments", "");
    message.function_call = std::move(fn);
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
      ChatCompletionMessageToolCall structured;
      structured.raw = call_json;
      std::string call_type = call_json.value("type", "");
      if (call_type == "function" && call_json.contains("function") && call_json.at("function").is_object()) {
        structured.type = ChatCompletionMessageToolCall::Type::Function;
        ChatCompletionMessageFunctionToolCall function_call;
        function_call.raw = call_json;
        function_call.id = call_json.value("id", "");
        function_call.type = call_type;
        ChatFunctionCall fn;
        fn.raw = call_json.at("function");
        fn.name = call_json.at("function").value("name", "");
        fn.arguments = call_json.at("function").value("arguments", "");
        function_call.function = std::move(fn);
        structured.function_call = std::move(function_call);
      } else if (call_type == "custom" && call_json.contains("custom") && call_json.at("custom").is_object()) {
        structured.type = ChatCompletionMessageToolCall::Type::Custom;
        ChatCompletionMessageCustomToolCall custom_call;
        custom_call.raw = call_json;
        custom_call.id = call_json.value("id", "");
        custom_call.type = call_type;
        ChatCompletionMessageCustomToolCallPayload payload_custom;
        payload_custom.raw = call_json.at("custom");
        payload_custom.name = call_json.at("custom").value("name", "");
        payload_custom.input = call_json.at("custom").value("input", "");
        custom_call.custom = std::move(payload_custom);
        structured.custom_call = std::move(custom_call);
      }
      message.structured_tool_calls.push_back(std::move(structured));
    }
  }
  if (payload.contains("refusal") && payload["refusal"].is_string()) {
    message.refusal = payload["refusal"].get<std::string>();
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
  if (payload.contains("logprobs") && !payload["logprobs"].is_null()) {
    choice.raw_logprobs = payload.at("logprobs");
    choice.logprobs = parse_logprobs(payload.at("logprobs"));
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
  if (payload.contains("service_tier") && payload["service_tier"].is_string()) {
    completion.service_tier = payload["service_tier"].get<std::string>();
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) {
        completion.metadata[it.key()] = it.value().get<std::string>();
      }
    }
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

ChatCompletionList parse_chat_completion_list(const json& payload) {
  ChatCompletionList list;
  list.raw = payload;
  if (payload.contains("data") && payload["data"].is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_chat_completion(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload["next_cursor"].is_string()) {
    list.next_cursor = payload["next_cursor"].get<std::string>();
  }
  if (!list.next_cursor && payload.contains("last_id") && payload["last_id"].is_string()) {
    list.next_cursor = payload["last_id"].get<std::string>();
  }
  if (!list.next_cursor && !list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

ChatCompletionDeleted parse_chat_completion_deleted(const json& payload) {
  ChatCompletionDeleted deleted;
  deleted.raw = payload;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
}

ChatCompletionStoreMessage parse_chat_completion_store_message(const json& payload) {
  ChatCompletionStoreMessage store_message;
  store_message.raw = payload;
  store_message.message = parse_message(payload);
  store_message.id = payload.value("id", store_message.message.id.value_or(""));
  store_message.content_parts = store_message.message.content;
  return store_message;
}

ChatCompletionStoreMessageList parse_chat_completion_store_message_list(const json& payload) {
  ChatCompletionStoreMessageList list;
  list.raw = payload;
  if (payload.contains("data") && payload["data"].is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_chat_completion_store_message(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload["next_cursor"].is_string()) {
    list.next_cursor = payload["next_cursor"].get<std::string>();
  }
  if (!list.next_cursor && payload.contains("last_id") && payload["last_id"].is_string()) {
    list.next_cursor = payload["last_id"].get<std::string>();
  }
  if (!list.next_cursor && !list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

}  // namespace

ChatCompletion ChatCompletionsResource::create(const ChatCompletionRequest& request) const {
  return create(request, RequestOptions{});
}

ChatCompletion ChatCompletionsResource::create(const ChatCompletionRequest& request,
                                               const RequestOptions& options) const {
  json body = build_chat_request_body(request, std::nullopt);
  auto response = client_.perform_request("POST", kChatCompletionsPath, body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_chat_completion(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion response: ") + ex.what());
  }
}

void ChatCompletionsResource::create_stream(const ChatCompletionRequest& request,
                                            const std::function<bool(const ServerSentEvent&)>& on_event,
                                            const RequestOptions& options) const {
  json body = build_chat_request_body(request, true);

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;

  SSEEventStream stream([&](const ServerSentEvent& event) {
    if (on_event) {
      return on_event(event);
    }
    return true;
  });
  request_options.on_chunk = [&](const char* data, std::size_t size) {
    stream.feed(data, size);
  };

  client_.perform_request("POST", kChatCompletionsPath, body.dump(), request_options);

  stream.finalize();
}

std::vector<ServerSentEvent> ChatCompletionsResource::create_stream(const ChatCompletionRequest& request) const {
  return create_stream(request, RequestOptions{});
}

void ChatCompletionsResource::create_stream(const ChatCompletionRequest& request,
                                            const std::function<bool(const ServerSentEvent&)>& on_event) const {
  create_stream(request, on_event, RequestOptions{});
}

std::vector<ServerSentEvent> ChatCompletionsResource::create_stream(const ChatCompletionRequest& request,
                                                                    const RequestOptions& options) const {
  std::vector<ServerSentEvent> events;
  create_stream(
      request,
      [&](const ServerSentEvent& event) {
        events.push_back(event);
        return true;
      },
      options);
  return events;
}

ChatCompletion ChatCompletionsResource::retrieve(const std::string& completion_id) const {
  return retrieve(completion_id, RequestOptions{});
}

ChatCompletion ChatCompletionsResource::retrieve(const std::string& completion_id,
                                                 const RequestOptions& options) const {
  const std::string path = std::string(kChatCompletionsPath) + "/" + completion_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_chat_completion(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion retrieve response: ") + ex.what());
  }
}

ChatCompletion ChatCompletionsResource::update(const std::string& completion_id,
                                               const ChatCompletionUpdateRequest& request) const {
  return update(completion_id, request, RequestOptions{});
}

ChatCompletion ChatCompletionsResource::update(const std::string& completion_id,
                                               const ChatCompletionUpdateRequest& request,
                                               const RequestOptions& options) const {
  const std::string path = std::string(kChatCompletionsPath) + "/" + completion_id;
  json body = json::object();
  if (request.clear_metadata) {
    body["metadata"] = nullptr;
  } else if (request.metadata) {
    body["metadata"] = metadata_to_json(*request.metadata);
  }

  auto response = client_.perform_request("POST", path, body.dump(), options);
  try {
    return parse_chat_completion(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion update response: ") + ex.what());
  }
}

ChatCompletionList ChatCompletionsResource::list() const {
  return list(ChatCompletionListParams{}, RequestOptions{});
}

ChatCompletionList ChatCompletionsResource::list(const ChatCompletionListParams& params) const {
  return list(params, RequestOptions{});
}

ChatCompletionList ChatCompletionsResource::list(const ChatCompletionListParams& params,
                                                 const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_list_params(params, request_options);

  auto response = client_.perform_request("GET", kChatCompletionsPath, "", request_options);
  try {
    return parse_chat_completion_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion list response: ") + ex.what());
  }
}

CursorPage<ChatCompletion> ChatCompletionsResource::list_page() const {
  return list_page(ChatCompletionListParams{}, RequestOptions{});
}

CursorPage<ChatCompletion> ChatCompletionsResource::list_page(const ChatCompletionListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<ChatCompletion> ChatCompletionsResource::list_page(const ChatCompletionListParams& params,
                                                              const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_list_params(params, request_options);

  auto fetch_impl = std::make_shared<std::function<CursorPage<ChatCompletion>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<ChatCompletion> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);

    ChatCompletionList list;
    try {
      list = parse_chat_completion_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse chat completion list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<ChatCompletion>(std::move(list.data),
                                      list.has_more,
                                      std::move(cursor),
                                      request_options,
                                      *fetch_impl,
                                      "after",
                                      std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kChatCompletionsPath;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

ChatCompletionDeleted ChatCompletionsResource::remove(const std::string& completion_id) const {
  return remove(completion_id, RequestOptions{});
}

ChatCompletionDeleted ChatCompletionsResource::remove(const std::string& completion_id,
                                                      const RequestOptions& options) const {
  const std::string path = std::string(kChatCompletionsPath) + "/" + completion_id;
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "*/*";
  auto response = client_.perform_request("DELETE", path, "", request_options);
  try {
    return parse_chat_completion_deleted(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion delete response: ") + ex.what());
  }
}

ChatCompletionToolRunResult ChatCompletionsResource::run_tools(const ChatCompletionToolRunParams& params) const {
  return run_tools(params, RequestOptions{});
}

ChatCompletionToolRunResult ChatCompletionsResource::run_tools(const ChatCompletionToolRunParams& params,
                                                               const RequestOptions& options) const {
  if (params.max_iterations == 0) {
    throw OpenAIError("ChatCompletionToolRunParams.max_iterations must be greater than 0");
  }

  ChatCompletionRequest base_request = params.request;
  base_request.stream = false;

  std::map<std::string, ChatToolFunctionHandler> handlers;
  std::vector<ChatCompletionToolDefinition> tool_definitions = base_request.tools;
  for (const auto& handler : params.functions) {
    if (!handler.definition.function || handler.definition.type != "function") {
      throw OpenAIError("ChatCompletion tool handlers must define a function tool");
    }
    const auto& name = handler.definition.function->name;
    handlers[name] = handler;

    const bool already_defined = std::any_of(
        tool_definitions.begin(), tool_definitions.end(), [&](const ChatCompletionToolDefinition& existing) {
          if (existing.type != handler.definition.type) return false;
          if (!existing.function || !handler.definition.function) return false;
          return existing.function->name == name;
        });
    if (!already_defined) {
      tool_definitions.push_back(handler.definition);
    }
  }

  base_request.tools = std::move(tool_definitions);

  std::vector<ChatMessage> transcript = base_request.messages;
  std::vector<ChatCompletion> completions;

  for (std::size_t iteration = 0; iteration < params.max_iterations; ++iteration) {
    ChatCompletionRequest iteration_request = base_request;
    iteration_request.messages = transcript;
    iteration_request.stream = false;

    ChatCompletion completion = create(iteration_request, options);
    completions.push_back(completion);

    if (completion.choices.empty() || !completion.choices[0].message.has_value()) {
      throw OpenAIError("ChatCompletion response missing assistant message");
    }

    transcript.push_back(*completion.choices[0].message);
    ChatMessage& assistant_message = transcript.back();

    if (assistant_message.tool_calls.empty()) {
      ChatCompletionToolRunResult result;
      result.final_completion = completion;
      result.completions = std::move(completions);
      result.transcript = std::move(transcript);
      return result;
    }

    for (const auto& tool_call : assistant_message.tool_calls) {
      if (tool_call.type != "function") {
        continue;
      }

      const std::string tool_name = tool_call.function.value("name", "");
      ChatMessage tool_message;
      tool_message.role = "tool";
      tool_message.tool_call_id = tool_call.id;

      ChatMessageContent content_block;
      content_block.type = ChatMessageContent::Type::Text;

      auto handler_it = handlers.find(tool_name);
      if (handler_it == handlers.end()) {
        content_block.text = "Tool '" + tool_name + "' is not registered.";
        tool_message.content.push_back(content_block);
        transcript.push_back(std::move(tool_message));
        continue;
      }

      json args_json = json::object();
      if (tool_call.function.contains("arguments")) {
        const auto& arguments = tool_call.function.at("arguments");
        if (arguments.is_string()) {
          const auto args_text = arguments.get<std::string>();
          if (!args_text.empty()) {
            try {
              args_json = json::parse(args_text);
            } catch (const json::exception&) {
              args_json = args_text;
            }
          }
        } else {
          args_json = arguments;
        }
      }

      try {
        json callback_result = handler_it->second.callback(args_json);
        if (callback_result.is_string()) {
          content_block.text = callback_result.get<std::string>();
        } else if (callback_result.is_null()) {
          content_block.text = "null";
        } else {
          content_block.text = callback_result.dump();
        }
      } catch (const std::exception& ex) {
        content_block.text = ex.what();
      } catch (...) {
        content_block.text = "Tool execution failed.";
      }

      tool_message.content.push_back(content_block);
      transcript.push_back(std::move(tool_message));
    }
  }

  throw OpenAIError("Exceeded maximum chat completion iterations while running tools");
}

ChatCompletionStoreMessageList ChatCompletionsResource::MessagesResource::list(
    const std::string& completion_id) const {
  return list(completion_id, ChatCompletionMessageListParams{}, RequestOptions{});
}

ChatCompletionStoreMessageList ChatCompletionsResource::MessagesResource::list(
    const std::string& completion_id,
    const ChatCompletionMessageListParams& params) const {
  return list(completion_id, params, RequestOptions{});
}

ChatCompletionStoreMessageList ChatCompletionsResource::MessagesResource::list(
    const std::string& completion_id,
    const ChatCompletionMessageListParams& params,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_message_list_params(params, request_options);

  const std::string path = std::string(kChatCompletionsPath) + "/" + completion_id + "/messages";
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_chat_completion_store_message_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse chat completion messages list response: ") + ex.what());
  }
}

CursorPage<ChatCompletionStoreMessage> ChatCompletionsResource::MessagesResource::list_page(
    const std::string& completion_id) const {
  return list_page(completion_id, ChatCompletionMessageListParams{}, RequestOptions{});
}

CursorPage<ChatCompletionStoreMessage> ChatCompletionsResource::MessagesResource::list_page(
    const std::string& completion_id,
    const ChatCompletionMessageListParams& params) const {
  return list_page(completion_id, params, RequestOptions{});
}

CursorPage<ChatCompletionStoreMessage> ChatCompletionsResource::MessagesResource::list_page(
    const std::string& completion_id,
    const ChatCompletionMessageListParams& params,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_message_list_params(params, request_options);

  auto fetch_impl =
      std::make_shared<std::function<CursorPage<ChatCompletionStoreMessage>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options)
      -> CursorPage<ChatCompletionStoreMessage> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);

    ChatCompletionStoreMessageList list;
    try {
      list = parse_chat_completion_store_message_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse chat completion messages list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<ChatCompletionStoreMessage>(std::move(list.data),
                                                  list.has_more,
                                                  std::move(cursor),
                                                  request_options,
                                                  *fetch_impl,
                                                  "after",
                                                  std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = std::string(kChatCompletionsPath) + "/" + completion_id + "/messages";
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

}  // namespace openai

#include "openai/responses.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/streaming.hpp"

#include <nlohmann/json.hpp>

#include <sstream>
#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

const char* kResponseEndpoint = "/responses";

void ensure_model_present(const json& body) {
  if (!body.contains("model") || body.at("model").is_null()) {
    throw OpenAIError("ResponsesRequest body must include a model");
  }
}

json build_request_body(const ResponseRequest& request) {
  json body = json::object();
  body["model"] = request.model;

  json input = json::array();
  for (const auto& item : request.input) {
    json message;
    message["role"] = item.role;
    if (!item.metadata.empty()) {
      message["metadata"] = item.metadata;
    }

    json content = json::array();
    for (const auto& piece : item.content) {
      json content_item;
      switch (piece.type) {
        case ResponseInputContent::Type::Text:
          content_item["type"] = "input_text";
          content_item["text"] = piece.text;
          break;
        case ResponseInputContent::Type::Image:
          content_item["type"] = "input_image";
          if (!piece.image_url.empty()) content_item["image_url"] = piece.image_url;
          if (!piece.image_detail.empty()) content_item["detail"] = piece.image_detail;
          if (!piece.file_id.empty()) content_item["file_id"] = piece.file_id;
          break;
        case ResponseInputContent::Type::File:
          content_item["type"] = "input_file";
          if (!piece.file_id.empty()) content_item["file_id"] = piece.file_id;
          if (!piece.file_url.empty()) content_item["file_url"] = piece.file_url;
          if (!piece.filename.empty()) content_item["filename"] = piece.filename;
          break;
        case ResponseInputContent::Type::Audio:
          content_item["type"] = "input_audio";
          if (!piece.audio_data.empty()) content_item["audio"] = { {"data", piece.audio_data}, {"format", piece.audio_format} };
          break;
        case ResponseInputContent::Type::Raw:
          content_item = piece.raw;
          break;
      }
      if (piece.type != ResponseInputContent::Type::Raw) {
        for (auto it = piece.raw.begin(); it != piece.raw.end(); ++it) {
          content_item[it.key()] = it.value();
        }
      }
      content.push_back(std::move(content_item));
    }
    message["content"] = std::move(content);
    input.push_back(std::move(message));
  }
  body["input"] = std::move(input);

  if (request.background) body["background"] = *request.background;
  if (request.conversation_id) body["conversation"] = *request.conversation_id;
  if (!request.include.empty()) body["include"] = request.include;
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.max_output_tokens) body["max_output_tokens"] = *request.max_output_tokens;
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.previous_response_id) body["previous_response_id"] = *request.previous_response_id;
  if (request.prompt) {
    json prompt;
    prompt["id"] = request.prompt->id;
    if (!request.prompt->variables.empty()) prompt["variables"] = request.prompt->variables;
    for (auto it = request.prompt->extra.begin(); it != request.prompt->extra.end(); ++it) {
      prompt[it.key()] = it.value();
    }
    body["prompt"] = std::move(prompt);
  }
  if (request.prompt_cache_key) body["prompt_cache_key"] = *request.prompt_cache_key;
  if (request.reasoning) {
    json reasoning;
    if (request.reasoning->effort) reasoning["effort"] = *request.reasoning->effort;
    for (auto it = request.reasoning->extra.begin(); it != request.reasoning->extra.end(); ++it) {
      reasoning[it.key()] = it.value();
    }
    body["reasoning"] = std::move(reasoning);
  }
  if (request.safety_identifier) body["safety_identifier"] = *request.safety_identifier;
  if (request.service_tier) body["service_tier"] = *request.service_tier;
  if (request.store) body["store"] = *request.store;
  if (request.stream) body["stream"] = *request.stream;
  if (request.stream_options) {
    json stream_options;
    if (request.stream_options->include_usage) stream_options["include_usage"] = *request.stream_options->include_usage;
    for (auto it = request.stream_options->extra.begin(); it != request.stream_options->extra.end(); ++it) {
      stream_options[it.key()] = it.value();
    }
    body["stream_options"] = std::move(stream_options);
  }
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (!request.tools.empty()) body["tools"] = request.tools;
  if (request.tool_choice) body["tool_choice"] = *request.tool_choice;

  ensure_model_present(body);
  return body;
}

std::string join_messages(const std::vector<ResponseOutputMessage>& messages) {
  std::ostringstream stream;
  for (const auto& message : messages) {
    for (const auto& segment : message.text_segments) {
      stream << segment.text;
    }
  }
  return stream.str();
}

ResponseUsage parse_usage(const json& payload) {
  ResponseUsage usage;
  usage.input_tokens = payload.value("input_tokens", 0);
  usage.output_tokens = payload.value("output_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  usage.extra = payload;
  return usage;
}

ResponseOutputMessage parse_output_message(const json& payload) {
  ResponseOutputMessage message;
  message.role = payload.value("role", "assistant");
  if (payload.contains("content")) {
    for (const auto& block : payload.at("content")) {
      if (block.value("type", std::string{}) == "output_text") {
        ResponseOutputTextSegment segment;
        segment.text = block.value("text", std::string{});
        message.text_segments.push_back(std::move(segment));
      }
    }
  }
  return message;
}

std::vector<ResponseOutputMessage> parse_output_messages(const json& payload) {
  std::vector<ResponseOutputMessage> messages;
  if (!payload.contains("output")) {
    return messages;
  }
  for (const auto& item : payload.at("output")) {
    if (item.value("type", std::string{}) != "message") {
      continue;
    }
    if (!item.contains("content")) {
      continue;
    }
    messages.push_back(parse_output_message(item));
  }
  return messages;
}

Response parse_response(const json& payload) {
  Response response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.object = payload.value("object", "");
  response.created = payload.value("created", 0);
  response.model = payload.value("model", "");

  response.messages = parse_output_messages(payload);
  response.output_text = join_messages(response.messages);

  if (payload.contains("usage")) {
    response.usage = parse_usage(payload.at("usage"));
  }

  return response;
}

ResponseList parse_response_list(const json& payload) {
  ResponseList list;
  list.raw = payload;
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_response(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  return list;
}

json build_retrieve_query(const ResponseRetrieveOptions& options) {
  json query = json::object();
  query["stream"] = options.stream;
  return query;
}

std::string build_response_path(const std::string& response_id) {
  return std::string(kResponseEndpoint) + "/" + response_id;
}

}  // namespace

Response ResponsesResource::create(const ResponseRequest& request, const RequestOptions& options) const {
  auto body = build_request_body(request);
  auto response = client_.perform_request("POST", kResponseEndpoint, body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse response body: ") + ex.what());
  }
}

Response ResponsesResource::create(const ResponseRequest& request) const {
  return create(request, RequestOptions{});
}

Response ResponsesResource::retrieve(const std::string& response_id,
                                     const ResponseRetrieveOptions& retrieve_options,
                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.query_params["stream"] = retrieve_options.stream ? "true" : "false";

  auto response = client_.perform_request("GET", build_response_path(response_id), "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse response retrieval: ") + ex.what());
  }
}

Response ResponsesResource::retrieve(const std::string& response_id) const {
  return retrieve(response_id, ResponseRetrieveOptions{}, RequestOptions{});
}

void ResponsesResource::remove(const std::string& response_id, const RequestOptions& options) const {
  auto response = client_.perform_request("DELETE", build_response_path(response_id), "", options);
  (void)response;
}

void ResponsesResource::remove(const std::string& response_id) const {
  remove(response_id, RequestOptions{});
}

Response ResponsesResource::cancel(const std::string& response_id, const RequestOptions& options) const {
  auto path = build_response_path(response_id) + "/cancel";
  auto response = client_.perform_request("POST", path, json::object().dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse cancel response: ") + ex.what());
  }
}

Response ResponsesResource::cancel(const std::string& response_id) const {
  return cancel(response_id, RequestOptions{});
}

ResponseList ResponsesResource::list(const RequestOptions& options) const {
  auto response = client_.perform_request("GET", kResponseEndpoint, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_response_list(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse responses list: ") + ex.what());
  }
}

ResponseList ResponsesResource::list() const {
  return list(RequestOptions{});
}

std::vector<ServerSentEvent> ResponsesResource::create_stream(const ResponseRequest& request,
                                                              const RequestOptions& options) const {
  SSEParser parser;
  std::vector<ServerSentEvent> events;

  auto body = build_request_body(request);
  body["stream"] = true;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) {
    auto chunk_events = parser.feed(data, size);
    events.insert(events.end(), chunk_events.begin(), chunk_events.end());
  };

  client_.perform_request("POST", kResponseEndpoint, body.dump(), request_options);

  auto remaining = parser.finalize();
  events.insert(events.end(), remaining.begin(), remaining.end());
  return events;
}

std::vector<ServerSentEvent> ResponsesResource::create_stream(const ResponseRequest& request) const {
  return create_stream(request, RequestOptions{});
}

std::vector<ServerSentEvent> ResponsesResource::retrieve_stream(const std::string& response_id,
                                                                const ResponseRetrieveOptions& retrieve_options,
                                                                const RequestOptions& options) const {
  SSEParser parser;
  std::vector<ServerSentEvent> events;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.query_params["stream"] = retrieve_options.stream ? "true" : "false";
  request_options.on_chunk = [&](const char* data, std::size_t size) {
    auto chunk_events = parser.feed(data, size);
    events.insert(events.end(), chunk_events.begin(), chunk_events.end());
  };

  client_.perform_request("GET", build_response_path(response_id), "", request_options);

  auto remaining = parser.finalize();
  events.insert(events.end(), remaining.begin(), remaining.end());
  return events;
}

std::vector<ServerSentEvent> ResponsesResource::retrieve_stream(const std::string& response_id) const {
  return retrieve_stream(response_id, ResponseRetrieveOptions{.stream = true}, RequestOptions{});
}

}  // namespace openai

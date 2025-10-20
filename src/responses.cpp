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
  if (!request.body.is_object()) {
    throw OpenAIError("ResponseRequest.body must be a JSON object");
  }
  json body = request.body;
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

#include "openai/runs.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/assistant_stream.hpp"
#include "openai/streaming.hpp"

#include <nlohmann/json.hpp>

#include <thread>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

std::string runs_path(const std::string& thread_id) {
  return "/threads/" + thread_id + "/runs";
}

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

json metadata_to_json(const std::map<std::string, std::string>& metadata) {
  json value = json::object();
  for (const auto& entry : metadata) value[entry.first] = entry.second;
  return value;
}

json tool_choice_to_json(const std::optional<AssistantToolChoice>& choice) {
  if (!choice) return json();
  json value;
  value["type"] = choice->type;
  if (choice->function_name) {
    value["function"] = json::object({{"name", *choice->function_name}});
  }
  for (auto it = choice->raw.begin(); it != choice->raw.end(); ++it) {
    value[it.key()] = it.value();
  }
  return value;
}

json tool_to_json(const AssistantTool& tool) {
  json value;
  switch (tool.type) {
    case AssistantTool::Type::CodeInterpreter:
      value["type"] = "code_interpreter";
      break;
    case AssistantTool::Type::FileSearch:
      value["type"] = "file_search";
      if (tool.file_search) {
        json overrides;
        if (tool.file_search->max_num_results) overrides["max_num_results"] = *tool.file_search->max_num_results;
        if (tool.file_search->ranker || tool.file_search->score_threshold) {
          json ranking;
          if (tool.file_search->ranker) ranking["ranker"] = *tool.file_search->ranker;
          if (tool.file_search->score_threshold) ranking["score_threshold"] = *tool.file_search->score_threshold;
          overrides["ranking_options"] = std::move(ranking);
        }
        value["file_search"] = std::move(overrides);
      }
      break;
    case AssistantTool::Type::Function:
      value["type"] = "function";
      if (tool.function) {
        json fn;
        fn["name"] = tool.function->name;
        if (tool.function->description) fn["description"] = *tool.function->description;
        if (!tool.function->parameters.is_null() && !tool.function->parameters.empty()) fn["parameters"] = tool.function->parameters;
        value["function"] = std::move(fn);
      }
      break;
  }
  for (auto it = tool.raw.begin(); it != tool.raw.end(); ++it) value[it.key()] = it.value();
  return value;
}

json truncation_to_json(const std::optional<RunTruncationStrategy>& strategy) {
  if (!strategy) return json();
  json value;
  value["type"] = strategy->type == RunTruncationStrategy::Type::Auto ? "auto" : "last_messages";
  if (strategy->last_messages) value["last_messages"] = *strategy->last_messages;
  return value;
}

json response_format_to_json(const std::optional<AssistantResponseFormat>& format) {
  if (!format) return json();
  json value;
  value["type"] = format->type;
  if (!format->json_schema.is_null() && !format->json_schema.empty()) value["json_schema"] = format->json_schema;
  return value;
}

json message_content_to_json(const std::variant<std::string, std::vector<ThreadMessageContentPart>>& content) {
  if (std::holds_alternative<std::string>(content)) return json(std::get<std::string>(content));
  json array = json::array();
  for (const auto& part : std::get<std::vector<ThreadMessageContentPart>>(content)) {
    json item;
    switch (part.type) {
      case ThreadMessageContentPart::Type::Text:
        item["type"] = "text";
        item["text"] = part.text;
        break;
      case ThreadMessageContentPart::Type::ImageFile:
        item["type"] = "image_file";
        if (part.image_file) {
          json image;
          image["file_id"] = part.image_file->file_id;
          if (part.image_file->detail) image["detail"] = *part.image_file->detail;
          item["image_file"] = std::move(image);
        }
        break;
      case ThreadMessageContentPart::Type::ImageURL:
        item["type"] = "image_url";
        if (part.image_url) {
          json image;
          image["url"] = part.image_url->url;
          if (part.image_url->detail) image["detail"] = *part.image_url->detail;
          item["image_url"] = std::move(image);
        }
        break;
      case ThreadMessageContentPart::Type::Raw:
        item = part.raw;
        break;
    }
    if (part.type != ThreadMessageContentPart::Type::Raw) {
      for (auto it = part.raw.begin(); it != part.raw.end(); ++it) item[it.key()] = it.value();
    }
    array.push_back(std::move(item));
  }
  return array;
}

json attachments_to_json(const std::vector<MessageAttachment>& attachments) {
  json array = json::array();
  for (const auto& attachment : attachments) {
    json obj;
    obj["file_id"] = attachment.file_id;
    if (!attachment.tools.empty()) {
      json tools = json::array();
      for (const auto& tool : attachment.tools) {
        if (tool.type == ThreadMessageAttachmentTool::Type::FileSearch) {
          tools.push_back(json::object({{"type", "file_search"}}));
        } else {
          tools.push_back(json::object({{"type", "code_interpreter"}}));
        }
      }
      obj["tools"] = std::move(tools);
    }
    array.push_back(std::move(obj));
  }
  return array;
}

json additional_messages_to_json(const std::vector<RunAdditionalMessage>& messages) {
  if (messages.empty()) return json();
  json array = json::array();
  for (const auto& message : messages) {
    json obj;
    obj["role"] = message.role;
    obj["content"] = message_content_to_json(message.content);
    if (!message.attachments.empty()) obj["attachments"] = attachments_to_json(message.attachments);
    if (!message.metadata.empty()) obj["metadata"] = metadata_to_json(message.metadata);
    array.push_back(std::move(obj));
  }
  return array;
}

json build_run_create_body_impl(const RunCreateRequest& request) {
  json body;
  body["assistant_id"] = request.assistant_id;
  if (request.additional_instructions) body["additional_instructions"] = *request.additional_instructions;
  if (!request.additional_messages.empty()) body["additional_messages"] = additional_messages_to_json(request.additional_messages);
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.max_completion_tokens) body["max_completion_tokens"] = *request.max_completion_tokens;
  if (request.max_prompt_tokens) body["max_prompt_tokens"] = *request.max_prompt_tokens;
  if (!request.metadata.empty()) body["metadata"] = metadata_to_json(request.metadata);
  if (request.model) body["model"] = *request.model;
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.reasoning_effort) body["reasoning_effort"] = *request.reasoning_effort;
  if (request.response_format) body["response_format"] = response_format_to_json(request.response_format);
  if (request.stream) body["stream"] = *request.stream;
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.tool_choice) body["tool_choice"] = tool_choice_to_json(request.tool_choice);
  if (!request.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : request.tools) tools.push_back(tool_to_json(tool));
    body["tools"] = std::move(tools);
  }
  if (request.truncation_strategy) body["truncation_strategy"] = truncation_to_json(request.truncation_strategy);
  return body;
}

json update_request_to_json(const RunUpdateRequest& request) {
  json body;
  if (request.metadata) body["metadata"] = metadata_to_json(*request.metadata);
  return body;
}

json submit_tool_outputs_to_json(const RunSubmitToolOutputsRequest& request) {
  json body;
  json outputs = json::array();
  for (const auto& output : request.outputs) {
    outputs.push_back(json::object({{"tool_call_id", output.tool_call_id}, {"output", output.output}}));
  }
  body["tool_outputs"] = std::move(outputs);
  if (request.stream) body["stream"] = *request.stream;
  return body;
}

AssistantResponseFormat parse_response_format(const json& payload) {
  AssistantResponseFormat format;
  format.type = payload.value("type", "");
  if (payload.contains("json_schema")) {
    format.json_schema = payload.at("json_schema");
  }
  return format;
}

AssistantTool parse_assistant_tool(const json& payload) {
  AssistantTool tool;
  const std::string type = payload.value("type", "");
  if (type == "code_interpreter") {
    tool.type = AssistantTool::Type::CodeInterpreter;
  } else if (type == "file_search") {
    tool.type = AssistantTool::Type::FileSearch;
    AssistantTool::FileSearchOverrides overrides;
    if (payload.contains("file_search") && payload["file_search"].is_object()) {
      const auto& obj = payload.at("file_search");
      if (obj.contains("max_num_results") && obj["max_num_results"].is_number_integer()) {
        overrides.max_num_results = obj["max_num_results"].get<int>();
      }
      if (obj.contains("ranking_options") && obj["ranking_options"].is_object()) {
        const auto& ranking = obj.at("ranking_options");
        if (ranking.contains("ranker") && ranking["ranker"].is_string()) overrides.ranker = ranking["ranker"].get<std::string>();
        if (ranking.contains("score_threshold") && ranking["score_threshold"].is_number()) overrides.score_threshold = ranking["score_threshold"].get<double>();
      }
    }
    tool.file_search = overrides;
  } else if (type == "function") {
    tool.type = AssistantTool::Type::Function;
    if (payload.contains("function") && payload["function"].is_object()) {
      AssistantTool::FunctionDefinition fn;
      const auto& fn_obj = payload.at("function");
      fn.name = fn_obj.value("name", "");
      if (fn_obj.contains("description") && fn_obj["description"].is_string()) fn.description = fn_obj["description"].get<std::string>();
      if (fn_obj.contains("parameters")) fn.parameters = fn_obj.at("parameters");
      tool.function = fn;
    }
  }
  tool.raw = payload;
  return tool;
}

RunTruncationStrategy parse_truncation(const json& payload) {
  RunTruncationStrategy strategy;
  const std::string type = payload.value("type", "auto");
  strategy.type = type == "last_messages" ? RunTruncationStrategy::Type::LastMessages : RunTruncationStrategy::Type::Auto;
  if (payload.contains("last_messages") && payload["last_messages"].is_number_integer()) {
    strategy.last_messages = payload["last_messages"].get<int>();
  }
  return strategy;
}

RunUsage parse_usage(const json& payload) {
  RunUsage usage;
  usage.prompt_tokens = payload.value("prompt_tokens", 0);
  usage.completion_tokens = payload.value("completion_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  usage.extra = payload;
  return usage;
}

RunRequiredAction parse_required_action(const json& payload) {
  RunRequiredAction action;
  if (payload.contains("submit_tool_outputs") && payload["submit_tool_outputs"].is_object()) {
    const auto& submit = payload.at("submit_tool_outputs");
    if (submit.contains("tool_calls") && submit["tool_calls"].is_array()) {
      for (const auto& call : submit.at("tool_calls")) {
        RunRequiredAction::ToolCall tool_call;
        tool_call.id = call.value("id", "");
        if (call.contains("function") && call["function"].is_object()) {
          tool_call.function.name = call.at("function").value("name", "");
          tool_call.function.arguments = call.at("function").value("arguments", "");
        }
        action.tool_calls.push_back(tool_call);
      }
    }
  }
  return action;
}

Run parse_run_impl(const json& payload) {
  Run run;
  run.raw = payload;
  run.id = payload.value("id", "");
  run.assistant_id = payload.value("assistant_id", "");
  if (payload.contains("cancelled_at") && !payload["cancelled_at"].is_null()) run.cancelled_at = payload["cancelled_at"].get<int>();
  if (payload.contains("completed_at") && !payload["completed_at"].is_null()) run.completed_at = payload["completed_at"].get<int>();
  run.created_at = payload.value("created_at", 0);
  if (payload.contains("expires_at") && !payload["expires_at"].is_null()) run.expires_at = payload["expires_at"].get<int>();
  if (payload.contains("failed_at") && !payload["failed_at"].is_null()) run.failed_at = payload["failed_at"].get<int>();
  if (payload.contains("incomplete_details") && payload["incomplete_details"].is_object()) {
    RunIncompleteDetails details;
    details.reason = payload.at("incomplete_details").value("reason", "");
    run.incomplete_details = details;
  }
  run.instructions = payload.value("instructions", "");
  if (payload.contains("last_error") && payload["last_error"].is_object()) {
    RunLastError error;
    error.code = payload.at("last_error").value("code", "");
    error.message = payload.at("last_error").value("message", "");
    run.last_error = error;
  }
  if (payload.contains("max_completion_tokens") && !payload["max_completion_tokens"].is_null()) {
    run.max_completion_tokens = payload["max_completion_tokens"].get<int>();
  }
  if (payload.contains("max_prompt_tokens") && !payload["max_prompt_tokens"].is_null()) {
    run.max_prompt_tokens = payload["max_prompt_tokens"].get<int>();
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) run.metadata[it.key()] = it.value().get<std::string>();
    }
  }
  run.model = payload.value("model", "");
  run.object = payload.value("object", "");
  run.parallel_tool_calls = payload.value("parallel_tool_calls", false);
  if (payload.contains("required_action") && payload["required_action"].is_object()) {
    run.required_action = parse_required_action(payload.at("required_action"));
  }
  if (payload.contains("response_format") && payload["response_format"].is_object()) {
    run.response_format = parse_response_format(payload.at("response_format"));
  }
  if (payload.contains("started_at") && !payload["started_at"].is_null()) run.started_at = payload["started_at"].get<int>();
  run.status = payload.value("status", "");
  run.thread_id = payload.value("thread_id", "");
  if (payload.contains("tool_choice") && payload["tool_choice"].is_object()) {
    AssistantToolChoice choice;
    choice.type = payload.at("tool_choice").value("type", "");
    if (payload.at("tool_choice").contains("function") && payload.at("tool_choice").at("function").is_object()) {
      choice.function_name = payload.at("tool_choice").at("function").value("name", "");
    }
    choice.raw = payload.at("tool_choice");
    run.tool_choice = choice;
  }
  if (payload.contains("tools") && payload["tools"].is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
    run.tools.push_back(parse_assistant_tool(tool_json));
    }
  }
  if (payload.contains("truncation_strategy") && payload["truncation_strategy"].is_object()) {
    run.truncation_strategy = parse_truncation(payload.at("truncation_strategy"));
  }
  if (payload.contains("usage") && payload["usage"].is_object()) {
    run.usage = parse_usage(payload.at("usage"));
  }
  if (payload.contains("temperature") && payload["temperature"].is_number()) run.temperature = payload["temperature"].get<double>();
  if (payload.contains("top_p") && payload["top_p"].is_number()) run.top_p = payload["top_p"].get<double>();
  return run;
}

RunList parse_run_list_impl(const json& payload) {
  RunList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("first_id") && payload["first_id"].is_string()) list.first_id = payload["first_id"].get<std::string>();
  if (payload.contains("last_id") && payload["last_id"].is_string()) list.last_id = payload["last_id"].get<std::string>();
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) list.data.push_back(parse_run_impl(item));
  }
  return list;
}

Run parse_run_response(const std::string& body) {
  try {
    return parse_run_impl(json::parse(body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse run: ") + ex.what());
  }
}

}  // namespace

json build_run_create_body(const RunCreateRequest& request) {
  return build_run_create_body_impl(request);
}

Run parse_run_json(const nlohmann::json& payload) {
  return parse_run_impl(payload);
}

RunList parse_run_list_json(const nlohmann::json& payload) {
  return parse_run_list_impl(payload);
}

Run RunsResource::create(const std::string& thread_id, const RunCreateRequest& request) const {
  return create(thread_id, request, RequestOptions{});
}

Run RunsResource::create(const std::string& thread_id, const RunCreateRequest& request, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  if (request.include && !request.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < request.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*request.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }
  const auto body = build_run_create_body(request);
  auto response = client_.perform_request("POST", runs_path(thread_id), body.dump(), request_options);
  return parse_run_response(response.body);
}

Run RunsResource::retrieve(const std::string& thread_id, const std::string& run_id) const {
  return retrieve(thread_id, run_id, RequestOptions{});
}

Run RunsResource::retrieve(const std::string& thread_id, const std::string& run_id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", runs_path(thread_id) + "/" + run_id, "", request_options);
  return parse_run_response(response.body);
}

Run RunsResource::update(const std::string& thread_id, const std::string& run_id, const RunUpdateRequest& request) const {
  return update(thread_id, run_id, request, RequestOptions{});
}

Run RunsResource::update(const std::string& thread_id,
                         const std::string& run_id,
                         const RunUpdateRequest& request,
                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = update_request_to_json(request);
  auto response = client_.perform_request("POST", runs_path(thread_id) + "/" + run_id, body.dump(), request_options);
  return parse_run_response(response.body);
}

RunList RunsResource::list(const std::string& thread_id) const {
  return list(thread_id, RunListParams{}, RequestOptions{});
}

RunList RunsResource::list(const std::string& thread_id, const RunListParams& params) const {
  return list(thread_id, params, RequestOptions{});
}

RunList RunsResource::list(const std::string& thread_id, const RequestOptions& options) const {
  return list(thread_id, RunListParams{}, options);
}

RunList RunsResource::list(const std::string& thread_id,
                           const RunListParams& params,
                           const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.before) request_options.query_params["before"] = *params.before;
  if (params.status) request_options.query_params["status"] = *params.status;
  auto response = client_.perform_request("GET", runs_path(thread_id), "", request_options);
  try {
    return parse_run_list_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse run list: ") + ex.what());
  }
}

Run RunsResource::cancel(const std::string& thread_id, const std::string& run_id) const {
  return cancel(thread_id, run_id, RequestOptions{});
}

Run RunsResource::cancel(const std::string& thread_id, const std::string& run_id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("POST", runs_path(thread_id) + "/" + run_id + "/cancel", json::object().dump(),
                                          request_options);
  return parse_run_response(response.body);
}

Run RunsResource::submit_tool_outputs(const std::string& thread_id,
                                      const std::string& run_id,
                                      const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs(thread_id, run_id, request, RequestOptions{});
}

Run RunsResource::submit_tool_outputs(const std::string& thread_id,
                                      const std::string& run_id,
                                      const RunSubmitToolOutputsRequest& request,
                                      const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = submit_tool_outputs_to_json(request);
  auto response = client_.perform_request("POST", runs_path(thread_id) + "/" + run_id + "/submit_tool_outputs",
                                          body.dump(), request_options);
  return parse_run_response(response.body);
}

Run RunsResource::submit_tool_outputs(const std::string& run_id,
                                      const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs(run_id, request, RequestOptions{});
}

Run RunsResource::submit_tool_outputs(const std::string& run_id,
                                      const RunSubmitToolOutputsRequest& request,
                                      const RequestOptions& options) const {
  if (request.thread_id.empty()) {
    throw OpenAIError("RunSubmitToolOutputsRequest.thread_id must be set when using this helper");
  }
  return submit_tool_outputs(request.thread_id, run_id, request, options);
}

std::vector<AssistantStreamEvent> RunsResource::create_stream(const std::string& thread_id,
                                                              const RunCreateRequest& request) const {
  return create_stream_snapshot(thread_id, request).events();
}

std::vector<AssistantStreamEvent> RunsResource::create_stream(const std::string& thread_id,
                                                              const RunCreateRequest& request,
                                                              const RequestOptions& options) const {
  return create_stream_snapshot(thread_id, request, options).events();
}

AssistantStreamSnapshot RunsResource::create_stream_snapshot(const std::string& thread_id,
                                                             const RunCreateRequest& request) const {
  return create_stream_snapshot(thread_id, request, RequestOptions{});
}

AssistantStreamSnapshot RunsResource::create_stream_snapshot(const std::string& thread_id,
                                                             const RunCreateRequest& request,
                                                             const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  request_options.collect_body = false;
  request_options.headers["X-Stainless-Helper-Method"] = "stream";

  RunCreateRequest streaming_request = request;
  streaming_request.stream = true;

  SSEParser sse_parser;
  AssistantStreamSnapshot snapshot;
  AssistantStreamParser parser([&](const AssistantStreamEvent& ev) { snapshot.ingest(ev); });

  request_options.on_chunk = [&](const char* data, std::size_t size) {
    auto sse_events = sse_parser.feed(data, size);
    for (const auto& sse_event : sse_events) {
      parser.feed(sse_event);
    }
  };

  if (streaming_request.include && !streaming_request.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < streaming_request.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*streaming_request.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }

  const auto body = build_run_create_body(streaming_request);
  client_.perform_request("POST", runs_path(thread_id), body.dump(), request_options);

  auto remaining = sse_parser.finalize();
  for (const auto& sse_event : remaining) {
    parser.feed(sse_event);
  }

  return snapshot;
}

std::vector<AssistantStreamEvent> RunsResource::stream(const std::string& thread_id,
                                                       const RunCreateRequest& request) const {
  return stream(thread_id, request, RequestOptions{});
}

std::vector<AssistantStreamEvent> RunsResource::stream(const std::string& thread_id,
                                                       const RunCreateRequest& request,
                                                       const RequestOptions& options) const {
  return create_stream(thread_id, request, options);
}

std::vector<AssistantStreamEvent> RunsResource::submit_tool_outputs_stream(
    const std::string& thread_id, const std::string& run_id, const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_stream_snapshot(thread_id, run_id, request).events();
}

std::vector<AssistantStreamEvent> RunsResource::submit_tool_outputs_stream(
    const std::string& thread_id,
    const std::string& run_id,
    const RunSubmitToolOutputsRequest& request,
    const RequestOptions& options) const {
  return submit_tool_outputs_stream_snapshot(thread_id, run_id, request, options).events();
}

AssistantStreamSnapshot RunsResource::submit_tool_outputs_stream_snapshot(
    const std::string& thread_id,
    const std::string& run_id,
    const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_stream_snapshot(thread_id, run_id, request, RequestOptions{});
}

AssistantStreamSnapshot RunsResource::submit_tool_outputs_stream_snapshot(
    const std::string& thread_id,
    const std::string& run_id,
    const RunSubmitToolOutputsRequest& request,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  request_options.collect_body = false;
  request_options.headers["X-Stainless-Helper-Method"] = "stream";

  RunSubmitToolOutputsRequest streaming_request = request;
  streaming_request.stream = true;

  SSEParser sse_parser;
  AssistantStreamSnapshot snapshot;
  AssistantStreamParser parser([&](const AssistantStreamEvent& ev) { snapshot.ingest(ev); });

  request_options.on_chunk = [&](const char* data, std::size_t size) {
    auto sse_events = sse_parser.feed(data, size);
    for (const auto& sse_event : sse_events) {
      parser.feed(sse_event);
    }
  };

  const auto body = submit_tool_outputs_to_json(streaming_request);
  client_.perform_request("POST", runs_path(thread_id) + "/" + run_id + "/submit_tool_outputs", body.dump(), request_options);

  auto remaining = sse_parser.finalize();
  for (const auto& sse_event : remaining) {
    parser.feed(sse_event);
  }

  return snapshot;
}

std::vector<AssistantStreamEvent> RunsResource::submit_tool_outputs_stream(
    const std::string& run_id, const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_stream_snapshot(run_id, request).events();
}

std::vector<AssistantStreamEvent> RunsResource::submit_tool_outputs_stream(
    const std::string& run_id,
    const RunSubmitToolOutputsRequest& request,
    const RequestOptions& options) const {
  return submit_tool_outputs_stream_snapshot(run_id, request, options).events();
}

AssistantStreamSnapshot RunsResource::submit_tool_outputs_stream_snapshot(
    const std::string& run_id, const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_stream_snapshot(run_id, request, RequestOptions{});
}

AssistantStreamSnapshot RunsResource::submit_tool_outputs_stream_snapshot(
    const std::string& run_id,
    const RunSubmitToolOutputsRequest& request,
    const RequestOptions& options) const {
  if (request.thread_id.empty()) {
    throw OpenAIError("RunSubmitToolOutputsRequest.thread_id must be set when using this helper");
  }
  return submit_tool_outputs_stream_snapshot(request.thread_id, run_id, request, options);
}

namespace {

bool is_terminal_status(const std::string& status) {
  return status == "completed" || status == "requires_action" || status == "failed" || status == "cancelled" ||
         status == "incomplete" || status == "expired";
}

}

Run RunsResource::poll(const std::string& run_id, const RunRetrieveParams& params) const {
  return poll(run_id, params, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::poll(const std::string& run_id,
                       const RunRetrieveParams& params,
                       const RequestOptions& options,
                       std::chrono::milliseconds poll_interval) const {
  while (true) {
    RequestOptions request_options = options;
    apply_beta_header(request_options);
    auto response = client_.perform_request("GET", runs_path(params.thread_id) + "/" + run_id, "", request_options);
    auto run = parse_run_response(response.body);

    if (is_terminal_status(run.status)) {
      return run;
    }

    std::chrono::milliseconds sleep_for = poll_interval;
    auto header_it = response.headers.find("openai-poll-after-ms");
    if (header_it != response.headers.end()) {
      try {
        sleep_for = std::chrono::milliseconds(std::stoll(header_it->second));
      } catch (const std::exception&) {
        // ignore malformed header, fall back to configured interval
      }
    }

    if (sleep_for.count() > 0) {
      std::this_thread::sleep_for(sleep_for);
    }
  }
}

Run RunsResource::create_and_run_poll(const std::string& thread_id, const RunCreateRequest& request) const {
  return create_and_run_poll(thread_id, request, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::create_and_run_poll(const std::string& thread_id,
                                      const RunCreateRequest& request,
                                      const RequestOptions& options,
                                      std::chrono::milliseconds poll_interval) const {
  Run initial = create(thread_id, request, options);
  if (is_terminal_status(initial.status)) {
    return initial;
  }

  RunRetrieveParams retrieve_params;
  retrieve_params.thread_id = thread_id;
  return poll(initial.id, retrieve_params, options, poll_interval);
}

Run RunsResource::submit_tool_outputs_and_poll(const std::string& thread_id,
                                               const std::string& run_id,
                                               const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_and_poll(thread_id, run_id, request, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::submit_tool_outputs_and_poll(const std::string& thread_id,
                                               const std::string& run_id,
                                               const RunSubmitToolOutputsRequest& request,
                                               const RequestOptions& options,
                                               std::chrono::milliseconds poll_interval) const {
  Run intermediate = submit_tool_outputs(thread_id, run_id, request, options);
  if (is_terminal_status(intermediate.status)) {
    return intermediate;
  }

  RunRetrieveParams retrieve_params;
  retrieve_params.thread_id = thread_id;
  return poll(run_id, retrieve_params, options, poll_interval);
}

Run RunsResource::submit_tool_outputs_and_poll(const std::string& run_id,
                                               const RunSubmitToolOutputsRequest& request) const {
  return submit_tool_outputs_and_poll(run_id, request, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::submit_tool_outputs_and_poll(const std::string& run_id,
                                               const RunSubmitToolOutputsRequest& request,
                                               const RequestOptions& options,
                                               std::chrono::milliseconds poll_interval) const {
  if (request.thread_id.empty()) {
    throw OpenAIError("RunSubmitToolOutputsRequest.thread_id must be set when using this helper");
  }
  return submit_tool_outputs_and_poll(request.thread_id, run_id, request, options, poll_interval);
}

Run RunsResource::resolve_required_action(const Run& run, const ToolOutputGenerator& generator) const {
  return resolve_required_action(run, generator, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::resolve_required_action(const Run& run,
                                          const ToolOutputGenerator& generator,
                                          const RequestOptions& options,
                                          std::chrono::milliseconds poll_interval) const {
  if (!run.required_action.has_value()) {
    return run;
  }
  if (!generator) {
    throw OpenAIError("ToolOutputGenerator must be provided");
  }

  Run current = run;
  while (current.status == "requires_action") {
    const auto& required = current.required_action.value();
    auto outputs = generator(current, required);
    if (outputs.empty()) {
      throw OpenAIError("ToolOutputGenerator returned no tool outputs");
    }

    RunSubmitToolOutputsRequest submit_request;
    submit_request.thread_id = current.thread_id;
    submit_request.outputs = std::move(outputs);

    current = submit_tool_outputs_and_poll(current.id, submit_request, options, poll_interval);
  }
  return current;
}

Run RunsResource::create_and_run_auto(const std::string& thread_id,
                                      const RunCreateRequest& request,
                                      const ToolOutputGenerator& generator) const {
  return create_and_run_auto(thread_id, request, generator, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run RunsResource::create_and_run_auto(const std::string& thread_id,
                                      const RunCreateRequest& request,
                                      const ToolOutputGenerator& generator,
                                      const RequestOptions& options,
                                      std::chrono::milliseconds poll_interval) const {
  Run initial = create(thread_id, request, options);
  return resolve_required_action(initial, generator, options, poll_interval);
}

}  // namespace openai

#include "openai/run_steps.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

std::string run_steps_path(const std::string& thread_id, const std::string& run_id) {
  return "/threads/" + thread_id + "/runs/" + run_id + "/steps";
}

json include_to_json(const std::optional<std::vector<std::string>>& include) {
  json value = json::array();
  if (include) {
    for (const auto& entry : *include) value.push_back(entry);
  }
  return value;
}

CodeInterpreterToolCallDetails parse_code_tool_call(const json& payload) {
  CodeInterpreterToolCallDetails call;
  call.id = payload.value("id", "");
  if (payload.contains("code_interpreter") && payload["code_interpreter"].is_object()) {
    const auto& interpreter = payload.at("code_interpreter");
    call.input = interpreter.value("input", "");
    if (interpreter.contains("outputs") && interpreter["outputs"].is_array()) {
      for (const auto& output : interpreter.at("outputs")) {
        const std::string type = output.value("type", "");
        if (type == "logs") {
          CodeInterpreterLogOutput log;
          log.index = output.value("index", 0);
          log.logs = output.value("logs", "");
          call.log_outputs.push_back(log);
        } else if (type == "image") {
          CodeInterpreterImageOutput image;
          image.index = output.value("index", 0);
          if (output.contains("image") && output["image"].is_object()) {
            image.file_id = output.at("image").value("file_id", "");
          }
          call.image_outputs.push_back(image);
        }
      }
    }
  }
  return call;
}

FileSearchToolCallDetails parse_file_search_call(const json& payload) {
  FileSearchToolCallDetails call;
  call.id = payload.value("id", "");
  if (payload.contains("file_search") && payload["file_search"].is_object()) {
    const auto& search = payload.at("file_search");
    if (search.contains("ranking_options") && search["ranking_options"].is_object()) {
      const auto& ranking = search.at("ranking_options");
      FileSearchRankingOptions options;
      options.ranker = ranking.value("ranker", "");
      options.score_threshold = ranking.value("score_threshold", 0.0);
      call.ranking_options = options;
    }
    if (search.contains("results") && search["results"].is_array()) {
      for (const auto& result_json : search.at("results")) {
        FileSearchResult result;
        result.file_id = result_json.value("file_id", "");
        result.file_name = result_json.value("file_name", "");
        result.score = result_json.value("score", 0.0);
        if (result_json.contains("content") && result_json["content"].is_array()) {
          for (const auto& content : result_json.at("content")) {
            FileSearchResultContent item;
            item.type = content.value("type", "");
            if (content.contains("text") && content["text"].is_string()) item.text = content["text"].get<std::string>();
            result.content.push_back(item);
          }
        }
        call.results.push_back(result);
      }
    }
  }
  return call;
}

FunctionToolCallDetails parse_function_call(const json& payload) {
  FunctionToolCallDetails call;
  call.id = payload.value("id", "");
  if (payload.contains("function") && payload["function"].is_object()) {
    const auto& fn = payload.at("function");
    call.name = fn.value("name", "");
    call.arguments = fn.value("arguments", "");
    if (!fn.at("output").is_null()) call.output = fn.at("output").get<std::string>();
  }
  return call;
}

ToolCallDetails parse_tool_call(const json& payload) {
  ToolCallDetails details;
  const std::string type = payload.value("type", "");
  if (type == "code_interpreter") {
    details.type = ToolCallDetails::Type::CodeInterpreter;
    details.code_interpreter = parse_code_tool_call(payload);
  } else if (type == "file_search") {
    details.type = ToolCallDetails::Type::FileSearch;
    details.file_search = parse_file_search_call(payload);
  } else if (type == "function") {
    details.type = ToolCallDetails::Type::Function;
    details.function = parse_function_call(payload);
  }
  return details;
}

ToolCallDelta parse_tool_call_delta(const json& payload) {
  ToolCallDelta delta;
  const std::string type = payload.value("type", "");
  delta.index = payload.value("index", 0);
  if (payload.contains("id") && payload["id"].is_string()) delta.id = payload["id"].get<std::string>();
  if (type == "code_interpreter") {
    delta.type = ToolCallDetails::Type::CodeInterpreter;
    delta.code_interpreter = parse_code_tool_call(payload);
  } else if (type == "file_search") {
    delta.type = ToolCallDetails::Type::FileSearch;
    delta.file_search = parse_file_search_call(payload);
  } else if (type == "function") {
    delta.type = ToolCallDetails::Type::Function;
    delta.function = parse_function_call(payload);
  }
  return delta;
}

RunStepDetails parse_step_details(const json& payload) {
  RunStepDetails details;
  const std::string type = payload.value("type", "");
  if (type == "message_creation") {
    details.type = RunStepDetails::Type::MessageCreation;
    if (payload.contains("message_creation") && payload["message_creation"].is_object()) {
      MessageCreationDetails creation;
      creation.message_id = payload.at("message_creation").value("message_id", "");
      details.message_creation = creation;
    }
  } else if (type == "tool_calls") {
    details.type = RunStepDetails::Type::ToolCalls;
    if (payload.contains("tool_calls") && payload["tool_calls"].is_array()) {
      for (const auto& call : payload.at("tool_calls")) {
        details.tool_calls.push_back(parse_tool_call(call));
      }
    }
  }
  return details;
}

RunStepUsage parse_usage(const json& payload) {
  RunStepUsage usage;
  usage.completion_tokens = payload.value("completion_tokens", 0);
  usage.prompt_tokens = payload.value("prompt_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  return usage;
}

RunStep parse_run_step_impl(const json& payload) {
  RunStep step;
  step.raw = payload;
  step.id = payload.value("id", "");
  step.assistant_id = payload.value("assistant_id", "");
  if (payload.contains("cancelled_at") && !payload["cancelled_at"].is_null()) step.cancelled_at = payload["cancelled_at"].get<int>();
  if (payload.contains("completed_at") && !payload["completed_at"].is_null()) step.completed_at = payload["completed_at"].get<int>();
  step.created_at = payload.value("created_at", 0);
  if (payload.contains("expired_at") && !payload["expired_at"].is_null()) step.expired_at = payload["expired_at"].get<int>();
  if (payload.contains("failed_at") && !payload["failed_at"].is_null()) step.failed_at = payload["failed_at"].get<int>();
  if (payload.contains("last_error") && payload["last_error"].is_object()) {
    RunLastError error;
    error.code = payload.at("last_error").value("code", "");
    error.message = payload.at("last_error").value("message", "");
    step.last_error = error;
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) step.metadata[it.key()] = it.value().get<std::string>();
    }
  }
  step.object = payload.value("object", "");
  step.run_id = payload.value("run_id", "");
  step.status = payload.value("status", "");
  if (payload.contains("step_details") && payload["step_details"].is_object()) {
    step.details = parse_step_details(payload.at("step_details"));
  }
  step.thread_id = payload.value("thread_id", "");
  if (payload.contains("usage") && payload["usage"].is_object()) {
    step.usage = parse_usage(payload.at("usage"));
  }
  return step;
}

RunStepList parse_run_step_list_impl(const json& payload) {
  RunStepList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("first_id") && payload["first_id"].is_string()) list.first_id = payload["first_id"].get<std::string>();
  if (payload.contains("last_id") && payload["last_id"].is_string()) list.last_id = payload["last_id"].get<std::string>();
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) list.data.push_back(parse_run_step_impl(item));
  }
  return list;
}

RunStep parse_run_step_response(const std::string& body) {
  try {
    return parse_run_step_impl(json::parse(body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse run step: ") + ex.what());
  }
}

RunStepList parse_run_step_list_response(const std::string& body) {
  try {
    return parse_run_step_list_impl(json::parse(body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse run step list: ") + ex.what());
  }
}

RunStepDelta parse_run_step_delta_impl(const json& payload) {
  RunStepDelta delta;
  delta.raw = payload;
  if (payload.contains("step_details") && payload["step_details"].is_object()) {
    RunStepDeltaDetails details;
    const auto& details_json = payload.at("step_details");
    const std::string type = details_json.value("type", "");
    if (type == "message_creation") {
      details.type = RunStepDeltaDetails::Type::MessageCreation;
      if (details_json.contains("message_creation") && details_json["message_creation"].is_object()) {
        MessageCreationDeltaDetails creation;
        creation.message_id = details_json.at("message_creation").value("message_id", std::string());
        details.message_creation = creation;
      }
    } else {
      details.type = RunStepDeltaDetails::Type::ToolCalls;
      if (details_json.contains("tool_calls") && details_json["tool_calls"].is_array()) {
        for (const auto& call : details_json.at("tool_calls")) {
          details.tool_calls.push_back(parse_tool_call_delta(call));
        }
      }
    }
    delta.details = details;
  }
  return delta;
}

RunStepDeltaEvent parse_run_step_delta_event_impl(const nlohmann::json& payload) {
  RunStepDeltaEvent event;
  event.raw = payload;
  event.id = payload.value("id", "");
  event.object = payload.value("object", "");
  if (payload.contains("delta") && payload["delta"].is_object()) {
    event.delta = parse_run_step_delta_impl(payload.at("delta"));
  }
  return event;
}

}  // namespace

RunStep parse_run_step_json(const nlohmann::json& payload) {
  return parse_run_step_impl(payload);
}

RunStepList parse_run_step_list_json(const nlohmann::json& payload) {
  return parse_run_step_list_impl(payload);
}

RunStepDeltaEvent parse_run_step_delta_json(const nlohmann::json& payload) {
  return parse_run_step_delta_event_impl(payload);
}

RunStep RunStepsResource::retrieve(const std::string& run_id,
                                   const std::string& step_id,
                                   const RunStepRetrieveParams& params) const {
  return retrieve(run_id, step_id, params, RequestOptions{});
}

RunStep RunStepsResource::retrieve(const std::string& run_id,
                                   const std::string& step_id,
                                   const RunStepRetrieveParams& params,
                                   const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  std::string url = run_steps_path(params.thread_id, run_id) + "/" + step_id;
  if (params.include && !params.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < params.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*params.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }
  auto response = client_.perform_request("GET", url, "", request_options);
  return parse_run_step_response(response.body);
}

RunStepList RunStepsResource::list(const std::string& run_id, const RunStepListParams& params) const {
  return list(run_id, params, RequestOptions{});
}

RunStepList RunStepsResource::list(const std::string& run_id,
                                   const RunStepListParams& params,
                                   const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.before) request_options.query_params["before"] = *params.before;
  if (params.include && !params.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < params.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*params.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }
  auto response = client_.perform_request("GET", run_steps_path(params.thread_id, run_id), "", request_options);
  return parse_run_step_list_response(response.body);
}

}  // namespace openai

#include "openai/chatkit.hpp"

#include "openai/beta.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kChatKitBetaHeader = "chatkit_beta=v1";
constexpr const char* kChatKitHeaderName = "OpenAI-Beta";

void apply_chatkit_beta_header(RequestOptions& options) {
  options.headers[kChatKitHeaderName] = kChatKitBetaHeader;
}

json state_variables_to_json(const std::optional<std::map<std::string, beta::ChatKitStateVariableValue>>& values) {
  if (!values || values->empty()) return json::object();

  json object = json::object();
  for (const auto& entry : *values) {
    const auto& key = entry.first;
    const auto& value = entry.second;
    std::visit(
        [&](auto&& arg) {
          object[key] = arg;
        },
        value);
  }
  return object;
}

std::optional<std::map<std::string, beta::ChatKitStateVariableValue>> parse_state_variables(const json& payload) {
  if (!payload.is_object()) return std::nullopt;

  std::map<std::string, beta::ChatKitStateVariableValue> result;
  for (const auto& [key, value] : payload.items()) {
    if (value.is_boolean()) {
      result.emplace(key, value.get<bool>());
    } else if (value.is_number_float() || value.is_number_integer()) {
      result.emplace(key, value.get<double>());
    } else if (value.is_string()) {
      result.emplace(key, value.get<std::string>());
    }
  }
  return result;
}

json workflow_param_to_json(const beta::ChatKitSessionWorkflowParam& workflow) {
  json payload = json::object();
  payload["id"] = workflow.id;

  if (workflow.state_variables) {
    auto state_variables_json = state_variables_to_json(workflow.state_variables);
    if (!state_variables_json.empty()) payload["state_variables"] = std::move(state_variables_json);
  }

  if (workflow.tracing) {
    json tracing = json::object();
    if (workflow.tracing->enabled) tracing["enabled"] = *workflow.tracing->enabled;
    if (!tracing.empty()) payload["tracing"] = std::move(tracing);
  }

  if (workflow.version) payload["version"] = *workflow.version;
  return payload;
}

json chatkit_configuration_to_json(const beta::ChatKitSessionChatKitConfigurationParam& configuration) {
  json payload = json::object();

  if (configuration.automatic_thread_titling) {
    json titling = json::object();
    if (configuration.automatic_thread_titling->enabled) {
      titling["enabled"] = *configuration.automatic_thread_titling->enabled;
    }
    if (!titling.empty()) payload["automatic_thread_titling"] = std::move(titling);
  }

  if (configuration.file_upload) {
    json upload = json::object();
    if (configuration.file_upload->enabled) upload["enabled"] = *configuration.file_upload->enabled;
    if (configuration.file_upload->max_file_size) {
      upload["max_file_size"] = *configuration.file_upload->max_file_size;
    }
    if (configuration.file_upload->max_files) upload["max_files"] = *configuration.file_upload->max_files;
    if (!upload.empty()) payload["file_upload"] = std::move(upload);
  }

  if (configuration.history) {
    json history = json::object();
    if (configuration.history->enabled) history["enabled"] = *configuration.history->enabled;
    if (configuration.history->recent_threads) history["recent_threads"] = *configuration.history->recent_threads;
    if (!history.empty()) payload["history"] = std::move(history);
  }

  return payload;
}

json session_create_body(const beta::ChatKitSessionCreateParams& params) {
  json payload = json::object();
  payload["user"] = params.user;
  payload["workflow"] = workflow_param_to_json(params.workflow);

  if (params.chatkit_configuration) {
    auto config = chatkit_configuration_to_json(*params.chatkit_configuration);
    if (!config.empty()) payload["chatkit_configuration"] = std::move(config);
  }

  if (params.expires_after) {
    json expires = json::object();
    expires["anchor"] = params.expires_after->anchor;
    expires["seconds"] = params.expires_after->seconds;
    payload["expires_after"] = std::move(expires);
  }

  if (params.rate_limits && params.rate_limits->max_requests_per_1_minute) {
    json rate_limits = json::object();
    rate_limits["max_requests_per_1_minute"] = *params.rate_limits->max_requests_per_1_minute;
    payload["rate_limits"] = std::move(rate_limits);
  }

  return payload;
}

beta::ChatKitWorkflowTracing parse_chatkit_tracing(const json& payload) {
  beta::ChatKitWorkflowTracing tracing;
  if (payload.contains("enabled") && payload.at("enabled").is_boolean()) {
    tracing.enabled = payload.at("enabled").get<bool>();
  }
  return tracing;
}

beta::ChatKitWorkflow parse_chatkit_workflow(const json& payload) {
  beta::ChatKitWorkflow workflow;
  workflow.raw = payload;
  workflow.id = payload.value("id", "");
  if (payload.contains("state_variables")) {
    workflow.state_variables = parse_state_variables(payload.at("state_variables"));
  }
  if (payload.contains("tracing") && payload.at("tracing").is_object()) {
    workflow.tracing = parse_chatkit_tracing(payload.at("tracing"));
  }
  if (payload.contains("version") && payload.at("version").is_string()) {
    workflow.version = payload.at("version").get<std::string>();
  }
  return workflow;
}

beta::ChatKitSessionStatus parse_session_status(const std::string& status) {
  if (status == "active") return beta::ChatKitSessionStatus::Active;
  if (status == "expired") return beta::ChatKitSessionStatus::Expired;
  if (status == "cancelled") return beta::ChatKitSessionStatus::Cancelled;
  return beta::ChatKitSessionStatus::Unknown;
}

beta::ChatKitSessionChatKitConfiguration parse_chatkit_configuration(const json& payload) {
  beta::ChatKitSessionChatKitConfiguration configuration;
  configuration.raw = payload;

  if (payload.contains("automatic_thread_titling") && payload.at("automatic_thread_titling").is_object()) {
    const auto& titling = payload.at("automatic_thread_titling");
    configuration.automatic_thread_titling.enabled = titling.value("enabled", true);
  }

  if (payload.contains("file_upload") && payload.at("file_upload").is_object()) {
    const auto& upload = payload.at("file_upload");
    configuration.file_upload.enabled = upload.value("enabled", false);
    if (upload.contains("max_file_size") && !upload.at("max_file_size").is_null()) {
      configuration.file_upload.max_file_size = upload.at("max_file_size").get<int>();
    }
    if (upload.contains("max_files") && !upload.at("max_files").is_null()) {
      configuration.file_upload.max_files = upload.at("max_files").get<int>();
    }
  }

  if (payload.contains("history") && payload.at("history").is_object()) {
    const auto& history = payload.at("history");
    configuration.history.enabled = history.value("enabled", true);
    if (history.contains("recent_threads") && !history.at("recent_threads").is_null()) {
      configuration.history.recent_threads = history.at("recent_threads").get<int>();
    }
  }

  return configuration;
}

beta::ChatKitSessionRateLimits parse_chatkit_rate_limits(const json& payload) {
  beta::ChatKitSessionRateLimits rate_limits;
  rate_limits.raw = payload;
  if (payload.contains("max_requests_per_1_minute") && !payload.at("max_requests_per_1_minute").is_null()) {
    rate_limits.max_requests_per_1_minute = payload.at("max_requests_per_1_minute").get<int>();
  }
  return rate_limits;
}

beta::ChatKitSession parse_chatkit_session(const json& payload) {
  beta::ChatKitSession session;
  session.raw = payload;
  session.id = payload.value("id", "");
  session.object = payload.value("object", "");
  session.expires_at = payload.value("expires_at", 0);
  if (payload.contains("client_secret") && payload.at("client_secret").is_string()) {
    session.client_secret = payload.at("client_secret").get<std::string>();
  }
  if (payload.contains("max_requests_per_1_minute") && !payload.at("max_requests_per_1_minute").is_null()) {
    session.max_requests_per_1_minute = payload.at("max_requests_per_1_minute").get<int>();
  }
  session.status = parse_session_status(payload.value("status", ""));
  session.user = payload.value("user", "");
  if (payload.contains("workflow") && payload.at("workflow").is_object()) {
    session.workflow = parse_chatkit_workflow(payload.at("workflow"));
  }
  if (payload.contains("chatkit_configuration") && payload.at("chatkit_configuration").is_object()) {
    session.chatkit_configuration = parse_chatkit_configuration(payload.at("chatkit_configuration"));
  }
  if (payload.contains("rate_limits") && payload.at("rate_limits").is_object()) {
    session.rate_limits = parse_chatkit_rate_limits(payload.at("rate_limits"));
  }
  return session;
}

beta::ChatKitThreadStatusType parse_thread_status_type(const std::string& type) {
  if (type == "active") return beta::ChatKitThreadStatusType::Active;
  if (type == "locked") return beta::ChatKitThreadStatusType::Locked;
  if (type == "closed") return beta::ChatKitThreadStatusType::Closed;
  return beta::ChatKitThreadStatusType::Unknown;
}

beta::ChatKitThreadStatus parse_chatkit_thread_status(const json& payload) {
  beta::ChatKitThreadStatus status;
  status.type = parse_thread_status_type(payload.value("type", ""));
  if (payload.contains("reason") && !payload.at("reason").is_null()) {
    status.reason = payload.at("reason").get<std::string>();
  }
  return status;
}

beta::ChatKitThread parse_chatkit_thread(const json& payload) {
  beta::ChatKitThread thread;
  thread.raw = payload;
  thread.id = payload.value("id", "");
  thread.created_at = payload.value("created_at", 0);
  thread.object = payload.value("object", "");
  if (payload.contains("status") && payload.at("status").is_object()) {
    thread.status = parse_chatkit_thread_status(payload.at("status"));
  }
  if (payload.contains("title") && !payload.at("title").is_null()) {
    thread.title = payload.at("title").get<std::string>();
  }
  thread.user = payload.value("user", "");
  return thread;
}

beta::ChatKitAttachment parse_chatkit_attachment(const json& payload) {
  beta::ChatKitAttachment attachment;
  attachment.id = payload.value("id", "");
  attachment.mime_type = payload.value("mime_type", "");
  attachment.name = payload.value("name", "");
  if (payload.contains("preview_url") && !payload.at("preview_url").is_null()) {
    attachment.preview_url = payload.at("preview_url").get<std::string>();
  }
  attachment.type = payload.value("type", "");
  return attachment;
}

std::optional<beta::ChatKitResponseOutputTextAnnotation> parse_chatkit_annotation(const json& payload) {
  const auto type = payload.value("type", "");
  if (type == "file") {
    beta::ChatKitResponseOutputTextAnnotationFile annotation;
    if (payload.contains("source") && payload.at("source").is_object()) {
      const auto& source = payload.at("source");
      annotation.source.filename = source.value("filename", "");
      annotation.source.type = source.value("type", "");
    }
    annotation.type = type;
    return annotation;
  }
  if (type == "url") {
    beta::ChatKitResponseOutputTextAnnotationURL annotation;
    if (payload.contains("source") && payload.at("source").is_object()) {
      const auto& source = payload.at("source");
      annotation.source.type = source.value("type", "");
      annotation.source.url = source.value("url", "");
    }
    annotation.type = type;
    return annotation;
  }
  return std::nullopt;
}

beta::ChatKitResponseOutputText parse_chatkit_response_output_text(const json& payload) {
  beta::ChatKitResponseOutputText text;
  text.raw = payload;
  text.text = payload.value("text", "");
  text.type = payload.value("type", "");
  if (payload.contains("annotations") && payload.at("annotations").is_array()) {
    for (const auto& annotation : payload.at("annotations")) {
      if (auto parsed = parse_chatkit_annotation(annotation)) {
        text.annotations.push_back(std::move(*parsed));
      }
    }
  }
  return text;
}

beta::ChatKitThreadUserMessageContent parse_chatkit_user_message_content(const json& payload) {
  beta::ChatKitThreadUserMessageContent content;
  const auto type = payload.value("type", "");
  if (type == "input_text") {
    content.type = beta::ChatKitThreadUserMessageContent::Type::InputText;
  } else if (type == "quoted_text") {
    content.type = beta::ChatKitThreadUserMessageContent::Type::QuotedText;
  } else {
    content.type = beta::ChatKitThreadUserMessageContent::Type::Unknown;
  }
  content.text = payload.value("text", "");
  return content;
}

std::optional<beta::ChatKitThreadUserMessageItem::InferenceOptions> parse_inference_options(const json& payload) {
  if (payload.is_null() || !payload.is_object()) return std::nullopt;

  beta::ChatKitThreadUserMessageItem::InferenceOptions options;
  if (payload.contains("model") && !payload.at("model").is_null()) {
    options.model = payload.at("model").get<std::string>();
  }
  if (payload.contains("tool_choice") && payload.at("tool_choice").is_object()) {
    beta::ChatKitThreadUserMessageItem::InferenceOptions::ToolChoice tool_choice;
    tool_choice.id = payload.at("tool_choice").value("id", "");
    options.tool_choice = tool_choice;
  }
  return options;
}

beta::ChatKitThreadUserMessageItem parse_chatkit_user_message_item(const json& payload) {
  beta::ChatKitThreadUserMessageItem item;
  item.id = payload.value("id", "");
  item.created_at = payload.value("created_at", 0);
  item.object = payload.value("object", "");
  item.thread_id = payload.value("thread_id", "");
  item.type = payload.value("type", "");

  if (payload.contains("attachments") && payload.at("attachments").is_array()) {
    for (const auto& attachment : payload.at("attachments")) {
      item.attachments.push_back(parse_chatkit_attachment(attachment));
    }
  }

  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& content : payload.at("content")) {
      item.content.push_back(parse_chatkit_user_message_content(content));
    }
  }

  if (payload.contains("inference_options")) {
    item.inference_options = parse_inference_options(payload.at("inference_options"));
  }

  return item;
}

beta::ChatKitThreadAssistantMessageItem parse_chatkit_assistant_message_item(const json& payload) {
  beta::ChatKitThreadAssistantMessageItem item;
  item.id = payload.value("id", "");
  item.created_at = payload.value("created_at", 0);
  item.object = payload.value("object", "");
  item.thread_id = payload.value("thread_id", "");
  item.type = payload.value("type", "");

  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& content : payload.at("content")) {
      item.content.push_back(parse_chatkit_response_output_text(content));
    }
  }

  return item;
}

beta::ChatKitWidgetItem parse_chatkit_widget_item(const json& payload) {
  beta::ChatKitWidgetItem item;
  item.id = payload.value("id", "");
  item.created_at = payload.value("created_at", 0);
  item.object = payload.value("object", "");
  item.thread_id = payload.value("thread_id", "");
  item.type = payload.value("type", "");
  item.widget = payload.value("widget", "");
  return item;
}

beta::ChatKitThreadClientToolCall parse_chatkit_client_tool_call(const json& payload) {
  beta::ChatKitThreadClientToolCall call;
  call.id = payload.value("id", "");
  call.arguments = payload.value("arguments", "");
  call.call_id = payload.value("call_id", "");
  call.created_at = payload.value("created_at", 0);
  call.name = payload.value("name", "");
  call.object = payload.value("object", "");
  if (payload.contains("output") && !payload.at("output").is_null()) {
    call.output = payload.at("output").dump();
  }
  call.status = payload.value("status", "");
  call.thread_id = payload.value("thread_id", "");
  call.type = payload.value("type", "");
  return call;
}

beta::ChatKitThreadTask parse_chatkit_task(const json& payload) {
  beta::ChatKitThreadTask task;
  task.id = payload.value("id", "");
  task.created_at = payload.value("created_at", 0);
  if (payload.contains("heading") && !payload.at("heading").is_null()) {
    task.heading = payload.at("heading").get<std::string>();
  }
  task.object = payload.value("object", "");
  if (payload.contains("summary") && !payload.at("summary").is_null()) {
    task.summary = payload.at("summary").get<std::string>();
  }
  task.task_type = payload.value("task_type", "");
  task.thread_id = payload.value("thread_id", "");
  task.type = payload.value("type", "");
  return task;
}

beta::ChatKitThreadTaskGroup parse_chatkit_task_group(const json& payload) {
  beta::ChatKitThreadTaskGroup group;
  group.id = payload.value("id", "");
  group.created_at = payload.value("created_at", 0);
  group.object = payload.value("object", "");
  group.thread_id = payload.value("thread_id", "");
  group.type = payload.value("type", "");

  if (payload.contains("tasks") && payload.at("tasks").is_array()) {
    for (const auto& task_payload : payload.at("tasks")) {
      beta::ChatKitThreadTaskGroup::Task task;
      if (task_payload.contains("heading") && !task_payload.at("heading").is_null()) {
        task.heading = task_payload.at("heading").get<std::string>();
      }
      if (task_payload.contains("summary") && !task_payload.at("summary").is_null()) {
        task.summary = task_payload.at("summary").get<std::string>();
      }
      task.type = task_payload.value("type", "");
      group.tasks.push_back(std::move(task));
    }
  }

  return group;
}

beta::ChatKitThreadItem parse_chatkit_thread_item(const json& payload) {
  beta::ChatKitThreadItem item;
  item.raw = payload;
  const auto type = payload.value("type", "");

  if (type == "chatkit.assistant_message") {
    item.kind = beta::ChatKitThreadItem::Kind::AssistantMessage;
    item.assistant_message = parse_chatkit_assistant_message_item(payload);
  } else if (type == "chatkit.user_message") {
    item.kind = beta::ChatKitThreadItem::Kind::UserMessage;
    item.user_message = parse_chatkit_user_message_item(payload);
  } else if (type == "chatkit.widget") {
    item.kind = beta::ChatKitThreadItem::Kind::Widget;
    item.widget = parse_chatkit_widget_item(payload);
  } else if (type == "chatkit.client_tool_call") {
    item.kind = beta::ChatKitThreadItem::Kind::ClientToolCall;
    item.client_tool_call = parse_chatkit_client_tool_call(payload);
  } else if (type == "chatkit.task") {
    item.kind = beta::ChatKitThreadItem::Kind::Task;
    item.task = parse_chatkit_task(payload);
  } else if (type == "chatkit.task_group") {
    item.kind = beta::ChatKitThreadItem::Kind::TaskGroup;
    item.task_group = parse_chatkit_task_group(payload);
  } else {
    item.kind = beta::ChatKitThreadItem::Kind::Unknown;
  }

  return item;
}

beta::ChatKitThreadList parse_chatkit_thread_list(const json& payload) {
  beta::ChatKitThreadList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_chatkit_thread(entry));
    }
  }
  if (payload.contains("first_id") && !payload.at("first_id").is_null()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && !payload.at("last_id").is_null()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  if (payload.contains("next_cursor") && !payload.at("next_cursor").is_null()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("object") && payload.at("object").is_string()) {
    list.object = payload.at("object").get<std::string>();
  }
  return list;
}

beta::ChatKitThreadItemList parse_chatkit_thread_item_list(const json& payload) {
  beta::ChatKitThreadItemList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_chatkit_thread_item(entry));
    }
  }
  if (payload.contains("first_id") && !payload.at("first_id").is_null()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && !payload.at("last_id").is_null()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  if (payload.contains("next_cursor") && !payload.at("next_cursor").is_null()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  list.object = payload.value("object", "");
  return list;
}

beta::ChatKitThreadDeleteResponse parse_chatkit_thread_delete(const json& payload) {
  beta::ChatKitThreadDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

json build_list_query(const beta::ChatKitThreadListParams& params) {
  json query = json::object();
  if (params.limit) query["limit"] = *params.limit;
  if (params.after) query["after"] = *params.after;
  if (params.before) query["before"] = *params.before;
  if (params.order) query["order"] = *params.order;
  if (params.user) query["user"] = *params.user;
  return query;
}

json build_list_items_query(const beta::ChatKitThreadListItemsParams& params) {
  json query = json::object();
  if (params.limit) query["limit"] = *params.limit;
  if (params.after) query["after"] = *params.after;
  if (params.before) query["before"] = *params.before;
  if (params.order) query["order"] = *params.order;
  return query;
}

}  // namespace

beta::ChatKitSession beta::ChatKitSessionsResource::create(const ChatKitSessionCreateParams& params) const {
  return create(params, RequestOptions{});
}

beta::ChatKitSession beta::ChatKitSessionsResource::create(const ChatKitSessionCreateParams& params,
                                                           const RequestOptions& options) const {
  auto body = session_create_body(params).dump();
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto response = client_.perform_request("POST", "/chatkit/sessions", body, request_options);
  try {
    return parse_chatkit_session(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit session response: ") + ex.what());
  }
}

beta::ChatKitSession beta::ChatKitSessionsResource::cancel(const std::string& session_id) const {
  return cancel(session_id, RequestOptions{});
}

beta::ChatKitSession beta::ChatKitSessionsResource::cancel(const std::string& session_id,
                                                           const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto response = client_.perform_request("POST", "/chatkit/sessions/" + session_id + "/cancel", "", request_options);
  try {
    return parse_chatkit_session(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit session cancel response: ") + ex.what());
  }
}

beta::ChatKitThread beta::ChatKitThreadsResource::retrieve(const std::string& thread_id) const {
  return retrieve(thread_id, RequestOptions{});
}

beta::ChatKitThread beta::ChatKitThreadsResource::retrieve(const std::string& thread_id,
                                                           const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto response = client_.perform_request("GET", "/chatkit/threads/" + thread_id, "", request_options);
  try {
    return parse_chatkit_thread(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit thread response: ") + ex.what());
  }
}

beta::ChatKitThreadList beta::ChatKitThreadsResource::list() const {
  return list(ChatKitThreadListParams{});
}

beta::ChatKitThreadList beta::ChatKitThreadsResource::list(const ChatKitThreadListParams& params) const {
  return list(params, RequestOptions{});
}

beta::ChatKitThreadList beta::ChatKitThreadsResource::list(const ChatKitThreadListParams& params,
                                                           const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto query = build_list_query(params);
  if (!query.empty()) request_options.query = query;
  auto response = client_.perform_request("GET", "/chatkit/threads", "", request_options);
  try {
    return parse_chatkit_thread_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit thread list: ") + ex.what());
  }
}

beta::ChatKitThreadList beta::ChatKitThreadsResource::list(const RequestOptions& options) const {
  return list(ChatKitThreadListParams{}, options);
}

beta::ChatKitThreadDeleteResponse beta::ChatKitThreadsResource::remove(const std::string& thread_id) const {
  return remove(thread_id, RequestOptions{});
}

beta::ChatKitThreadDeleteResponse beta::ChatKitThreadsResource::remove(const std::string& thread_id,
                                                                       const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto response = client_.perform_request("DELETE", "/chatkit/threads/" + thread_id, "", request_options);
  try {
    return parse_chatkit_thread_delete(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit thread delete response: ") + ex.what());
  }
}

beta::ChatKitThreadItemList beta::ChatKitThreadsResource::list_items(const std::string& thread_id) const {
  return list_items(thread_id, ChatKitThreadListItemsParams{});
}

beta::ChatKitThreadItemList beta::ChatKitThreadsResource::list_items(
    const std::string& thread_id,
    const ChatKitThreadListItemsParams& params) const {
  return list_items(thread_id, params, RequestOptions{});
}

beta::ChatKitThreadItemList beta::ChatKitThreadsResource::list_items(
    const std::string& thread_id,
    const ChatKitThreadListItemsParams& params,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_chatkit_beta_header(request_options);
  auto query = build_list_items_query(params);
  if (!query.empty()) request_options.query = query;
  auto response =
      client_.perform_request("GET", "/chatkit/threads/" + thread_id + "/items", "", request_options);
  try {
    return parse_chatkit_thread_item_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse ChatKit thread item list: ") + ex.what());
  }
}

beta::ChatKitThreadItemList beta::ChatKitThreadsResource::list_items(const std::string& thread_id,
                                                                     const RequestOptions& options) const {
  return list_items(thread_id, ChatKitThreadListItemsParams{}, options);
}

}  // namespace openai

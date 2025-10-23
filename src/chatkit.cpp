#include "openai/chatkit.hpp"

#include "openai/beta.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/utils/qs.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kChatKitBetaHeader = "chatkit_beta=v1";
constexpr const char* kChatKitHeaderName = "OpenAI-Beta";

void apply_chatkit_beta_header(RequestOptions& options) {
  options.headers[kChatKitHeaderName] = kChatKitBetaHeader;
}

json workflow_param_to_json(const beta::ChatKitSessionWorkflowParam& workflow) {
  json payload = json::object();
  payload["id"] = workflow.id;
  if (workflow.state_variables && workflow.state_variables->is_object()) {
    payload["state_variables"] = *workflow.state_variables;
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

beta::ChatKitSession parse_chatkit_session(const json& payload) {
  beta::ChatKitSession session;
  session.raw = payload;
  session.id = payload.value("id", "");
  session.object = payload.value("object", "");
  session.expires_at = payload.value("expires_at", 0);
  if (payload.contains("client_secret") && payload.at("client_secret").is_string()) {
    session.client_secret = payload.at("client_secret").get<std::string>();
  }
  if (payload.contains("max_requests_per_1_minute") && payload.at("max_requests_per_1_minute").is_number_integer()) {
    session.max_requests_per_1_minute = payload.at("max_requests_per_1_minute").get<int>();
  }
  session.status = payload.value("status", "");
  session.user = payload.value("user", "");
  if (payload.contains("workflow")) session.workflow = payload.at("workflow");
  if (payload.contains("chatkit_configuration")) session.configuration = payload.at("chatkit_configuration");
  return session;
}

beta::ChatKitThread parse_chatkit_thread(const json& payload) {
  beta::ChatKitThread thread;
  thread.raw = payload;
  thread.id = payload.value("id", "");
  thread.object = payload.value("object", "");
  thread.created_at = payload.value("created_at", 0);
  thread.status = payload.value("status", "");
  thread.user = payload.value("user", "");
  if (payload.contains("workflow")) {
    thread.workflow.raw = payload.at("workflow");
  }
  return thread;
}

beta::ChatKitThreadList parse_chatkit_thread_list(const json& payload) {
  beta::ChatKitThreadList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_chatkit_thread(item));
    }
  }
  return list;
}

beta::ChatKitThreadItem parse_chatkit_thread_item(const json& payload) {
  beta::ChatKitThreadItem item;
  item.raw = payload;
  item.id = payload.value("id", "");
  item.object = payload.value("object", "");
  item.type = payload.value("type", "");
  return item;
}

beta::ChatKitThreadItemList parse_chatkit_thread_item_list(const json& payload) {
  beta::ChatKitThreadItemList items;
  items.raw = payload;
  items.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    items.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      items.data.push_back(parse_chatkit_thread_item(entry));
    }
  }
  return items;
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
  return query;
}

json build_list_items_query(const beta::ChatKitThreadListItemsParams& params) {
  json query = json::object();
  if (params.limit) query["limit"] = *params.limit;
  if (params.after) query["after"] = *params.after;
  if (params.before) query["before"] = *params.before;
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

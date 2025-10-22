#include "openai/conversations.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

Conversation parse_conversation(const json& payload) {
  Conversation convo;
  convo.raw = payload;
  convo.id = payload.value("id", "");
  convo.created_at = payload.value("created_at", 0);
  if (payload.contains("metadata")) {
    convo.metadata = payload.at("metadata");
  }
  convo.object = payload.value("object", "");
  return convo;
}

ConversationDeleted parse_conversation_deleted(const json& payload) {
  ConversationDeleted deleted;
  deleted.raw = payload;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
}

ConversationItem parse_item(const json& payload) {
  ConversationItem item;
  if (payload.is_object()) {
    item.type = payload.value("type", "");
    item.raw = payload;
  }
  return item;
}

ConversationItemList parse_item_list(const json& payload) {
  ConversationItemList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_item(entry));
    }
  }
  return list;
}

ConversationItemPage parse_item_page(const json& payload) {
  ConversationItemPage page;
  page.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      page.data.push_back(parse_item(entry));
    }
  }
  page.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    page.next_cursor = payload.at("last_id").get<std::string>();
  }
  return page;
}

json conversation_create_to_json(const ConversationCreateParams& params) {
  json body = json::object();
  if (params.metadata) {
    body["metadata"] = *params.metadata;
  }
  return body;
}

json conversation_update_to_json(const ConversationUpdateParams& params) {
  json body = json::object();
  if (params.metadata) {
    body["metadata"] = *params.metadata;
  }
  return body;
}

std::string join_include(const std::optional<std::vector<std::string>>& include) {
  if (!include || include->empty()) return {};
  std::ostringstream oss;
  for (std::size_t i = 0; i < include->size(); ++i) {
    if (i > 0) oss << ",";
    oss << (*include)[i];
  }
  return oss.str();
}

}  // namespace

ConversationsResource::ConversationsResource(OpenAIClient& client)
    : client_(client), items_(std::make_unique<ConversationItemsResource>(client)) {}

Conversation ConversationsResource::create(const ConversationCreateParams& params) const {
  return create(params, RequestOptions{});
}

Conversation ConversationsResource::create(const ConversationCreateParams& params,
                                           const RequestOptions& options) const {
  auto body = conversation_create_to_json(params).dump();
  auto response = client_.perform_request("POST", "/conversations", body, options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation create response: ") + ex.what());
  }
}

Conversation ConversationsResource::create() const {
  return create(ConversationCreateParams{});
}

Conversation ConversationsResource::retrieve(const std::string& conversation_id) const {
  return retrieve(conversation_id, RequestOptions{});
}

Conversation ConversationsResource::retrieve(const std::string& conversation_id,
                                             const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation retrieve response: ") + ex.what());
  }
}

Conversation ConversationsResource::update(const std::string& conversation_id,
                                           const ConversationUpdateParams& params) const {
  return update(conversation_id, params, RequestOptions{});
}

Conversation ConversationsResource::update(const std::string& conversation_id,
                                           const ConversationUpdateParams& params,
                                           const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto body = conversation_update_to_json(params).dump();
  auto response = client_.perform_request("POST", path, body, options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation update response: ") + ex.what());
  }
}

ConversationDeleted ConversationsResource::remove(const std::string& conversation_id) const {
  return remove(conversation_id, RequestOptions{});
}

ConversationDeleted ConversationsResource::remove(const std::string& conversation_id,
                                                  const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_conversation_deleted(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation delete response: ") + ex.what());
  }
}

ConversationItemsResource& ConversationsResource::items() { return *items_; }

const ConversationItemsResource& ConversationsResource::items() const { return *items_; }

ConversationItemList ConversationItemsResource::create(const std::string& conversation_id,
                                                       const ItemCreateParams& params) const {
  return create(conversation_id, params, RequestOptions{});
}

ConversationItemList ConversationItemsResource::create(const std::string& conversation_id,
                                                       const ItemCreateParams& params,
                                                       const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.include) {
    request_options.query_params["include"] = join_include(params.include);
  }
  auto path = std::string("/conversations/") + conversation_id + "/items";
  auto response = client_.perform_request("POST", path, params.body.dump(), request_options);
  try {
    return parse_item_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation items create response: ") + ex.what());
  }
}

ConversationItem ConversationItemsResource::retrieve(const std::string& conversation_id,
                                                     const std::string& item_id,
                                                     const ItemRetrieveParams& params) const {
  return retrieve(conversation_id, item_id, params, RequestOptions{});
}

ConversationItem ConversationItemsResource::retrieve(const std::string& conversation_id,
                                                     const std::string& item_id,
                                                     const ItemRetrieveParams& params,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.include) {
    request_options.query_params["include"] = join_include(params.include);
  }
  auto path = std::string("/conversations/") + params.conversation_id + "/items/" + item_id;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_item(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation item retrieve response: ") + ex.what());
  }
}

ConversationItemPage ConversationItemsResource::list(const std::string& conversation_id,
                                                     const ItemListParams& params) const {
  return list(conversation_id, params, RequestOptions{});
}

ConversationItemPage ConversationItemsResource::list(const std::string& conversation_id,
                                                     const ItemListParams& params,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.include) request_options.query_params["include"] = join_include(params.include);

  auto path = std::string("/conversations/") + conversation_id + "/items";
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_item_page(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation items list response: ") + ex.what());
  }
}

ConversationItemPage ConversationItemsResource::list(const std::string& conversation_id) const {
  return list(conversation_id, ItemListParams{});
}

CursorPage<ConversationItem> ConversationItemsResource::list_page(const std::string& conversation_id,
                                                                  const ItemListParams& params) const {
  return list_page(conversation_id, params, RequestOptions{});
}

CursorPage<ConversationItem> ConversationItemsResource::list_page(const std::string& conversation_id,
                                                                  const ItemListParams& params,
                                                                  const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.include) request_options.query_params["include"] = join_include(params.include);

  auto fetch_impl = std::make_shared<std::function<CursorPage<ConversationItem>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<ConversationItem> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    ConversationItemPage page;
    try {
      page = parse_item_page(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse conversation items list response: ") + ex.what());
    }

    return CursorPage<ConversationItem>(std::move(page.data),
                                        page.has_more,
                                        page.next_cursor,
                                        request_options,
                                        *fetch_impl,
                                        "after",
                                        std::move(page.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = std::string("/conversations/") + conversation_id + "/items";
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

Conversation ConversationItemsResource::remove(const std::string& conversation_id, const std::string& item_id) const {
  return remove(conversation_id, item_id, ItemDeleteParams{conversation_id}, RequestOptions{});
}

Conversation ConversationItemsResource::remove(const std::string& conversation_id,
                                               const std::string& item_id,
                                               const ItemDeleteParams& params,
                                               const RequestOptions& options) const {
  auto path = std::string("/conversations/") + params.conversation_id + "/items/" + item_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation item delete response: ") + ex.what());
  }
}

}  // namespace openai

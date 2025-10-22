#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

struct Conversation {
  std::string id;
  int created_at = 0;
  nlohmann::json metadata = nlohmann::json::object();
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationCreateParams {
  std::optional<std::map<std::string, std::string>> metadata;
};

struct ConversationUpdateParams {
  std::optional<std::map<std::string, std::string>> metadata;
};

struct ConversationItem {
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationItemList {
  std::vector<ConversationItem> data;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationItemPage {
  std::vector<ConversationItem> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
};

struct ItemCreateParams {
  std::optional<std::vector<std::string>> include;
  nlohmann::json body = nlohmann::json::object();
};

struct ItemRetrieveParams {
  std::string conversation_id;
  std::optional<std::vector<std::string>> include;
};

struct ItemListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::vector<std::string>> include;
};

struct ItemDeleteParams {
  std::string conversation_id;
};

class ConversationItemsResource;

class ConversationsResource {
public:
  explicit ConversationsResource(OpenAIClient& client);

  Conversation create(const ConversationCreateParams& params) const;
  Conversation create(const ConversationCreateParams& params, const RequestOptions& options) const;
  Conversation create() const;

  Conversation retrieve(const std::string& conversation_id) const;
  Conversation retrieve(const std::string& conversation_id, const RequestOptions& options) const;

  Conversation update(const std::string& conversation_id, const ConversationUpdateParams& params) const;
  Conversation update(const std::string& conversation_id,
                      const ConversationUpdateParams& params,
                      const RequestOptions& options) const;

  ConversationDeleted remove(const std::string& conversation_id) const;
  ConversationDeleted remove(const std::string& conversation_id, const RequestOptions& options) const;

  ConversationItemsResource& items();
  const ConversationItemsResource& items() const;

private:
  OpenAIClient& client_;
  std::unique_ptr<ConversationItemsResource> items_;
};

class ConversationItemsResource {
public:
  explicit ConversationItemsResource(OpenAIClient& client) : client_(client) {}

  ConversationItemList create(const std::string& conversation_id, const ItemCreateParams& params) const;
  ConversationItemList create(const std::string& conversation_id,
                              const ItemCreateParams& params,
                              const RequestOptions& options) const;

  ConversationItem retrieve(const std::string& conversation_id,
                            const std::string& item_id,
                            const ItemRetrieveParams& params) const;
  ConversationItem retrieve(const std::string& conversation_id,
                            const std::string& item_id,
                            const ItemRetrieveParams& params,
                            const RequestOptions& options) const;

  ConversationItemPage list(const std::string& conversation_id, const ItemListParams& params) const;
  ConversationItemPage list(const std::string& conversation_id,
                            const ItemListParams& params,
                            const RequestOptions& options) const;
  ConversationItemPage list(const std::string& conversation_id) const;

  CursorPage<ConversationItem> list_page(const std::string& conversation_id, const ItemListParams& params) const;
  CursorPage<ConversationItem> list_page(const std::string& conversation_id,
                                         const ItemListParams& params,
                                         const RequestOptions& options) const;

  Conversation remove(const std::string& conversation_id, const std::string& item_id) const;
  Conversation remove(const std::string& conversation_id,
                      const std::string& item_id,
                      const ItemDeleteParams& params,
                      const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/client.hpp"
#include "openai/conversations.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(ConversationsResourceTest, CreateSerializeMetadata) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "conv_123",
    "created_at": 1700000000,
    "metadata": {"project":"alpha"},
    "object": "conversation"
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  ConversationCreateParams params;
  params.metadata = std::map<std::string, std::string>{{"project", "alpha"}};

  auto convo = client.conversations().create(params);
  EXPECT_EQ(convo.id, "conv_123");
  ASSERT_TRUE(convo.metadata.is_object());
  EXPECT_EQ(convo.metadata.at("project"), "alpha");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/conversations"), std::string::npos);
  auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("metadata").at("project"), "alpha");
}

TEST(ConversationsResourceTest, ItemsCreateIncludesQuery) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "data": [
      {"id":"msg_1","type":"message","role":"user","status":"completed","content":[]}
    ]
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  ItemCreateParams params;
  params.include = std::vector<std::string>{"response"};
  ResponseInputItem item;
  item.type = ResponseInputItem::Type::Message;
  item.message.role = "user";
  ResponseInputContent content;
  content.type = ResponseInputContent::Type::Text;
  content.text = "";
  item.message.content.push_back(content);
  params.items.push_back(item);

  auto items = client.conversations().items().create("conv_123", params);
  ASSERT_EQ(items.data.size(), 1u);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("include=response"), std::string::npos);
  auto payload = nlohmann::json::parse(request.body);
  ASSERT_TRUE(payload.contains("items"));
  ASSERT_FALSE(payload.at("items").empty());
  EXPECT_EQ(payload.at("items")[0].at("role"), "user");
  EXPECT_EQ(payload.at("items")[0].at("content")[0].at("type"), "input_text");
}

TEST(ConversationsResourceTest, ItemsListHandlesCursor) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "data": [
      {"id":"msg_1","type":"message","role":"user","status":"completed","content":[]}
    ],
    "has_more": true,
    "last_id": "msg_1"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  ItemListParams params;
  params.limit = 10;

  auto page = client.conversations().items().list("conv_123", params);
  EXPECT_TRUE(page.has_more);
  ASSERT_TRUE(page.next_cursor.has_value());
  EXPECT_EQ(*page.next_cursor, "msg_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=10"), std::string::npos);
}


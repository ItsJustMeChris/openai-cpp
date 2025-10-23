#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/beta.hpp"
#include "openai/chatkit.hpp"
#include "openai/client.hpp"
#include "support/mock_http_client.hpp"

namespace openai {
namespace {

using namespace openai::testing;

TEST(BetaChatKitSessionsTest, CreateSendsHeaderAndSerializesBody) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "cksess_123",
    "object": "chatkit.session",
    "expires_at": 1700000100,
    "client_secret": "secret",
    "status": "active",
    "user": "user-123",
    "workflow": {"id": "workflow_id"},
    "chatkit_configuration": {"file_upload": {"enabled": true}}
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  beta::ChatKitSessionCreateParams params;
  params.user = "user-123";
  params.workflow.id = "workflow_id";
  params.workflow.version = "1";
  beta::ChatKitSessionChatKitConfigurationParam configuration;
  beta::ChatKitSessionChatKitConfigurationParam::FileUpload upload;
  upload.enabled = true;
  configuration.file_upload = upload;
  params.chatkit_configuration = configuration;
  beta::ChatKitSessionRateLimitsParam rate_limits;
  rate_limits.max_requests_per_1_minute = 15;
  params.rate_limits = rate_limits;

  auto session = client.beta().chatkit().sessions().create(params);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/chatkit/sessions"), std::string::npos);
  EXPECT_EQ(request.headers.at("OpenAI-Beta"), "chatkit_beta=v1");

  auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("user"), "user-123");
  EXPECT_EQ(payload.at("workflow").at("id"), "workflow_id");
  EXPECT_EQ(payload.at("rate_limits").at("max_requests_per_1_minute"), 15);

  EXPECT_EQ(session.id, "cksess_123");
  EXPECT_EQ(session.user, "user-123");
  EXPECT_EQ(session.status, "active");
  ASSERT_TRUE(session.client_secret.has_value());
  EXPECT_EQ(*session.client_secret, "secret");
}

TEST(BetaChatKitSessionsTest, CancelUsesCorrectEndpoint) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();
  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"id":"cksess_123","object":"chatkit.session","status":"cancelled"})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  auto session = client.beta().chatkit().sessions().cancel("cksess_123");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/chatkit/sessions/cksess_123/cancel"), std::string::npos);
  EXPECT_EQ(request.headers.at("OpenAI-Beta"), "chatkit_beta=v1");
  EXPECT_EQ(session.status, "cancelled");
}

TEST(BetaChatKitThreadsTest, RetrieveAndListUseBetaHeader) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string retrieve_body = R"({
    "id": "cthr_123",
    "object": "chatkit.thread",
    "created_at": 1700000000,
    "status": "active",
    "user": "user-123"
  })";
  const std::string list_body = R"({
    "data": [{
      "id": "cthr_123",
      "object": "chatkit.thread",
      "created_at": 1700000000,
      "status": "active",
      "user": "user-123"
    }],
    "has_more": false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, retrieve_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, list_body});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  auto thread = client.beta().chatkit().threads().retrieve("cthr_123");
  EXPECT_EQ(thread.id, "cthr_123");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  {
    const auto& request = *mock_ptr->last_request();
    EXPECT_EQ(request.headers.at("OpenAI-Beta"), "chatkit_beta=v1");
    EXPECT_NE(request.url.find("/chatkit/threads/cthr_123"), std::string::npos);
  }

  beta::ChatKitThreadListParams params;
  params.limit = 5;
  auto list = client.beta().chatkit().threads().list(params);
  EXPECT_EQ(list.data.size(), 1u);
  EXPECT_FALSE(list.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  {
    const auto& request = *mock_ptr->last_request();
    EXPECT_EQ(request.headers.at("OpenAI-Beta"), "chatkit_beta=v1");
    EXPECT_NE(request.url.find("/chatkit/threads"), std::string::npos);
    EXPECT_NE(request.url.find("limit=5"), std::string::npos);
  }
}

TEST(BetaChatKitThreadsTest, ListItemsAppliesQueryParams) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();
  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({
    "data": [{
      "id": "item_1",
      "object": "chatkit.thread_item",
      "type": "chatkit.task"
    }],
    "has_more": true,
    "next_cursor": "cursor_2"
  })"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  beta::ChatKitThreadListItemsParams params;
  params.limit = 1;
  params.after = "cursor_1";

  auto items = client.beta().chatkit().threads().list_items("cthr_123", params);
  EXPECT_EQ(items.data.size(), 1u);
  EXPECT_TRUE(items.has_more);
  ASSERT_TRUE(items.next_cursor.has_value());
  EXPECT_EQ(*items.next_cursor, "cursor_2");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.headers.at("OpenAI-Beta"), "chatkit_beta=v1");
  EXPECT_NE(request.url.find("/chatkit/threads/cthr_123/items"), std::string::npos);
  EXPECT_NE(request.url.find("limit=1"), std::string::npos);
  EXPECT_NE(request.url.find("after=cursor_1"), std::string::npos);
}

}  // namespace
}  // namespace openai

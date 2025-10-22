#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/batches.hpp"
#include "openai/client.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(BatchesResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  const std::string response_body = R"({
    "id": "batch_123",
    "completion_window": "24h",
    "created_at": 1700000000,
    "endpoint": "/v1/responses",
    "input_file_id": "file_1",
    "metadata": {"team": "infra"},
    "object": "batch",
    "status": "in_progress"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  BatchCreateRequest request;
  request.completion_window = "24h";
  request.endpoint = "/v1/responses";
  request.input_file_id = "file_1";
  request.metadata = std::map<std::string, std::string>{{"team", "infra"}};
  BatchCreateRequest::OutputExpiresAfter expires_after;
  expires_after.anchor = "created_at";
  expires_after.seconds = 7200;
  request.output_expires_after = expires_after;

  auto batch = client.batches().create(request);
  EXPECT_EQ(batch.id, "batch_123");
  ASSERT_TRUE(batch.metadata.has_value());
  EXPECT_EQ(batch.metadata->at("team"), "infra");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.url.find("/batches"), std::string::npos);

  const auto payload = nlohmann::json::parse(http_request.body);
  EXPECT_EQ(payload.at("completion_window"), "24h");
  EXPECT_EQ(payload.at("endpoint"), "/v1/responses");
  EXPECT_EQ(payload.at("input_file_id"), "file_1");
  EXPECT_EQ(payload.at("metadata").at("team"), "infra");
  EXPECT_EQ(payload.at("output_expires_after").at("anchor"), "created_at");
  EXPECT_EQ(payload.at("output_expires_after").at("seconds"), 7200);
}

TEST(BatchesResourceTest, ListWithParamsParsesResponse) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  const std::string response_body = R"({
    "data": [
      {
        "id": "batch_1",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/responses",
        "input_file_id": "file_1",
        "object": "batch",
        "status": "completed",
        "request_counts": {"completed": 10, "failed": 1, "total": 11},
        "usage": {
          "input_tokens": 100,
          "input_tokens_details": {"cached_tokens": 20},
          "output_tokens": 50,
          "output_tokens_details": {"reasoning_tokens": 5},
          "total_tokens": 150
        }
      }
    ],
    "has_more": true
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  BatchListParams params;
  params.limit = 2;
  params.after = std::string("batch_0");

  auto list = client.batches().list(params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_TRUE(list.has_more);
  ASSERT_TRUE(list.next_cursor.has_value());
  EXPECT_EQ(*list.next_cursor, "batch_1");

  const auto& batch = list.data.front();
  EXPECT_EQ(batch.status, "completed");
  ASSERT_TRUE(batch.request_counts.has_value());
  EXPECT_EQ(batch.request_counts->completed, 10);
  ASSERT_TRUE(batch.usage.has_value());
  EXPECT_EQ(batch.usage->input_tokens_details.cached_tokens, 20);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "GET");
  EXPECT_NE(http_request.url.find("/batches"), std::string::npos);
  EXPECT_NE(http_request.url.find("limit=2"), std::string::npos);
  EXPECT_NE(http_request.url.find("after=batch_0"), std::string::npos);
}

TEST(BatchesResourceTest, CancelPostsToEndpoint) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  const std::string response_body = R"({
    "id": "batch_123",
    "completion_window": "24h",
    "created_at": 1700000000,
    "endpoint": "/v1/responses",
    "input_file_id": "file_1",
    "object": "batch",
    "status": "cancelling"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto batch = client.batches().cancel("batch_123");
  EXPECT_EQ(batch.status, "cancelling");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.url.find("/batches/batch_123/cancel"), std::string::npos);

  nlohmann::json payload = nlohmann::json::parse(http_request.body);
  EXPECT_TRUE(payload.is_object());
  EXPECT_TRUE(payload.empty());
}

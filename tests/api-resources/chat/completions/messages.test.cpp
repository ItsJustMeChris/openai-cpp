#include <gtest/gtest.h>

#include "openai/chat.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "support/mock_http_client.hpp"

#include <memory>
#include <string>

namespace oait = openai::testing;

TEST(ChatCompletionsMessagesResourceTest, ListParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"json({
    "data": [
      {
        "id": "msg_123",
        "role": "assistant",
        "content": [
          {"type": "text", "text": "Hello"}
        ]
      }
    ],
    "has_more": false
  })json";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";
  options.base_url = "http://127.0.0.1:4010";

  OpenAIClient client(options, std::move(mock_client));

  auto messages = client.chat().completions().messages().list("completion_id");
  ASSERT_EQ(messages.data.size(), 1u);
  EXPECT_EQ(messages.data.front().id, "msg_123");
  ASSERT_FALSE(messages.data.front().content_parts.empty());
  EXPECT_EQ(messages.data.front().content_parts.front().text, "Hello");
  EXPECT_FALSE(messages.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "GET");
  EXPECT_NE(request.url.find("/chat/completions/completion_id/messages"), std::string::npos);
}

TEST(ChatCompletionsMessagesResourceTest, ListAppliesParamsAndRequestOptions) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string error_body = R"json({"error": {"message": "Not Found"}})json";
  mock_ptr->enqueue_response(openai::HttpResponse{404, {}, error_body});

  ClientOptions options;
  options.api_key = "sk-test";
  options.base_url = "http://127.0.0.1:4010";

  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionMessageListParams params;
  params.after = std::string("after");
  params.limit = 0;
  params.order = std::string("asc");

  RequestOptions request_options;
  request_options.headers["X-Test-Header"] = "custom";
  request_options.query_params["foo"] = "bar";

  EXPECT_THROW(
      client.chat().completions().messages().list("completion_id", params, request_options),
      NotFoundError);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  auto header_it = request.headers.find("X-Test-Header");
  ASSERT_NE(header_it, request.headers.end());
  EXPECT_EQ(header_it->second, "custom");

  auto query_pos = request.url.find('?');
  ASSERT_NE(query_pos, std::string::npos);
  const auto query = request.url.substr(query_pos + 1);
  EXPECT_NE(query.find("after=after"), std::string::npos);
  EXPECT_NE(query.find("limit=0"), std::string::npos);
  EXPECT_NE(query.find("order=asc"), std::string::npos);
  EXPECT_NE(query.find("foo=bar"), std::string::npos);
}

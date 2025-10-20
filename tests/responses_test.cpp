#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/responses.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace oait = openai::testing;

TEST(ResponsesResourceTest, ParsesOutputTextAndUsage) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"json(
    {
      "id": "resp_123",
      "object": "response",
      "created": 1700000000,
      "model": "gpt-4o-mini",
      "output": [
        {
          "type": "message",
          "role": "assistant",
          "content": [
            { "type": "output_text", "text": "Hello" },
            { "type": "output_text", "text": ", world!" }
          ]
        }
      ],
      "usage": {
        "input_tokens": 5,
        "output_tokens": 7,
        "total_tokens": 12
      }
    }
  )json";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRequest request;
  request.body = json{{"model", "gpt-4o-mini"}, {"input", json::array()}};

  auto response = client.responses().create(request);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "POST");
  EXPECT_NE(last_request.body.find("gpt-4o-mini"), std::string::npos);

  EXPECT_EQ(response.id, "resp_123");
  EXPECT_EQ(response.model, "gpt-4o-mini");
  ASSERT_EQ(response.messages.size(), 1u);
  EXPECT_EQ(response.messages[0].text_segments.size(), 2u);
  EXPECT_EQ(response.output_text, "Hello, world!");
  ASSERT_TRUE(response.usage.has_value());
  EXPECT_EQ(response.usage->input_tokens, 5);
  EXPECT_EQ(response.usage->output_tokens, 7);
  EXPECT_EQ(response.usage->total_tokens, 12);
}

TEST(ResponsesResourceTest, RetrieveAddsStreamQueryParam) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, R"({"id":"resp_1","object":"response","created":1,"model":"gpt-4o","output":[]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRetrieveOptions retrieve_options;
  retrieve_options.stream = true;

  auto response = client.responses().retrieve("resp_1", retrieve_options, RequestOptions{});
  EXPECT_EQ(response.id, "resp_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "GET");
  EXPECT_NE(request.url.find("stream=true"), std::string::npos);
}

TEST(ResponsesResourceTest, CancelParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, R"({"id":"resp_cancel","object":"response","created":2,"model":"gpt-4o","output":[]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto response = client.responses().cancel("resp_cancel");
  EXPECT_EQ(response.id, "resp_cancel");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/responses/resp_cancel/cancel"), std::string::npos);
}

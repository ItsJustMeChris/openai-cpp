#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/beta.hpp"
#include "openai/client.hpp"
#include "openai/thread_types.hpp"
#include "support/mock_http_client.hpp"

namespace openai {
namespace {

using namespace openai::testing;

TEST(BetaThreadsResourceTest, CreateDelegatesToThreadsResource) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "thread_123",
    "object": "thread",
    "created_at": 1700000000,
    "metadata": {}
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  ThreadCreateRequest request;
  request.metadata["purpose"] = "test";

  auto thread = client.beta().threads().create(request);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& recorded = *mock_ptr->last_request();
  EXPECT_EQ(recorded.method, "POST");
  EXPECT_NE(recorded.url.find("/threads"), std::string::npos);
  EXPECT_EQ(recorded.headers.at("OpenAI-Beta"), "assistants=v2");

  auto payload = nlohmann::json::parse(recorded.body);
  EXPECT_EQ(payload.at("metadata").at("purpose"), "test");
  EXPECT_EQ(thread.id, "thread_123");

  EXPECT_EQ(&client.beta().threads().messages(), &client.thread_messages());
  EXPECT_EQ(&client.beta().threads().runs(), &client.runs());
  EXPECT_EQ(&client.beta().threads().run_steps(), &client.run_steps());
}

}  // namespace
}  // namespace openai

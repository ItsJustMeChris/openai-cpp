#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/beta.hpp"
#include "openai/client.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(BetaRealtimeSessionsTest, CreateSendsBetaHeaderAndBody) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "sess_123",
    "client_secret": "secret",
    "model": "gpt-4o-mini-realtime-preview",
    "modalities": ["text", "audio"],
    "tools": [{"type": "function", "definition": {"name": "foo"}}]
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  beta::RealtimeSessionCreateParams params;
  params.model = "gpt-4o-mini-realtime-preview";
  params.modalities = std::vector<std::string>{"text", "audio"};
  beta::RealtimeSessionTool tool;
  tool.type = "function";
  tool.definition = nlohmann::json{{"name", "foo"}};
  params.tools.push_back(tool);

  auto session = client.beta().realtime().sessions().create(params);
  EXPECT_EQ(session.id, "sess_123");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.headers.at("OpenAI-Beta"), "assistants=v2");
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/realtime/sessions"), std::string::npos);
  auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("model"), "gpt-4o-mini-realtime-preview");
  EXPECT_EQ(payload.at("modalities").size(), 2u);
  EXPECT_EQ(payload.at("tools")[0].at("definition").at("name"), "foo");
}


#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/chat.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(ChatCompletionsResourceTest, CreateStreamParsesEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = "event: message\n"
                           "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionRequest request;
  request.model = "gpt-4o";
  ChatMessage message;
  message.role = "user";
  ChatMessageContent content;
  content.type = ChatMessageContent::Type::Text;
  content.text = "Hi";
  message.content.push_back(content);
  request.messages.push_back(message);
  request.temperature = 0.2;

  auto events = client.chat().completions().create_stream(request);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_TRUE(events[0].event.has_value());
  EXPECT_EQ(*events[0].event, "message");
  EXPECT_NE(events[0].data.find("Hello"), std::string::npos);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
}

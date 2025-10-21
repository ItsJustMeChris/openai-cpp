#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/messages.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(ThreadMessagesResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "msg_123",
    "object": "thread.message",
    "thread_id": "thread_1",
    "role": "user",
    "status": "completed",
    "created_at": 1,
    "content": []
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  MessageCreateRequest request;
  request.role = "user";
  request.content = std::string("Hello");
  MessageAttachment attachment;
  attachment.file_id = "file_1";
  attachment.tools.push_back(ThreadMessageAttachmentTool{ThreadMessageAttachmentTool::Type::CodeInterpreter});
  request.attachments.push_back(attachment);
  request.metadata["project"] = "demo";

  auto message = client.thread_messages().create("thread_1", request);
  EXPECT_EQ(message.id, "msg_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.headers.at("OpenAI-Beta"), "assistants=v2");
  const auto payload = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(payload.at("role"), "user");
  EXPECT_EQ(payload.at("content"), "Hello");
  EXPECT_EQ(payload.at("attachments")[0].at("file_id"), "file_1");
  EXPECT_EQ(payload.at("metadata").at("project"), "demo");
}

TEST(ThreadMessagesResourceTest, ListWithParams) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string list_body = R"({
    "data": [
      {"id": "msg_1", "object": "thread.message", "thread_id": "thread_1", "role": "assistant", "status": "completed", "created_at": 1, "content": []}
    ],
    "has_more": false,
    "first_id": "msg_1",
    "last_id": "msg_1"
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, list_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  MessageListParams params;
  params.limit = 10;
  params.order = "desc";
  params.run_id = "run_123";

  auto list = client.thread_messages().list("thread_1", params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].id, "msg_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=10"), std::string::npos);
  EXPECT_NE(request.url.find("order=desc"), std::string::npos);
  EXPECT_NE(request.url.find("run_id=run_123"), std::string::npos);
}


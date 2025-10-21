#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/threads.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(ThreadsResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "thread_123",
    "created_at": 1700000000,
    "metadata": {"project": "demo"},
    "object": "thread",
    "tool_resources": {"code_interpreter": {"file_ids": ["file_1"]}}
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadCreateRequest request;
  ThreadMessageCreate message;
  message.role = "user";
  message.content = std::string("Hello");
  ThreadMessageAttachment attachment;
  attachment.file_id = "file_attach";
  attachment.tools.push_back(ThreadMessageAttachmentTool{ThreadMessageAttachmentTool::Type::FileSearch});
  message.attachments.push_back(attachment);
  message.metadata["topic"] = "greeting";
  request.messages.push_back(message);
  request.metadata["project"] = "demo";
  ThreadToolResources resources;
  resources.code_interpreter_file_ids.push_back("file_1");
  resources.file_search_vector_store_ids.push_back("vs_1");
  request.tool_resources = resources;

  auto thread = client.threads().create(request);
  EXPECT_EQ(thread.id, "thread_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.headers.at("OpenAI-Beta"), "assistants=v2");
  const auto payload = nlohmann::json::parse(last_request.body);
  ASSERT_TRUE(payload.contains("messages"));
  ASSERT_EQ(payload.at("messages").size(), 1u);
  EXPECT_EQ(payload.at("messages")[0].at("role"), "user");
  EXPECT_EQ(payload.at("messages")[0].at("content"), "Hello");
  ASSERT_TRUE(payload.at("messages")[0].contains("attachments"));
  EXPECT_EQ(payload.at("messages")[0].at("attachments")[0].at("file_id"), "file_attach");
  EXPECT_EQ(payload.at("metadata").at("project"), "demo");
  ASSERT_TRUE(payload.contains("tool_resources"));
  EXPECT_EQ(payload.at("tool_resources").at("code_interpreter").at("file_ids")[0], "file_1");
  EXPECT_EQ(payload.at("tool_resources").at("file_search").at("vector_store_ids")[0], "vs_1");
}

TEST(ThreadsResourceTest, UpdateAndDelete) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string update_body = R"({
    "id": "thread_123",
    "created_at": 1700000000,
    "metadata": {"scope": "updated"},
    "object": "thread"
  })";

  const std::string delete_body = R"({"id":"thread_123","deleted":true,"object":"thread.deleted"})";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, update_body});
  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, delete_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadUpdateRequest request;
  request.metadata = std::map<std::string, std::string>{{"scope", "updated"}};

  auto thread = client.threads().update("thread_123", request);
  EXPECT_EQ(thread.metadata.at("scope"), "updated");

  auto deleted = client.threads().remove("thread_123");
  EXPECT_TRUE(deleted.deleted);
  EXPECT_EQ(deleted.object, "thread.deleted");
}


#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/threads.hpp"
#include "openai/assistant_stream.hpp"
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
  ThreadToolResources::CodeInterpreter code_interpreter;
  code_interpreter.file_ids.push_back("file_1");
  resources.code_interpreter = std::move(code_interpreter);
  ThreadToolResources::FileSearch file_search;
  file_search.vector_store_ids.push_back("vs_1");
  resources.file_search = std::move(file_search);
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

TEST(ThreadsResourceTest, CreateAndRunCombinesThreadAndRunFields) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string run_body = R"({
    "id": "run_1",
    "assistant_id": "asst",
    "created_at": 1,
    "model": "gpt-4o",
    "object": "thread.run",
    "parallel_tool_calls": false,
    "status": "queued",
    "thread_id": "thread_1",
    "tools": []
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, run_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadCreateAndRunRequest request;
  ThreadCreateRequest thread_request;
  thread_request.metadata["project"] = "demo";
  request.thread = thread_request;
  request.run.assistant_id = "asst";
  request.run.instructions = "Do it";
  request.run.include = std::vector<std::string>{"step_details.tool_calls[*].file_search.results[*].content"};

  auto run = client.threads().create_and_run(request);
  EXPECT_EQ(run.id, "run_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.url.find("/threads/runs"), std::string::npos);
  EXPECT_NE(http_request.url.find("include="), std::string::npos);

  const auto payload = nlohmann::json::parse(http_request.body);
  EXPECT_EQ(payload.at("thread").at("metadata").at("project"), "demo");
  EXPECT_EQ(payload.at("assistant_id"), "asst");
  EXPECT_EQ(payload.at("instructions"), "Do it");
}

TEST(ThreadsResourceTest, CreateAndRunStreamCollectsEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string sse =
      "event: thread.created\n"
      "data: {\"id\":\"thread_1\",\"object\":\"thread\",\"created_at\":1}\n\n"
      "event: thread.run.created\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"in_progress\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n"
      "event: thread.run.step.delta\n"
      "data: {\"id\":\"step_delta\",\"object\":\"thread.run.step.delta\",\"delta\":{\"step_details\":{\"type\":\"tool_calls\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]}}}\n\n"
      "event: thread.message.delta\n"
      "data: {\"id\":\"msg_1\",\"object\":\"thread.message.delta\",\"delta\":{\"content\":[{\"type\":\"text\",\"index\":0,\"text\":{\"value\":\"partial\"}}]}}\n\n"
      "event: thread.run.completed\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"completed\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n"
      "event: thread.message.completed\n"
      "data: {\"id\":\"msg_1\",\"object\":\"thread.message\",\"created_at\":1,\"thread_id\":\"thread_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[],\"attachments\":[]}\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, sse});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadCreateAndRunRequest request;
  request.run.assistant_id = "asst";
  request.run.include = std::vector<std::string>{"step_details.tool_calls[*].file_search.results[*].content"};

  auto events = client.threads().create_and_run_stream(request);
  ASSERT_EQ(events.size(), 6u);
  EXPECT_TRUE(std::holds_alternative<AssistantThreadEvent>(events[0]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunEvent>(events[1]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunStepDeltaEvent>(events[2]));
  EXPECT_TRUE(std::holds_alternative<AssistantMessageDeltaEvent>(events[3]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunEvent>(events[4]));
  EXPECT_TRUE(std::holds_alternative<AssistantMessageEvent>(events[5]));

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.url.find("include="), std::string::npos);
  EXPECT_EQ(http_request.headers.at("X-Stainless-Helper-Method"), "stream");
}

TEST(ThreadsResourceTest, CreateAndRunStreamSnapshotProvidesFinalRun) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string sse =
      "event: thread.run.created\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"in_progress\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n"
      "event: thread.run.completed\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"completed\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, sse});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadCreateAndRunRequest request;
  request.run.assistant_id = "asst";

  auto snapshot = client.threads().create_and_run_stream_snapshot(request);
  ASSERT_TRUE(snapshot.final_run().has_value());
  EXPECT_EQ(snapshot.final_run()->status, "completed");
}

TEST(ThreadsResourceTest, CreateAndRunPollUsesRunsResource) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string created =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"in_progress","thread_id":"thread_1","tools":[]})";
  const std::string completed =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"completed","thread_id":"thread_1","tools":[]})";

  mock_ptr->enqueue_response(HttpResponse{200, {}, created});
  mock_ptr->enqueue_response(HttpResponse{200, {}, completed});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ThreadCreateAndRunRequest request;
  request.run.assistant_id = "asst";

  openai::Run result = client.threads().create_and_run_poll(request, RequestOptions{}, std::chrono::milliseconds(0));
  EXPECT_EQ(result.status, "completed");
}

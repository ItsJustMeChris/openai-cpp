#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/runs.hpp"
#include "openai/assistant_stream.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>
#include <chrono>

namespace oait = openai::testing;

TEST(RunsResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "run_123",
    "assistant_id": "asst_1",
    "created_at": 1700000000,
    "instructions": "instr",
    "model": "gpt-4o",
    "object": "thread.run",
    "parallel_tool_calls": false,
    "status": "queued",
    "thread_id": "thread_1",
    "tools": []
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunCreateRequest request;
  request.assistant_id = "asst_1";
  request.include = std::vector<std::string>{"step_details.tool_calls[*].file_search.results[*].content"};
  request.additional_instructions = "be helpful";
  RunAdditionalMessage additional;
  additional.role = "user";
  additional.content = std::string("Hi");
  request.additional_messages.push_back(additional);
  request.instructions = "instr";
  request.max_completion_tokens = 200;
  request.metadata["project"] = "demo";
  request.parallel_tool_calls = true;
  request.reasoning_effort = "medium";
  AssistantResponseFormat format;
  format.type = "json_object";
  request.response_format = format;
  request.stream = false;
  request.temperature = 0.3;
  request.top_p = 0.9;
  AssistantToolChoice choice;
  choice.type = "auto";
  request.tool_choice = choice;
  AssistantTool tool;
  tool.type = AssistantTool::Type::Function;
  AssistantTool::FunctionDefinition fn;
  fn.name = "lookup";
  fn.parameters = nlohmann::json::object({{"type", "object"}});
  tool.function = fn;
  request.tools.push_back(tool);
  RunTruncationStrategy truncation;
  truncation.type = RunTruncationStrategy::Type::LastMessages;
  truncation.last_messages = 5;
  request.truncation_strategy = truncation;

  auto run = client.runs().create("thread_1", request);
  EXPECT_EQ(run.id, "run_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.headers.at("OpenAI-Beta"), "assistants=v2");
  EXPECT_NE(last_request.url.find("include="), std::string::npos);
  const auto payload = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(payload.at("assistant_id"), "asst_1");
  EXPECT_EQ(payload.at("additional_instructions"), "be helpful");
  EXPECT_EQ(payload.at("additional_messages")[0].at("role"), "user");
  EXPECT_EQ(payload.at("max_completion_tokens"), 200);
  EXPECT_TRUE(payload.at("parallel_tool_calls").get<bool>());
  EXPECT_EQ(payload.at("response_format").at("type"), "json_object");
  EXPECT_EQ(payload.at("tool_choice").at("type"), "auto");
  EXPECT_EQ(payload.at("tools")[0].at("function").at("name"), "lookup");
  EXPECT_EQ(payload.at("truncation_strategy").at("type"), "last_messages");
}

TEST(RunsResourceTest, ListAndSubmitToolOutputs) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string list_body = R"({
    "data": [
      {"id": "run_1", "assistant_id": "asst_1", "created_at": 1, "instructions": "instr", "model": "gpt-4o", "object": "thread.run", "parallel_tool_calls": false, "status": "queued", "thread_id": "thread_1", "tools": []}
    ],
    "has_more": false,
    "first_id": "run_1",
    "last_id": "run_1"
  })";

  const std::string submit_body = R"({
    "id": "run_1",
    "assistant_id": "asst_1",
    "created_at": 1,
    "instructions": "instr",
    "model": "gpt-4o",
    "object": "thread.run",
    "parallel_tool_calls": false,
    "status": "in_progress",
    "thread_id": "thread_1",
    "tools": []
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, list_body});
  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, submit_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunListParams params;
  params.limit = 20;
  params.order = "desc";
  params.status = "in_progress";
  auto list = client.runs().list("thread_1", params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].id, "run_1");

  RunSubmitToolOutputsRequest submit;
  submit.thread_id = "thread_1";
  submit.outputs.push_back(RunSubmitToolOutput{.tool_call_id = "call_1", .output = "result"});
  auto run = client.runs().submit_tool_outputs("thread_1", "run_1", submit);
  EXPECT_EQ(run.status, "in_progress");
}

TEST(RunsResourceTest, CreateStreamCollectsEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string sse =
      "event: thread.created\n"
      "data: {\"id\":\"thread_1\",\"object\":\"thread\",\"created_at\":1}\n\n"
      "event: thread.run.created\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst_1\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"in_progress\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n"
      "event: thread.run.step.delta\n"
      "data: {\"id\":\"step_1\",\"object\":\"thread.run.step.delta\",\"delta\":{\"step_details\":{\"type\":\"tool_calls\",\"tool_calls\":[{\"type\":\"function\",\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}}\n\n"
      "event: thread.message.delta\n"
      "data: {\"id\":\"msg_1\",\"object\":\"thread.message.delta\",\"delta\":{\"content\":[{\"type\":\"text\",\"index\":0,\"text\":{\"value\":\"partial\"}}]}}\n\n"
      "event: thread.run.completed\n"
      "data: {\"id\":\"run_1\",\"assistant_id\":\"asst_1\",\"created_at\":1,\"model\":\"gpt-4o\",\"object\":\"thread.run\",\"parallel_tool_calls\":false,\"status\":\"completed\",\"thread_id\":\"thread_1\",\"tools\":[]}\n\n"
      "event: thread.message.completed\n"
      "data: {\"id\":\"msg_1\",\"object\":\"thread.message\",\"created_at\":1,\"thread_id\":\"thread_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[],\"attachments\":[]}\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, sse});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunCreateRequest request;
  request.assistant_id = "asst_1";

  auto events = client.runs().create_stream("thread_1", request);
  ASSERT_EQ(events.size(), 6u);
  EXPECT_TRUE(std::holds_alternative<AssistantThreadEvent>(events[0]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunEvent>(events[1]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunStepDeltaEvent>(events[2]));
  EXPECT_TRUE(std::holds_alternative<AssistantMessageDeltaEvent>(events[3]));
  EXPECT_TRUE(std::holds_alternative<AssistantRunEvent>(events[4]));
  EXPECT_TRUE(std::holds_alternative<AssistantMessageEvent>(events[5]));
}

TEST(RunsResourceTest, PollAdvancesUntilTerminalState) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string queued =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"in_progress","thread_id":"thread_1","tools":[]})";
  const std::string completed =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"completed","thread_id":"thread_1","tools":[]})";

  mock_ptr->enqueue_response(HttpResponse{200, {{"openai-poll-after-ms", "1"}}, queued});
  mock_ptr->enqueue_response(HttpResponse{200, {}, completed});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunRetrieveParams params;
  params.thread_id = "thread_1";

  RequestOptions request_options;
  auto run = client.runs().poll("run_1", params, request_options, std::chrono::milliseconds(0));
  EXPECT_EQ(run.status, "completed");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_NE(mock_ptr->last_request()->url.find("run_1"), std::string::npos);
}

TEST(RunsResourceTest, CreateAndRunPollUsesHelpers) {
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

  RunCreateRequest request;
  request.assistant_id = "asst";

  RequestOptions request_options;
  auto run = client.runs().create_and_run_poll("thread_1", request, request_options, std::chrono::milliseconds(0));
  EXPECT_EQ(run.status, "completed");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_NE(mock_ptr->last_request()->url.find("run_1"), std::string::npos);
}

TEST(RunsResourceTest, SubmitToolOutputsAndPoll) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string in_progress =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"in_progress","thread_id":"thread_1","tools":[]})";
  const std::string completed =
      R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"completed","thread_id":"thread_1","tools":[]})";

  mock_ptr->enqueue_response(HttpResponse{200, {}, in_progress});
  mock_ptr->enqueue_response(HttpResponse{200, {}, completed});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunSubmitToolOutputsRequest request;
  request.thread_id = "thread_1";
  request.outputs.push_back(RunSubmitToolOutput{.tool_call_id = "call", .output = "result"});

  RequestOptions request_options;
  auto run = client.runs().submit_tool_outputs_and_poll(
      "thread_1", "run_1", request, request_options, std::chrono::milliseconds(0));
  EXPECT_EQ(run.status, "completed");
}

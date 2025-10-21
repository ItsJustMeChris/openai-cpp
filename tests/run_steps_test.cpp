#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/run_steps.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(RunStepsResourceTest, RetrieveParsesDetails) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "step_1",
    "assistant_id": "asst_1",
    "created_at": 1700000000,
    "run_id": "run_1",
    "thread_id": "thread_1",
    "status": "completed",
    "object": "thread.run.step",
    "step_details": {
      "type": "tool_calls",
      "tool_calls": [
        {
          "type": "function",
          "id": "call_1",
          "function": {"name": "lookup", "arguments": "{}", "output": "done"}
        }
      ]
    },
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunStepRetrieveParams params;
  params.thread_id = "thread_1";
  params.run_id = "run_1";
  params.include = std::vector<std::string>{"step_details.tool_calls[*].file_search.results[*].content"};

  auto step = client.run_steps().retrieve("run_1", "step_1", params);
  EXPECT_EQ(step.id, "step_1");
  ASSERT_FALSE(step.details.tool_calls.empty());
  EXPECT_EQ(step.details.tool_calls[0].type, ToolCallDetails::Type::Function);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("include="), std::string::npos);
}

TEST(RunStepsResourceTest, ListParsesSteps) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string list_body = R"({
    "data": [
      {
        "id": "step_1",
        "assistant_id": "asst_1",
        "created_at": 1700000000,
        "run_id": "run_1",
        "thread_id": "thread_1",
        "status": "completed",
        "object": "thread.run.step",
        "step_details": {
          "type": "message_creation",
          "message_creation": {"message_id": "msg_1"}
        }
      }
    ],
    "has_more": false
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, list_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  RunStepListParams params;
  params.thread_id = "thread_1";
  params.limit = 5;
  params.order = "desc";

  auto list = client.run_steps().list("run_1", params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].details.type, RunStepDetails::Type::MessageCreation);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=5"), std::string::npos);
  EXPECT_NE(request.url.find("order=desc"), std::string::npos);
}

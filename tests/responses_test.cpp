#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/pagination.hpp"
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
          "id": "msg_123",
          "type": "message",
          "role": "assistant",
          "status": "completed",
          "content": [
            {
              "type": "output_text",
              "text": "Hello",
              "annotations": [
                {
                  "type": "url_citation",
                  "start_index": 0,
                  "end_index": 5,
                  "title": "Example",
                  "url": "https://example.com"
                }
              ]
            },
            { "type": "refusal", "refusal": "Sorry" },
            {
              "type": "output_text",
              "text": ", world!",
              "logprobs": [
                {
                  "token": "world",
                  "bytes": [119, 111, 114, 108, 100],
                  "logprob": -0.02,
                  "top_logprobs": [
                    { "token": "world", "bytes": [119, 111, 114, 108, 100], "logprob": -0.02 }
                  ]
                }
              ]
            }
          ]
        },
        {
          "type": "custom_tool_call",
          "id": "tool_item",
          "schema": {"name": "custom"}
        },
        {
          "type": "function_call",
          "id": "func_call_1",
          "call_id": "func",
          "name": "weather",
          "arguments": "{\"location\":\"SF\"}"
        },
        {
          "type": "function_call_output",
          "id": "func_output_1",
          "call_id": "func",
          "output": "{\"temp\":70}",
          "status": "completed"
        },
        {
          "type": "computer_call",
          "id": "comp_1",
          "call_id": "cmp_call",
          "status": "in_progress",
          "action": {
            "type": "scroll",
            "scroll_x": 0,
            "scroll_y": 200,
            "x": 42,
            "y": 84
          },
          "pending_safety_checks": [
            { "id": "psc_1", "code": "human_review", "message": "Requires confirmation" }
          ]
        },
        {
          "type": "computer_call_output",
          "id": "comp_out_1",
          "call_id": "cmp_call",
          "output": {
            "type": "computer_screenshot",
            "image_url": "https://example.com/screenshot.png"
          },
          "acknowledged_safety_checks": [
            { "id": "psc_1", "code": "human_review", "message": "Requires confirmation" }
          ],
          "status": "completed"
        },
        {
          "type": "web_search_call",
          "id": "ws_1",
          "status": "completed",
          "action": {
            "type": "search",
            "query": "weather tomorrow",
            "sources": [ { "type": "url", "url": "https://example.com/weather" } ]
          }
        },
        {
          "type": "local_shell_call",
          "id": "shell_1",
          "call_id": "shell_call",
          "status": "completed",
          "action": {
            "type": "exec",
            "command": ["ls", "-la"],
            "env": { "PATH": "/bin" },
            "timeout_ms": 1000,
            "user": "root",
            "working_directory": "/tmp"
          }
        },
        {
          "type": "local_shell_call_output",
          "id": "shell_out_1",
          "output": "{\"stdout\":\"ok\"}",
          "status": "completed"
        },
        {
          "type": "mcp_list_tools",
          "id": "mcp_list",
          "server_label": "deepwiki",
          "tools": [
            {
              "name": "lookup",
              "input_schema": {"type": "object"},
              "annotations": {"tags": ["docs", "search"]}
            }
          ],
          "next_page_token": "token-2"
        },
        {
          "type": "mcp_call",
          "id": "mcp_call_1",
          "name": "lookup",
          "server_label": "deepwiki",
          "arguments": "{}",
          "status": "completed",
          "output": "result"
        },
        {
          "type": "mcp_approval_request",
          "id": "approval_1",
          "arguments": "{}",
          "server_label": "deepwiki",
          "name": "lookup",
          "suggested_decision": "approved"
        },
        {
          "type": "mcp_approval_response",
          "id": "approval_1",
          "decision": "approved",
          "reason": "ok"
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
  request.model = "gpt-4o-mini";
  ResponseInputItem input;
  input.type = ResponseInputItem::Type::Message;
  input.message.role = "user";
  ResponseInputContent content;
  content.type = ResponseInputContent::Type::Text;
  content.text = "Say hello";
  input.message.content.push_back(std::move(content));
  request.input.push_back(std::move(input));

  auto response = client.responses().create(request);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "POST");
  EXPECT_NE(last_request.body.find("gpt-4o-mini"), std::string::npos);

  EXPECT_EQ(response.id, "resp_123");
  EXPECT_EQ(response.model, "gpt-4o-mini");
  ASSERT_EQ(response.output.size(), 13u);
  EXPECT_EQ(response.output[0].item_type, "message");
  ASSERT_TRUE(response.output[0].message.has_value());
  const auto& message = *response.output[0].message;
  EXPECT_EQ(message.id, "msg_123");
  ASSERT_TRUE(message.status.has_value());
  EXPECT_EQ(*message.status, "completed");
  ASSERT_EQ(message.content.size(), 3u);
  EXPECT_EQ(message.content[0].type, ResponseOutputContent::Type::Text);
  EXPECT_EQ(message.content[1].type, ResponseOutputContent::Type::Refusal);
  EXPECT_EQ(message.content[2].type, ResponseOutputContent::Type::Text);
  ASSERT_EQ(message.text_segments.size(), 2u);
  const auto& first_segment_annotations = message.text_segments[0].annotations;
  ASSERT_EQ(first_segment_annotations.size(), 1u);
  EXPECT_EQ(first_segment_annotations[0].type, ResponseOutputTextAnnotation::Type::UrlCitation);
  ASSERT_TRUE(first_segment_annotations[0].url.has_value());
  EXPECT_EQ(*first_segment_annotations[0].url, "https://example.com");
  const auto& second_segment_logprobs = message.text_segments[1].logprobs;
  ASSERT_EQ(second_segment_logprobs.size(), 1u);
  EXPECT_EQ(second_segment_logprobs[0].token, "world");
  EXPECT_DOUBLE_EQ(second_segment_logprobs[0].logprob, -0.02);
  ASSERT_EQ(second_segment_logprobs[0].top_logprobs.size(), 1u);
  EXPECT_EQ(second_segment_logprobs[0].top_logprobs[0].token, "world");
  EXPECT_EQ(response.output[1].item_type, "custom_tool_call");
  EXPECT_EQ(response.output[1].type, ResponseOutputItem::Type::CustomToolCall);
  ASSERT_TRUE(response.output[2].function_call.has_value());
  const auto& function_call = *response.output[2].function_call;
  EXPECT_EQ(function_call.id, "func_call_1");
  ASSERT_TRUE(function_call.parsed_arguments.has_value());
  EXPECT_EQ(function_call.parsed_arguments->at("location"), "SF");
  ASSERT_TRUE(response.output[3].function_call_output.has_value());
  const auto& function_call_output = *response.output[3].function_call_output;
  EXPECT_EQ(function_call_output.id, "func_output_1");
  ASSERT_TRUE(function_call_output.parsed_output_json.has_value());
  EXPECT_EQ(function_call_output.parsed_output_json->at("temp"), 70);
  ASSERT_TRUE(response.output[4].computer_call.has_value());
  const auto& computer_call = *response.output[4].computer_call;
  EXPECT_EQ(computer_call.id, "comp_1");
  EXPECT_EQ(computer_call.action.type, ResponseComputerToolCall::Action::Type::Scroll);
  ASSERT_TRUE(computer_call.action.scroll_y.has_value());
  EXPECT_EQ(*computer_call.action.scroll_y, 200);
  ASSERT_EQ(computer_call.pending_safety_checks.size(), 1u);
  EXPECT_EQ(computer_call.pending_safety_checks[0].code, "human_review");
  ASSERT_TRUE(response.output[6].web_search_call.has_value());
  const auto& web_search_call = *response.output[6].web_search_call;
  EXPECT_EQ(web_search_call.id, "ws_1");
  ASSERT_EQ(web_search_call.actions.size(), 1u);
  EXPECT_EQ(web_search_call.actions[0].type, ResponseFunctionWebSearch::Action::Type::Search);
  ASSERT_EQ(web_search_call.actions[0].sources.size(), 1u);
  EXPECT_EQ(web_search_call.actions[0].sources[0].url, "https://example.com/weather");
  ASSERT_TRUE(response.output[5].computer_call_output.has_value());
  const auto& computer_call_output = *response.output[5].computer_call_output;
  EXPECT_EQ(computer_call_output.id, "comp_out_1");
  ASSERT_TRUE(computer_call_output.screenshot.image_url.has_value());
  EXPECT_EQ(*computer_call_output.screenshot.image_url, "https://example.com/screenshot.png");
  ASSERT_EQ(computer_call_output.acknowledged_safety_checks.size(), 1u);
  EXPECT_EQ(computer_call_output.acknowledged_safety_checks[0].id, "psc_1");
  ASSERT_TRUE(response.output[7].local_shell_call.has_value());
  const auto& local_shell_call = *response.output[7].local_shell_call;
  EXPECT_EQ(local_shell_call.id, "shell_1");
  EXPECT_EQ(local_shell_call.action.type, ResponseLocalShellCall::Action::Type::Exec);
  ASSERT_EQ(local_shell_call.action.command.size(), 2u);
  EXPECT_EQ(local_shell_call.action.command[0], "ls");
  EXPECT_EQ(local_shell_call.action.env.at("PATH"), "/bin");
  ASSERT_TRUE(local_shell_call.action.timeout_ms.has_value());
  EXPECT_EQ(*local_shell_call.action.timeout_ms, 1000);
  ASSERT_TRUE(local_shell_call.action.working_directory.has_value());
  EXPECT_EQ(*local_shell_call.action.working_directory, "/tmp");
  ASSERT_TRUE(response.output[8].local_shell_output.has_value());
  const auto& local_shell_output = *response.output[8].local_shell_output;
  EXPECT_EQ(local_shell_output.id, "shell_out_1");
  ASSERT_TRUE(local_shell_output.parsed_output.has_value());
  EXPECT_EQ(local_shell_output.parsed_output->at("stdout"), "ok");
  ASSERT_TRUE(response.output[9].mcp_list_tools.has_value());
  const auto& mcp_list = *response.output[9].mcp_list_tools;
  EXPECT_EQ(mcp_list.id, "mcp_list");
  ASSERT_EQ(mcp_list.tools.size(), 1u);
  ASSERT_TRUE(mcp_list.tools[0].tags.has_value());
  EXPECT_EQ(mcp_list.tools[0].tags->at(0), "docs");
  ASSERT_TRUE(mcp_list.next_page_token.has_value());
  EXPECT_EQ(*mcp_list.next_page_token, "token-2");
  ASSERT_TRUE(response.output[10].mcp_call.has_value());
  const auto& mcp_call = *response.output[10].mcp_call;
  EXPECT_EQ(mcp_call.id, "mcp_call_1");
  EXPECT_EQ(mcp_call.status, ResponseMcpCall::Status::Completed);
  ASSERT_TRUE(response.output[11].mcp_approval_request.has_value());
  const auto& mcp_request = *response.output[11].mcp_approval_request;
  ASSERT_TRUE(mcp_request.suggested_decision.has_value());
  EXPECT_EQ(*mcp_request.suggested_decision, ResponseMcpApprovalRequest::Decision::Approved);
  ASSERT_TRUE(response.output[12].mcp_approval_response.has_value());
  const auto& mcp_response = *response.output[12].mcp_approval_response;
  EXPECT_EQ(mcp_response.decision, ResponseMcpApprovalResponse::Decision::Approved);
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

TEST(ResponsesResourceTest, CreateStreamParsesEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,"
                             "\"output_index\":0,\"content_index\":0,\"snapshot\":\"Hello\"}\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRequest request;
  request.model = "gpt-4o";
  ResponseInputItem input;
  input.type = ResponseInputItem::Type::Message;
  input.message.role = "user";
  ResponseInputContent content;
  content.type = ResponseInputContent::Type::Text;
  content.text = "Stream please";
  input.message.content.push_back(std::move(content));
  request.input.push_back(std::move(input));

  auto events = client.responses().stream(request);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_NE(events[0].data.find("Hello"), std::string::npos);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
}

TEST(ResponsesResourceTest, CreateStreamSnapshotAggregatesText) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"(data: {"response":{"id":"resp_snapshot","object":"response","created_at":1,"model":"gpt-4o-mini","status":"in_progress","metadata":{},"parallel_tool_calls":false,"output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}},"sequence_number":0,"type":"response.created"}

data: {"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]},"output_index":0,"sequence_number":1,"type":"response.output_item.added"}

data: {"content_index":0,"item_id":"msg_1","output_index":0,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":2,"type":"response.content_part.added"}

data: {"content_index":0,"delta":"Hello ","item_id":"msg_1","logprobs":[],"output_index":0,"sequence_number":3,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"world","item_id":"msg_1","logprobs":[],"output_index":0,"sequence_number":4,"type":"response.output_text.delta"}

data: {"content_index":0,"item_id":"msg_1","logprobs":[],"output_index":0,"sequence_number":5,"text":"Hello world","type":"response.output_text.done"}

data: {"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world","annotations":[]}]},"output_index":0,"sequence_number":6,"type":"response.output_item.done"}

data: {"response":{"id":"resp_snapshot","object":"response","created_at":1,"model":"gpt-4o-mini","status":"completed","metadata":{},"parallel_tool_calls":false,"output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world","annotations":[]}]}],"output_text":"Hello world","usage":{"input_tokens":0,"output_tokens":2,"total_tokens":2}},"sequence_number":7,"type":"response.completed"}

data: [DONE]

)";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRequest request;
  request.model = "gpt-4o-mini";
  ResponseInputItem input;
  input.type = ResponseInputItem::Type::Message;
  input.message.role = "user";
  ResponseInputContent content;
  content.type = ResponseInputContent::Type::Text;
  content.text = "Say hello";
  input.message.content.push_back(content);
  request.input.push_back(std::move(input));

  std::size_t event_count = 0;
  client.responses().stream(
      request,
      [&](const openai::ResponseStreamEvent& event) {
        ++event_count;
        return true;
      });

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
  EXPECT_EQ(event_count, 8u);
}

TEST(ResponsesResourceTest, CreateStreamSnapshotAggregatesReasoning) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"(data: {"response":{"id":"resp_reason","object":"response","created_at":10,"model":"o3","status":"in_progress","metadata":{},"parallel_tool_calls":false,"output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}},"sequence_number":0,"type":"response.created"}

data: {"item":{"id":"r1","type":"reasoning","status":"in_progress","summary":[],"content":[]},"output_index":0,"sequence_number":1,"type":"response.output_item.added"}

data: {"content_index":0,"item_id":"r1","output_index":0,"part":{"type":"reasoning_text","text":""},"sequence_number":2,"type":"response.content_part.added"}

data: {"content_index":0,"delta":"Chain: Step 1. ","item_id":"r1","output_index":0,"sequence_number":3,"type":"response.reasoning_text.delta"}

data: {"content_index":0,"delta":"Step 2.","item_id":"r1","output_index":0,"sequence_number":4,"type":"response.reasoning_text.delta"}

data: {"item":{"id":"msg_2","type":"message","role":"assistant","status":"in_progress","content":[]},"output_index":1,"sequence_number":5,"type":"response.output_item.added"}

data: {"content_index":0,"item_id":"msg_2","output_index":1,"part":{"type":"output_text","text":"","annotations":[]},"sequence_number":6,"type":"response.content_part.added"}

data: {"content_index":0,"delta":"The answer is ","item_id":"msg_2","logprobs":[],"output_index":1,"sequence_number":7,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"42","item_id":"msg_2","logprobs":[],"output_index":1,"sequence_number":8,"type":"response.output_text.delta"}

data: {"content_index":0,"item_id":"msg_2","logprobs":[],"output_index":1,"sequence_number":9,"text":"The answer is 42","type":"response.output_text.done"}

data: {"response":{"id":"resp_reason","object":"response","created_at":10,"model":"o3","status":"completed","metadata":{},"parallel_tool_calls":false,"output":[{"id":"r1","type":"reasoning","status":"completed","summary":[],"content":[{"type":"reasoning_text","text":"Chain: Step 1. Step 2."}]},{"id":"msg_2","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"The answer is 42","annotations":[]}]}],"output_text":"The answer is 42","usage":{"input_tokens":0,"output_tokens":6,"total_tokens":6}},"sequence_number":10,"type":"response.completed"}

data: [DONE]

)";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRequest request;
  request.model = "o3";
  ResponseInputItem input;
  input.type = ResponseInputItem::Type::Message;
  input.message.role = "user";
  ResponseInputContent content;
  content.type = ResponseInputContent::Type::Text;
  content.text = "Solve";
  input.message.content.push_back(content);
  request.input.push_back(std::move(input));

  std::size_t reasoning_events = 0;
  std::string final_text;
  client.responses().stream(
      request,
      [&](const openai::ResponseStreamEvent& event) {
        if (event.reasoning_text_delta) {
          ++reasoning_events;
        }
        if (event.text_done && event.text_done->text == "The answer is 42") {
          final_text = event.text_done->text;
        }
        return true;
      });

  EXPECT_EQ(reasoning_events, 2u);
  EXPECT_EQ(final_text, "The answer is 42");
}

TEST(ResponsesResourceTest, ListParsesResponsesArray) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string list_body = R"({
    "data": [
      {
        "id": "resp_1",
        "object": "response",
        "created": 10,
        "model": "gpt-4o",
        "output": []
      },
      {
        "id": "resp_2",
        "object": "response",
        "created": 11,
        "model": "gpt-4o",
        "output": []
      }
    ],
    "has_more": true
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, list_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto list = client.responses().list();
  ASSERT_EQ(list.data.size(), 2u);
  EXPECT_TRUE(list.has_more);
  EXPECT_EQ(list.data[0].id, "resp_1");
  EXPECT_EQ(list.data[1].id, "resp_2");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->method, "GET");
  EXPECT_NE(mock_ptr->last_request()->url.find("/responses"), std::string::npos);
}

TEST(ResponsesResourceTest, InputItemsListFetchesAndParsesItems) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "data": [
      {"type":"input_text","id":"item_1","text":"Hello world"},
      {"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hi!"}]},
      {"type":"function_call","id":"call_1","name":"lookup","arguments":"{}"}
    ],
    "first_id":"item_1",
    "last_id":"call_1",
    "has_more":false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseInputItemListParams params;
  params.include = std::vector<std::string>{"messages"};
  params.order = "asc";
  params.limit = 10;

  auto items = client.responses().input_items().list("resp_123", params);
  ASSERT_EQ(items.data.size(), 3u);
  EXPECT_FALSE(items.has_more);
  ASSERT_TRUE(items.first_id.has_value());
  EXPECT_EQ(*items.first_id, "item_1");
  ASSERT_TRUE(items.last_id.has_value());
  EXPECT_EQ(*items.last_id, "call_1");

  ASSERT_TRUE(items.data[0].input_text.has_value());
  EXPECT_EQ(items.data[0].input_text->text, "Hello world");
  ASSERT_TRUE(items.data[1].output_item.has_value());
  ASSERT_TRUE(items.data[1].output_item->message.has_value());
  EXPECT_EQ(items.data[1].output_item->message->role, "assistant");
  ASSERT_TRUE(items.data[2].output_item.has_value());
  EXPECT_EQ(items.data[2].output_item->item_type, "function_call");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "GET");
  EXPECT_NE(request.url.find("/responses/resp_123/input_items"), std::string::npos);
  EXPECT_NE(request.url.find("order=asc"), std::string::npos);
}

TEST(ResponsesResourceTest, CreateSerializesTypedFields) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(openai::HttpResponse{
      200,
      {},
      R"({"id":"resp_full","object":"response","created":1,"model":"gpt-4o","output":[]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ResponseRequest request;
  request.model = "gpt-4o-mini";
  request.metadata["project"] = "demo";
  request.background = true;
  request.conversation_id = "conv_123";
  request.include = {"usage", "messages"};
  request.instructions = "Keep answers brief.";
  request.max_output_tokens = 256;
  request.parallel_tool_calls = false;
  request.previous_response_id = "resp_prev";
  ResponsePrompt prompt;
  prompt.id = "prompt_abc";
  prompt.variables["foo"] = "bar";
  request.prompt = prompt;
  request.prompt_cache_key = "cache-key";
  ResponseReasoningConfig reasoning;
  reasoning.effort = "medium";
  request.reasoning = reasoning;
  request.safety_identifier = "safe-id";
  request.service_tier = "default";
  request.store = true;
  request.stream = false;
  ResponseStreamOptions stream_options;
  stream_options.include_usage = true;
  request.stream_options = stream_options;
  request.temperature = 0.1;
  request.top_p = 0.9;
  ResponseToolDefinition file_search_tool;
  file_search_tool.type = "file_search";
  ResponseFileSearchToolDefinition file_search_definition;
  file_search_definition.vector_store_ids = {"vs_123"};
  file_search_tool.file_search = file_search_definition;
  request.tools.push_back(std::move(file_search_tool));

  ResponseToolChoice tool_choice;
  tool_choice.kind = ResponseToolChoice::Kind::Simple;
  tool_choice.simple = ResponseToolChoiceSimpleOption::Required;
  request.tool_choice = tool_choice;

  ResponseInputItem input;
  input.type = ResponseInputItem::Type::Message;
  input.message.role = "user";
  input.message.metadata["topic"] = "intro";
  ResponseInputContent text_content;
  text_content.type = ResponseInputContent::Type::Text;
  text_content.text = "Hello!";
  input.message.content.push_back(text_content);
  ResponseInputContent image_content;
  image_content.type = ResponseInputContent::Type::Image;
  image_content.image_url = "https://example.com/image.png";
  image_content.image_detail = "auto";
  input.message.content.push_back(image_content);
  request.input.push_back(std::move(input));

  auto response = client.responses().create(request);
  EXPECT_EQ(response.id, "resp_full");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto payload = nlohmann::json::parse(mock_ptr->last_request()->body);

  EXPECT_EQ(payload.at("model"), "gpt-4o-mini");
  EXPECT_TRUE(payload.at("background").get<bool>());
  EXPECT_EQ(payload.at("conversation"), "conv_123");
  ASSERT_TRUE(payload.contains("include"));
  EXPECT_EQ(payload.at("include"), nlohmann::json::array({"usage", "messages"}));
  EXPECT_EQ(payload.at("instructions"), "Keep answers brief.");
  EXPECT_EQ(payload.at("max_output_tokens"), 256);
  EXPECT_FALSE(payload.at("parallel_tool_calls").get<bool>());
  EXPECT_EQ(payload.at("previous_response_id"), "resp_prev");
  EXPECT_EQ(payload.at("prompt").at("id"), "prompt_abc");
  EXPECT_EQ(payload.at("prompt").at("variables").at("foo"), "bar");
  EXPECT_EQ(payload.at("reasoning").at("effort"), "medium");
  EXPECT_EQ(payload.at("safety_identifier"), "safe-id");
  EXPECT_EQ(payload.at("service_tier"), "default");
  EXPECT_TRUE(payload.at("store").get<bool>());
  ASSERT_TRUE(payload.contains("stream_options"));
  EXPECT_TRUE(payload.at("stream_options").at("include_usage").get<bool>());
  EXPECT_DOUBLE_EQ(payload.at("temperature"), 0.1);
  EXPECT_DOUBLE_EQ(payload.at("top_p"), 0.9);
  ASSERT_TRUE(payload.contains("tools"));
  ASSERT_EQ(payload.at("tools").size(), 1u);
  EXPECT_EQ(payload.at("tools")[0].at("type"), "file_search");
  ASSERT_TRUE(payload.contains("tool_choice"));
  EXPECT_EQ(payload.at("tool_choice"), "required");

  ASSERT_EQ(payload.at("input").size(), 1u);
  const auto& first_input = payload.at("input")[0];
  EXPECT_EQ(first_input.at("role"), "user");
  EXPECT_EQ(first_input.at("metadata").at("topic"), "intro");
  ASSERT_EQ(first_input.at("content").size(), 2u);
  const auto& first_content = first_input.at("content")[0];
  EXPECT_EQ(first_content.at("type"), "input_text");
  EXPECT_EQ(first_content.at("text"), "Hello!");
  const auto& second_content = first_input.at("content")[1];
  EXPECT_EQ(second_content.at("type"), "input_image");
  EXPECT_EQ(second_content.at("image_url"), "https://example.com/image.png");
  EXPECT_EQ(second_content.at("detail"), "auto");
}

TEST(ResponsesStreamEventTest, ParsesTextDeltaEvent) {
  using namespace openai;

  ServerSentEvent sse_event;
  sse_event.event = "message";
  sse_event.data = R"({
    "type": "response.output_text.delta",
    "content_index": 0,
    "delta": "Hello",
    "item_id": "item_1",
    "output_index": 1,
    "sequence_number": 2,
    "logprobs": [
      { "token": "Hello", "logprob": -0.1, "top_logprobs": [ { "token": "Hello", "logprob": -0.1 } ] }
    ]
  })";

  auto parsed = parse_response_stream_event(sse_event);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->type, ResponseStreamEvent::Type::OutputTextDelta);
  ASSERT_TRUE(parsed->text_delta.has_value());
  EXPECT_EQ(parsed->text_delta->delta, "Hello");
  EXPECT_EQ(parsed->text_delta->sequence_number, 2);
  ASSERT_EQ(parsed->text_delta->logprobs.size(), 1u);
  EXPECT_EQ(parsed->text_delta->logprobs[0].token, "Hello");
  ASSERT_EQ(parsed->text_delta->logprobs[0].top_logprobs.size(), 1u);
}

TEST(ResponsesStreamEventTest, ParsesFunctionArgumentsDoneEvent) {
  using namespace openai;

  ServerSentEvent sse_event;
  sse_event.event = "message";
  sse_event.data = R"({
    "type": "response.function_call_arguments.done",
    "arguments": "{\"location\":\"SF\"}",
    "item_id": "item_2",
    "name": "weather",
    "output_index": 0,
    "sequence_number": 5
  })";

  auto parsed = parse_response_stream_event(sse_event);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->type, ResponseStreamEvent::Type::FunctionCallArgumentsDone);
  ASSERT_TRUE(parsed->function_arguments_done.has_value());
  EXPECT_EQ(parsed->function_arguments_done->name, "weather");
  EXPECT_EQ(parsed->function_arguments_done->sequence_number, 5);
}

TEST(ResponsesStreamEventTest, ParsesCreatedEvent) {
  using namespace openai;

  ServerSentEvent sse_event;
  sse_event.event = "message";
  sse_event.data = R"({
    "type": "response.created",
    "sequence_number": 1,
    "response": {
      "id": "resp_created",
      "object": "response",
      "created_at": 42,
      "model": "gpt-4o-mini",
      "status": "in_progress",
      "output": [],
      "metadata": {},
      "parallel_tool_calls": false,
      "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }
  })";

  auto parsed = parse_response_stream_event(sse_event);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->type, ResponseStreamEvent::Type::Created);
  EXPECT_EQ(parsed->sequence_number, 1);
  ASSERT_TRUE(parsed->created.has_value());
  EXPECT_EQ(parsed->created->response.id, "resp_created");
  EXPECT_EQ(parsed->created->response.created, 42);
  ASSERT_TRUE(parsed->created->response.status.has_value());
  EXPECT_EQ(*parsed->created->response.status, "in_progress");
}

TEST(ResponsesResourceTest, ListPageSupportsCursorPagination) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string first_body = R"({
    "data": [
      { "id": "resp_1", "object": "response", "created": 1, "model": "gpt" }
    ],
    "has_more": true,
    "last_id": "resp_1"
  })";

  const std::string second_body = R"({
    "data": [
      { "id": "resp_2", "object": "response", "created": 2, "model": "gpt" }
    ],
    "has_more": false,
    "last_id": "resp_2"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, first_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, second_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));
  auto page = client.responses().list_page();

  EXPECT_EQ(mock_ptr->call_count(), 1u);
  EXPECT_TRUE(page.has_next_page());
  ASSERT_TRUE(page.next_cursor().has_value());
  EXPECT_EQ(*page.next_cursor(), "resp_1");
  ASSERT_EQ(page.data().size(), 1u);
  EXPECT_EQ(page.data().front().id, "resp_1");

  auto next_page = page.next_page();
  EXPECT_EQ(mock_ptr->call_count(), 2u);
  EXPECT_FALSE(next_page.has_next_page());
  ASSERT_EQ(next_page.data().size(), 1u);
  EXPECT_EQ(next_page.data().front().id, "resp_2");

  const auto& last_request = mock_ptr->last_request();
  ASSERT_TRUE(last_request.has_value());
  EXPECT_NE(last_request->url.find("after=resp_1"), std::string::npos);
}

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
  request.model = "gpt-4o-mini";
  ResponseInput input;
  input.role = "user";
  input.content.push_back(ResponseInputContent{.text = "Say hello"});
  request.input.push_back(std::move(input));

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
  ResponseInput input;
  input.role = "user";
  input.content.push_back(ResponseInputContent{.text = "Stream please"});
  request.input.push_back(std::move(input));

  auto events = client.responses().create_stream(request);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_NE(events[0].data.find("Hello"), std::string::npos);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
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
  request.tools.push_back(nlohmann::json::object({{"type", "file_search"}}));
  request.tool_choice = nlohmann::json::object({{"type", "required"}});

  ResponseInput input;
  input.role = "user";
  input.metadata["topic"] = "intro";
  ResponseInputContent text_content;
  text_content.type = ResponseInputContent::Type::Text;
  text_content.text = "Hello!";
  input.content.push_back(text_content);
  ResponseInputContent image_content;
  image_content.type = ResponseInputContent::Type::Image;
  image_content.image_url = "https://example.com/image.png";
  image_content.image_detail = "auto";
  input.content.push_back(image_content);
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
  EXPECT_EQ(payload.at("tool_choice").at("type"), "required");

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

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

  auto events = client.chat().completions().stream(request);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_TRUE(events[0].event.has_value());
  EXPECT_EQ(*events[0].event, "message");
  EXPECT_NE(events[0].data.find("Hello"), std::string::npos);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
}

TEST(ChatCompletionsResourceTest, CreateStreamInvokesCallbackIncrementally) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body =
      "event: message\n"
      "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
      "event: message\n"
      "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n";

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

  std::vector<std::string> deltas;

  client.chat().completions().stream(
      request,
      [&](const ServerSentEvent& event) {
        if (event.data.find("content") != std::string::npos) {
          deltas.push_back(event.data);
        }
        // Stop after first chunk to ensure early termination works.
        return deltas.size() < 1;
      });

  ASSERT_EQ(deltas.size(), 1u);
  EXPECT_NE(deltas[0].find("Hello"), std::string::npos);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "text/event-stream");
}

TEST(ChatCompletionsResourceTest, CreateSerializesTypedFields) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200, {}, R"({"id":"chatcmpl-full","object":"chat.completion","created":1,"model":"gpt-4o","choices":[]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionRequest request;
  request.model = "gpt-4o-mini";
  request.metadata["project"] = "demo";
  request.max_tokens = 128;
  request.temperature = 0.4;
  request.top_p = 0.9;
  request.frequency_penalty = -0.6;
  request.presence_penalty = 0.1;
  request.logit_bias["1234"] = -2.0;
  request.logprobs = true;
  request.top_logprobs = 3;
  request.stop = std::vector<std::string>{"stop"};
  request.seed = 42;
  ChatResponseFormat response_format;
  response_format.type = "json_schema";
  response_format.json_schema = nlohmann::json::object({{"name", "Demo"}, {"schema", nlohmann::json::object()}});
  request.response_format = response_format;

  ChatCompletionToolDefinition tool;
  tool.type = "function";
  ChatToolFunctionDefinition fn;
  fn.name = "lookup";
  fn.description = "Lookup data";
  fn.parameters = nlohmann::json::object({{"type", "object"}});
  tool.function = fn;
  request.tools.push_back(tool);

  ChatToolChoice tool_choice;
  tool_choice.type = ChatToolChoice::Type::NamedFunction;
  ChatCompletionNamedToolChoice named_choice;
  named_choice.type = "function";
  named_choice.function.name = "lookup";
  tool_choice.named_function = named_choice;
  request.tool_choice = tool_choice;

  request.parallel_tool_calls = false;
  request.user = "user-123";
  request.stream = false;

  ChatMessage system_message;
  system_message.role = "system";
  system_message.metadata["scope"] = "demo";
  ChatMessageContent system_text;
  system_text.type = ChatMessageContent::Type::Text;
  system_text.text = "You are a demo assistant.";
  system_message.content.push_back(system_text);
  request.messages.push_back(system_message);

  ChatMessage user_message;
  user_message.role = "user";
  ChatMessageContent text_content;
  text_content.type = ChatMessageContent::Type::Text;
  text_content.text = "Hello";
  user_message.content.push_back(text_content);
  ChatMessageContent image_content;
  image_content.type = ChatMessageContent::Type::Image;
  image_content.image_url = "https://example.com/image.png";
  image_content.image_detail = "high";
  user_message.content.push_back(image_content);
  request.messages.push_back(user_message);

  auto completion = client.chat().completions().create(request);
  EXPECT_EQ(completion.id, "chatcmpl-full");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto payload = nlohmann::json::parse(mock_ptr->last_request()->body);
  EXPECT_EQ(payload.at("model"), "gpt-4o-mini");
  EXPECT_EQ(payload.at("metadata").at("project"), "demo");
  EXPECT_EQ(payload.at("max_tokens"), 128);
  EXPECT_DOUBLE_EQ(payload.at("temperature"), 0.4);
  EXPECT_DOUBLE_EQ(payload.at("top_p"), 0.9);
  EXPECT_DOUBLE_EQ(payload.at("frequency_penalty"), -0.6);
  EXPECT_DOUBLE_EQ(payload.at("presence_penalty"), 0.1);
  EXPECT_EQ(payload.at("logit_bias").at("1234"), -2.0);
  EXPECT_TRUE(payload.at("logprobs").get<bool>());
  EXPECT_EQ(payload.at("top_logprobs"), 3);
  ASSERT_TRUE(payload.contains("stop"));
  EXPECT_EQ(payload.at("stop"), nlohmann::json::array({"stop"}));
  EXPECT_EQ(payload.at("seed"), 42);
  EXPECT_EQ(payload.at("response_format").at("type"), "json_schema");
  ASSERT_TRUE(payload.at("tools").is_array());
  ASSERT_EQ(payload.at("tools").size(), 1u);
  EXPECT_EQ(payload.at("tools")[0].at("type"), "function");
  EXPECT_EQ(payload.at("tools")[0].at("function").at("name"), "lookup");
  ASSERT_TRUE(payload.contains("tool_choice"));
  EXPECT_EQ(payload.at("tool_choice").at("type"), "function");
  EXPECT_EQ(payload.at("tool_choice").at("function").at("name"), "lookup");
  EXPECT_FALSE(payload.at("parallel_tool_calls").get<bool>());
  EXPECT_EQ(payload.at("user"), "user-123");
  ASSERT_TRUE(payload.contains("messages"));
  ASSERT_EQ(payload.at("messages").size(), 2u);
  const auto& first_message = payload.at("messages")[0];
  EXPECT_EQ(first_message.at("role"), "system");
  EXPECT_EQ(first_message.at("metadata").at("scope"), "demo");
  EXPECT_EQ(first_message.at("content")[0].at("type"), "text");
  EXPECT_EQ(first_message.at("content")[0].at("text"), "You are a demo assistant.");
  const auto& second_message = payload.at("messages")[1];
  EXPECT_EQ(second_message.at("content")[0].at("text"), "Hello");
  EXPECT_EQ(second_message.at("content")[1].at("type"), "input_image");
  EXPECT_EQ(second_message.at("content")[1].at("image_url"), "https://example.com/image.png");
  EXPECT_EQ(second_message.at("content")[1].at("detail"), "high");
}

TEST(ChatCompletionsResourceTest, RetrieveStoredCompletionParsesMetadataAndServiceTier) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200,
      {},
      R"({"id":"chatcmpl-store","object":"chat.completion","created":10,"model":"gpt-4o","service_tier":"priority","metadata":{"ticket":"42"},"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  auto completion = client.chat().completions().retrieve("chatcmpl-store");
  EXPECT_EQ(completion.id, "chatcmpl-store");
  ASSERT_TRUE(completion.service_tier.has_value());
  EXPECT_EQ(*completion.service_tier, "priority");
  ASSERT_TRUE(completion.metadata.count("ticket"));
  EXPECT_EQ(completion.metadata.at("ticket"), "42");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_NE(mock_ptr->last_request()->url.find("/chat/completions/chatcmpl-store"), std::string::npos);
}

TEST(ChatCompletionsResourceTest, UpdateStoredCompletionSerializesMetadata) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200,
      {},
      R"({"id":"chatcmpl-store","object":"chat.completion","created":10,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionUpdateRequest update;
  update.metadata = std::map<std::string, std::string>{{"priority", "high"}};

  auto completion = client.chat().completions().update("chatcmpl-store", update);
  EXPECT_EQ(completion.id, "chatcmpl-store");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto payload = nlohmann::json::parse(mock_ptr->last_request()->body);
  ASSERT_TRUE(payload.contains("metadata"));
  EXPECT_EQ(payload.at("metadata").at("priority"), "high");
}

TEST(ChatCompletionsResourceTest, ListStoredCompletionsAppliesQueryParameters) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200,
      {},
      R"({"data":[{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}],"has_more":false,"next_cursor":null})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionListParams params;
  params.limit = 2;
  params.order = "desc";
  params.metadata = std::map<std::string, std::string>{{"ticket", "42"}};

  auto list = client.chat().completions().list(params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].id, "chatcmpl-1");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& url = mock_ptr->last_request()->url;
  EXPECT_NE(url.find("limit=2"), std::string::npos);
  EXPECT_NE(url.find("order=desc"), std::string::npos);
  EXPECT_NE(url.find("metadata%5Bticket%5D=42"), std::string::npos);
}

TEST(ChatCompletionsResourceTest, DeleteStoredCompletionParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(
      HttpResponse{200, {}, R"({"id":"chatcmpl-store","deleted":true,"object":"chat.completion.deleted"})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  auto removed = client.chat().completions().remove("chatcmpl-store");
  EXPECT_TRUE(removed.deleted);
  EXPECT_EQ(removed.id, "chatcmpl-store");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "*/*");
}

TEST(ChatCompletionsResourceTest, ListStoredMessagesParsesContent) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200,
      {},
      R"({"data":[{"id":"msg_1","role":"tool","tool_call_id":"call_123","content":"{\"value\":42}"}],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  auto messages = client.chat().completions().messages().list("chatcmpl-store");
  ASSERT_EQ(messages.data.size(), 1u);
  EXPECT_EQ(messages.data[0].id, "msg_1");
  EXPECT_TRUE(messages.data[0].message.tool_call_id.has_value());
  EXPECT_EQ(*messages.data[0].message.tool_call_id, "call_123");
  ASSERT_EQ(messages.data[0].content_parts.size(), 1u);
  EXPECT_EQ(messages.data[0].content_parts[0].text, "{\"value\":42}");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_NE(mock_ptr->last_request()->url.find("/chat/completions/chatcmpl-store/messages"), std::string::npos);
}

TEST(ChatCompletionsResourceTest, RunToolsExecutesCallbacksAndLoops) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string first =
      R"({"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"hello\"}"}}]}}]})";
  const std::string second =
      R"({"id":"chatcmpl-2","object":"chat.completion","created":2,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]})";

  mock_ptr->enqueue_response(HttpResponse{200, {}, first});
  mock_ptr->enqueue_response(HttpResponse{200, {}, second});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(options, std::move(mock_client));

  ChatCompletionRequest request;
  request.model = "gpt-4o";
  ChatMessage user;
  user.role = "user";
  ChatMessageContent content;
  content.type = ChatMessageContent::Type::Text;
  content.text = "hello";
  user.content.push_back(content);
  request.messages.push_back(user);

  ChatToolFunctionDefinition fn;
  fn.name = "lookup";
  fn.description = "Lookup a value";
  fn.parameters = nlohmann::json::object({{"type", "object"}});

  ChatCompletionToolDefinition tool;
  tool.type = "function";
  tool.function = fn;

  bool invoked = false;
  ChatCompletionToolRunParams params;
  params.request = request;
  params.functions.push_back(ChatToolFunctionHandler{
      tool,
      [&](const nlohmann::json& args) {
        invoked = true;
        EXPECT_TRUE(args.is_object());
        EXPECT_EQ(args.at("query"), "hello");
        return nlohmann::json::object({{"result", "42"}});
      }});

  auto result = client.chat().completions().run_tools(params);

  EXPECT_TRUE(invoked);
  EXPECT_EQ(result.final_completion.id, "chatcmpl-2");
  EXPECT_EQ(result.completions.size(), 2u);
  ASSERT_GE(result.transcript.size(), 4u);
  EXPECT_EQ(result.transcript[result.transcript.size() - 2].role, "tool");
  EXPECT_EQ(result.transcript.back().role, "assistant");
  EXPECT_EQ(result.transcript[result.transcript.size() - 2].content[0].text, R"({"result":"42"})");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto serialized = nlohmann::json::parse(mock_ptr->last_request()->body);
  ASSERT_TRUE(serialized.contains("messages"));
  bool tool_message_found = false;
  for (const auto& message_json : serialized.at("messages")) {
    if (message_json.contains("role") && message_json.at("role") == "tool") {
      tool_message_found = true;
      EXPECT_EQ(message_json.at("tool_call_id"), "call_1");
    }
  }
  EXPECT_TRUE(tool_message_found);
  EXPECT_EQ(mock_ptr->call_count(), 2u);
}

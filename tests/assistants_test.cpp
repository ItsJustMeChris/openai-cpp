#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/assistants.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(AssistantsResourceTest, CreateSendsBetaHeaderAndSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "asst_123",
    "created_at": 1700000000,
    "description": "helper",
    "instructions": "Be concise",
    "metadata": {"project": "demo"},
    "model": "gpt-4o",
    "name": "Demo assistant",
    "object": "assistant",
    "tools": [{"type": "code_interpreter"}],
    "temperature": 0.3
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  AssistantCreateRequest request;
  request.model = "gpt-4o";
  request.description = "helper";
  request.instructions = "Be concise";
  request.name = "Demo assistant";
  request.metadata["project"] = "demo";
  AssistantTool code_tool;
  code_tool.type = AssistantTool::Type::CodeInterpreter;
  request.tools.push_back(code_tool);
  AssistantTool function_tool;
  function_tool.type = AssistantTool::Type::Function;
  AssistantTool::FunctionDefinition fn;
  fn.name = "lookup";
  fn.description = std::string("Lookup info");
  fn.parameters = nlohmann::json::object({{"type", "object"}});
  function_tool.function = fn;
  request.tools.push_back(function_tool);
  AssistantToolResources resources;
  resources.code_interpreter_file_ids.push_back("file_1");
  resources.file_search_vector_store_ids.push_back("vs_1");
  request.tool_resources = resources;
  AssistantResponseFormat format;
  format.type = "json_schema";
  format.json_schema = nlohmann::json::object({{"name", "Test"}});
  request.response_format = format;
  request.temperature = 0.3;
  request.top_p = 0.9;

  auto assistant = client.assistants().create(request);
  EXPECT_EQ(assistant.id, "asst_123");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.headers.at("OpenAI-Beta"), "assistants=v2");
  const auto payload = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(payload.at("model"), "gpt-4o");
  EXPECT_EQ(payload.at("description"), "helper");
  EXPECT_EQ(payload.at("instructions"), "Be concise");
  EXPECT_EQ(payload.at("name"), "Demo assistant");
  EXPECT_EQ(payload.at("metadata").at("project"), "demo");
  ASSERT_EQ(payload.at("tools").size(), 2u);
  EXPECT_EQ(payload.at("tools")[0].at("type"), "code_interpreter");
  EXPECT_EQ(payload.at("tools")[1].at("function").at("name"), "lookup");
  ASSERT_TRUE(payload.contains("tool_resources"));
  EXPECT_EQ(payload.at("tool_resources").at("code_interpreter").at("file_ids")[0], "file_1");
  EXPECT_EQ(payload.at("tool_resources").at("file_search").at("vector_store_ids")[0], "vs_1");
  EXPECT_EQ(payload.at("response_format").at("type"), "json_schema");
  EXPECT_DOUBLE_EQ(payload.at("temperature"), 0.3);
  EXPECT_DOUBLE_EQ(payload.at("top_p"), 0.9);
}

TEST(AssistantsResourceTest, UpdateParsesAssistant) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "asst_123",
    "created_at": 1700000000,
    "description": "updated",
    "instructions": "Be helpful",
    "metadata": {"team": "core"},
    "model": "gpt-4o",
    "name": "Updated assistant",
    "object": "assistant",
    "tools": [],
    "top_p": 0.8
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  AssistantUpdateRequest request;
  request.description = "updated";
  request.instructions = "Be helpful";
  request.name = "Updated assistant";
  request.top_p = 0.8;
  request.metadata = std::map<std::string, std::string>{{"team", "core"}};

  auto assistant = client.assistants().update("asst_123", request);
  EXPECT_EQ(assistant.name.value_or(""), "Updated assistant");
  EXPECT_TRUE(assistant.metadata.count("team"));
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

TEST(AssistantsResourceTest, ListAndDeleteWork) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string list_body = R"({
    "data": [
      {"id": "asst_1", "object": "assistant", "created_at": 1, "model": "gpt-4o", "tools": []},
      {"id": "asst_2", "object": "assistant", "created_at": 2, "model": "gpt-4o-mini", "tools": []}
    ],
    "has_more": false
  })";

  const std::string delete_body = R"({"id": "asst_1", "deleted": true, "object": "assistant.deleted"})";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, list_body});
  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, delete_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  AssistantListParams params;
  params.limit = 10;
  params.order = "desc";
  auto list = client.assistants().list(params);
  ASSERT_EQ(list.data.size(), 2u);
  EXPECT_FALSE(list.has_more);

  auto del = client.assistants().remove("asst_1");
  EXPECT_TRUE(del.deleted);
  EXPECT_EQ(del.object, "assistant.deleted");
}

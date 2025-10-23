#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/evals.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(EvalsResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "eval_123",
    "created_at": 1700000000,
    "data_source_config": {"type":"custom","schema":{"type":"object"}},
    "metadata": {"env":"test"},
    "name": "My Eval",
    "object": "eval",
    "testing_criteria": [
      {"type":"string_check","input":"hello","name":"Check","operation":"eq","reference":"hello"}
    ]
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  evals::EvaluationCreateParams params;
  evals::CreateCustomDataSourceConfig custom;
  custom.item_schema = nlohmann::json::object({{"type", "object"}});
  params.data_source_config = custom;

  evals::StringCheckGrader string_check;
  string_check.grader.input = "hello";
  string_check.grader.name = "Check";
  string_check.grader.operation = "eq";
  string_check.grader.reference = "hello";
  string_check.grader.type = "string_check";
  params.testing_criteria.push_back(string_check);

  params.metadata = evals::Metadata{{"env", "test"}};
  params.name = std::string("My Eval");

  auto evaluation = client.evals().create(params);
  EXPECT_EQ(evaluation.id, "eval_123");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "POST");
  EXPECT_NE(last_request.url.find("/evals"), std::string::npos);

  const auto body = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(body.at("data_source_config").at("type"), "custom");
  EXPECT_EQ(body.at("testing_criteria").size(), 1u);
  EXPECT_EQ(body.at("testing_criteria")[0].at("type"), "string_check");
  EXPECT_EQ(body.at("metadata").at("env"), "test");
  EXPECT_EQ(body.at("name"), "My Eval");
}

TEST(EvalsRunsResourceTest, CreateRunSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "run_123",
    "created_at": 1700000100,
    "data_source": {
      "type": "completions",
      "source": {"type":"file_id","id":"file_1"}
    },
    "error": {"code":"", "message":""},
    "eval_id": "eval_123",
    "metadata": {"batch":"A"},
    "model": "o3-mini",
    "name": "Initial Run",
    "object": "eval.run",
    "per_model_usage": [],
    "per_testing_criteria_results": [],
    "report_url": "https://dashboard",
    "result_counts": {"errored":0,"failed":0,"passed":0,"total":0},
    "status": "queued"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  evals::RunCreateParams params;
  evals::CreateCompletionsRunDataSource completions;
  evals::RunFileIDSource file_source;
  file_source.id = "file_1";
  completions.source = file_source;
  completions.model = std::string("o3-mini");
  evals::RunSamplingParams sampling;
  sampling.temperature = 0.2;
  completions.sampling_params = sampling;
  params.data_source = completions;
  params.metadata = evals::Metadata{{"batch", "A"}};
  params.name = std::string("Initial Run");

  auto run = client.evals().runs().create("eval_123", params);
  EXPECT_EQ(run.id, "run_123");
  EXPECT_EQ(run.status, "queued");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "POST");
  EXPECT_NE(last_request.url.find("/evals/eval_123/runs"), std::string::npos);
  const auto body = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(body.at("data_source").at("source").at("id"), "file_1");
  EXPECT_EQ(body.at("data_source").at("model"), "o3-mini");
  EXPECT_EQ(body.at("metadata").at("batch"), "A");
  EXPECT_EQ(body.at("name"), "Initial Run");
}

TEST(EvalsRunsOutputItemsResourceTest, ListParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "data": [{
      "id": "out_1",
      "created_at": 1700000200,
      "datasource_item": {"id": 1},
      "datasource_item_id": 1,
      "eval_id": "eval_123",
      "object": "eval.run.output_item",
      "results": [{"name":"Check","passed":true,"score":1.0}],
      "run_id": "run_123",
      "sample": {
        "error": {"code": "", "message": ""},
        "finish_reason": "stop",
        "input": [{"role":"user","content":"hello"}],
        "max_completion_tokens": 64,
        "model": "o3-mini",
        "output": [{"role":"assistant","content":"world"}],
        "seed": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "usage": {"input_tokens":1,"output_tokens":1,"total_tokens":2}
      },
      "status": "completed"
    }],
    "has_more": false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  evals::OutputItemListParams params;
  params.limit = 10;

  auto list = client.evals().runs().output_items().list("eval_123", "run_123", params);
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].id, "out_1");
  EXPECT_EQ(list.data[0].results[0].name, "Check");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "GET");
  EXPECT_NE(last_request.url.find("/evals/eval_123/runs/run_123/output_items"), std::string::npos);
  EXPECT_NE(last_request.url.find("limit=10"), std::string::npos);
}

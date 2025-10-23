#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/fine_tuning.hpp"
#include "openai/client.hpp"
#include "support/mock_http_client.hpp"

namespace openai {
namespace {

using namespace openai::testing;

TEST(FineTuningAlphaGradersTest, RunAndValidateRequests) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string run_response_body = R"({
    "metadata": {
      "errors": {
        "formula_parse_error": false,
        "invalid_variable_error": false,
        "model_grader_parse_error": false,
        "model_grader_refusal_error": false,
        "model_grader_server_error": false,
        "model_grader_server_error_details": null,
        "other_error": false,
        "python_grader_runtime_error": false,
        "python_grader_runtime_error_details": null,
        "python_grader_server_error": false,
        "python_grader_server_error_type": null,
        "sample_parse_error": false,
        "truncated_observation_error": false,
        "unresponsive_reward_error": false
      },
      "execution_time": 1.5,
      "name": "string_check",
      "sampled_model_name": null,
      "scores": {"score": 0.5},
      "token_usage": 12,
      "type": "string_check"
    },
    "model_grader_token_usage_per_model": {"gpt-4o": {"tokens": 12}},
    "reward": 0.6,
    "sub_rewards": {"detail": 0.4}
  })";

  const std::string validate_response_body = R"({
    "grader": {
      "type": "string_check",
      "input": "{{ sample }}",
      "name": "grader",
      "operation": "eq",
      "reference": "expected"
    }
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, run_response_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, validate_response_body});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  graders::StringCheckGrader grader;
  grader.input = "{{ sample }}";
  grader.name = "grader";
  grader.operation = "eq";
  grader.reference = "expected";
  grader.type = "string_check";

  GraderRunParams run_params;
  run_params.grader = grader;
  run_params.model_sample = "sample-value";

  auto run_result = client.fine_tuning().alpha().graders().run(run_params);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  auto run_request = *mock_ptr->last_request();
  auto run_payload = nlohmann::json::parse(run_request.body);
  EXPECT_EQ(run_request.method, "POST");
  EXPECT_NE(run_request.url.find("/fine_tuning/alpha/graders/run"), std::string::npos);
  EXPECT_EQ(run_payload.at("model_sample"), "sample-value");
  EXPECT_EQ(run_payload.at("grader").at("operation"), "eq");
  EXPECT_DOUBLE_EQ(run_result.reward, 0.6);
  EXPECT_EQ(run_result.metadata.name, "string_check");
  ASSERT_TRUE(run_result.metadata.token_usage.has_value());
  EXPECT_DOUBLE_EQ(*run_result.metadata.token_usage, 12.0);

  GraderValidateParams validate_params;
  validate_params.grader = grader;

  auto validate_result = client.fine_tuning().alpha().graders().validate(validate_params);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  auto validate_request = *mock_ptr->last_request();
  auto validate_payload = nlohmann::json::parse(validate_request.body);
  EXPECT_EQ(validate_request.method, "POST");
  EXPECT_NE(validate_request.url.find("/fine_tuning/alpha/graders/validate"), std::string::npos);
  EXPECT_EQ(validate_payload.at("grader").at("reference"), "expected");
  ASSERT_TRUE(validate_result.grader.has_value());
  EXPECT_TRUE(std::holds_alternative<graders::StringCheckGrader>(*validate_result.grader));
}

}  // namespace
}  // namespace openai

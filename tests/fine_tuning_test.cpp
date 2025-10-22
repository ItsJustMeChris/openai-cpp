#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/client.hpp"
#include "openai/fine_tuning.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(FineTuningJobsResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_http.get();

  const std::string response_body = R"({
    "id": "ft_job_123",
    "created_at": 1700000000,
    "object": "fine_tuning.job",
    "model": "gpt-4o-mini",
    "organization_id": "org_123",
    "result_files": [],
    "seed": 42,
    "status": "queued",
    "training_file": "file_train"
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_http));

  JobCreateParams params;
  params.model = "gpt-4o-mini";
  params.training_file = "file_train";
  SupervisedHyperparameters hyper;
  hyper.batch_size = AutoInteger{16};
  hyper.learning_rate_multiplier = AutoNumber{std::string("auto")};
  params.hyperparameters = hyper;
  WandbIntegrationParams wandb;
  wandb.project = "demo";
  wandb.tags = {"tag1"};
  params.integrations.push_back({wandb});
  params.metadata = std::map<std::string, std::string>{{"purpose", "demo"}};
  FineTuningMethod method;
  method.type = FineTuningMethod::Type::Supervised;
  SupervisedMethodConfig supervised;
  SupervisedHyperparameters supervised_hyper;
  supervised_hyper.n_epochs = AutoInteger{std::string("auto")};
  supervised.hyperparameters = supervised_hyper;
  method.supervised = supervised;
  params.method = method;
  params.seed = 42;
  params.suffix = "custom";
  params.validation_file = "file_valid";

  auto job = client.fine_tuning().jobs().create(params);
  EXPECT_EQ(job.id, "ft_job_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/fine_tuning/jobs"), std::string::npos);

  const auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("model"), "gpt-4o-mini");
  EXPECT_EQ(payload.at("training_file"), "file_train");
  EXPECT_EQ(payload.at("seed"), 42);
  EXPECT_EQ(payload.at("suffix"), "custom");
  EXPECT_EQ(payload.at("validation_file"), "file_valid");
  EXPECT_EQ(payload.at("metadata").at("purpose"), "demo");
  EXPECT_EQ(payload.at("integrations").at(0).at("type"), "wandb");
  EXPECT_EQ(payload.at("method").at("type"), "supervised");
}

TEST(FineTuningJobsResourceTest, ListAppliesMetadataFilter) {
  using namespace openai;

  auto mock_http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_http.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"data":[],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_http));

  JobListParams params;
  params.limit = 5;
  params.after = std::string("ft_job_prev");
  params.order = std::string("desc");
  params.metadata = std::map<std::string, std::string>{{"purpose", "demo"}};

  auto list = client.fine_tuning().jobs().list(params);
  EXPECT_FALSE(list.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=5"), std::string::npos);
  EXPECT_NE(request.url.find("after=ft_job_prev"), std::string::npos);
  EXPECT_NE(request.url.find("order=desc"), std::string::npos);
  EXPECT_NE(request.url.find("metadata%5Bpurpose%5D=demo"), std::string::npos);
}

TEST(FineTuningJobsResourceTest, CancelUsesCorrectPath) {
  using namespace openai;

  auto mock_http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_http.get();

  const std::string response_body = R"({
    "id": "ft_job_123",
    "created_at": 1700000000,
    "object": "fine_tuning.job",
    "model": "gpt-4o-mini",
    "organization_id": "org_123",
    "result_files": [],
    "seed": 42,
    "status": "cancelled",
    "training_file": "file_train"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_http));

  auto job = client.fine_tuning().jobs().cancel("ft_job_123");
  EXPECT_EQ(job.status, "cancelled");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/fine_tuning/jobs/ft_job_123/cancel"), std::string::npos);
}

TEST(FineTuningJobsResourceTest, ListEventsSetsQueryParams) {
  using namespace openai;

  auto mock_http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_http.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"data":[],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_http));

  JobListEventsParams params;
  params.limit = 20;
  params.after = std::string("evt_prev");

  auto events = client.fine_tuning().jobs().list_events("ft_job_123", params);
  EXPECT_FALSE(events.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("/fine_tuning/jobs/ft_job_123/events"), std::string::npos);
  EXPECT_NE(request.url.find("limit=20"), std::string::npos);
  EXPECT_NE(request.url.find("after=evt_prev"), std::string::npos);
}

TEST(FineTuningJobCheckpointsResourceTest, ListRoutesCorrectly) {
  using namespace openai;

  auto mock_http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_http.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"data":[],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_http));

  FineTuningCheckpointListParams params;
  params.limit = 15;
  params.after = std::string("cp_prev");

  auto checkpoints = client.fine_tuning().jobs().checkpoints().list("ft_job_123", params);
  EXPECT_FALSE(checkpoints.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("/fine_tuning/jobs/ft_job_123/checkpoints"), std::string::npos);
  EXPECT_NE(request.url.find("limit=15"), std::string::npos);
  EXPECT_NE(request.url.find("after=cp_prev"), std::string::npos);
}


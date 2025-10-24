#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/graders.hpp"

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

using AutoNumber = std::variant<double, std::string>;
using AutoInteger = std::variant<int, std::string>;

struct FineTuningJobError {
  std::string code;
  std::string message;
  std::optional<std::string> param;
};

struct DpoHyperparameters {
  std::optional<AutoInteger> batch_size;
  std::optional<AutoNumber> beta;
  std::optional<AutoNumber> learning_rate_multiplier;
  std::optional<AutoInteger> n_epochs;
};

struct DpoMethodConfig {
  std::optional<DpoHyperparameters> hyperparameters;
};

struct ReinforcementHyperparameters {
  std::optional<AutoInteger> batch_size;
  std::optional<AutoNumber> compute_multiplier;
  std::optional<AutoInteger> eval_interval;
  std::optional<AutoInteger> eval_samples;
  std::optional<AutoNumber> learning_rate_multiplier;
  std::optional<AutoInteger> n_epochs;
  std::optional<std::string> reasoning_effort;
};

struct ReinforcementMethodConfig {
  std::optional<std::string> grader;
  std::optional<ReinforcementHyperparameters> hyperparameters;
};

struct SupervisedHyperparameters {
  std::optional<AutoInteger> batch_size;
  std::optional<AutoNumber> learning_rate_multiplier;
  std::optional<AutoInteger> n_epochs;
};

struct SupervisedMethodConfig {
  std::optional<SupervisedHyperparameters> hyperparameters;
};

struct FineTuningMethod {
  enum class Type { Supervised, DPO, Reinforcement };
  Type type = Type::Supervised;
  std::optional<SupervisedMethodConfig> supervised;
  std::optional<DpoMethodConfig> dpo;
  std::optional<ReinforcementMethodConfig> reinforcement;
};

struct WandbIntegrationParams {
  std::string project;
  std::optional<std::string> entity;
  std::optional<std::string> name;
  std::vector<std::string> tags;
};

struct JobIntegrationParams {
  WandbIntegrationParams wandb;
};

struct JobCreateParams {
  std::string model;
  std::string training_file;
  std::optional<SupervisedHyperparameters> hyperparameters;
  std::vector<JobIntegrationParams> integrations;
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<FineTuningMethod> method;
  std::optional<int> seed;
  std::optional<std::string> suffix;
  std::optional<std::string> validation_file;
};

struct JobCreateDeprecatedHyperparameters {
  std::optional<AutoInteger> batch_size;
  std::optional<AutoNumber> learning_rate_multiplier;
  std::optional<AutoInteger> n_epochs;
};

struct JobListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> order;
  std::optional<std::map<std::string, std::string>> metadata;
  bool metadata_null = false;
};

struct JobListEventsParams {
  std::optional<int> limit;
  std::optional<std::string> after;
};

struct FineTuningJobIntegration {
  std::string type;
  WandbIntegrationParams wandb;
};

struct FineTuningJobHyperparameters {
  std::optional<AutoInteger> batch_size;
  std::optional<AutoNumber> learning_rate_multiplier;
  std::optional<AutoInteger> n_epochs;
};

struct FineTuningJobMethod {
  FineTuningMethod::Type type = FineTuningMethod::Type::Supervised;
  std::optional<DpoMethodConfig> dpo;
  std::optional<ReinforcementMethodConfig> reinforcement;
  std::optional<SupervisedMethodConfig> supervised;
};

struct FineTuningJob {
  std::string id;
  int created_at = 0;
  std::optional<FineTuningJobError> error;
  std::optional<std::string> fine_tuned_model;
  std::optional<int> finished_at;
  std::optional<FineTuningJobHyperparameters> hyperparameters;
  std::string model;
  std::string object;
  std::string organization_id;
  std::vector<std::string> result_files;
  int seed = 0;
  std::string status;
  std::optional<int> trained_tokens;
  std::string training_file;
  std::optional<std::string> validation_file;
  std::optional<int> estimated_finish;
  std::vector<FineTuningJobIntegration> integrations;
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<FineTuningJobMethod> method;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningJobEvent {
  std::string id;
  int created_at = 0;
  std::string level;
  std::string message;
  std::string object;
  std::optional<nlohmann::json> data;
  std::optional<std::string> type;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningJobList {
  std::vector<FineTuningJob> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningJobEventsList {
  std::vector<FineTuningJobEvent> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct GraderRunMetadataErrors {
  bool formula_parse_error = false;
  bool invalid_variable_error = false;
  bool model_grader_parse_error = false;
  bool model_grader_refusal_error = false;
  bool model_grader_server_error = false;
  std::optional<std::string> model_grader_server_error_details;
  bool other_error = false;
  bool python_grader_runtime_error = false;
  std::optional<std::string> python_grader_runtime_error_details;
  bool python_grader_server_error = false;
  std::optional<std::string> python_grader_server_error_type;
  bool sample_parse_error = false;
  bool truncated_observation_error = false;
  bool unresponsive_reward_error = false;
};

struct GraderRunMetadata {
  GraderRunMetadataErrors errors;
  double execution_time = 0.0;
  std::string name;
  std::optional<std::string> sampled_model_name;
  std::map<std::string, nlohmann::json> scores;
  std::optional<double> token_usage;
  std::string type;
};

struct GraderRunResponse {
  GraderRunMetadata metadata;
  std::map<std::string, nlohmann::json> model_grader_token_usage_per_model;
  double reward = 0.0;
  std::map<std::string, nlohmann::json> sub_rewards;
  nlohmann::json raw = nlohmann::json::object();
};

struct GraderValidateResponse {
  std::optional<std::variant<graders::StringCheckGrader,
                             graders::TextSimilarityGrader,
                             graders::PythonGrader,
                             graders::ScoreModelGrader,
                             graders::MultiGrader,
                             graders::LabelModelGrader>> grader;
  nlohmann::json raw = nlohmann::json::object();
};

struct GraderRunParams {
  std::variant<graders::StringCheckGrader,
               graders::TextSimilarityGrader,
               graders::PythonGrader,
               graders::ScoreModelGrader,
               graders::MultiGrader> grader;
  std::string model_sample;
  std::optional<nlohmann::json> item;
};

struct GraderValidateParams {
  std::variant<graders::StringCheckGrader,
               graders::TextSimilarityGrader,
               graders::PythonGrader,
               graders::ScoreModelGrader,
               graders::MultiGrader,
               graders::LabelModelGrader> grader;
};

class FineTuningAlphaGradersResource {
public:
  explicit FineTuningAlphaGradersResource(OpenAIClient& client) : client_(client) {}

  GraderRunResponse run(const GraderRunParams& params) const;
  GraderRunResponse run(const GraderRunParams& params, const RequestOptions& options) const;

  GraderValidateResponse validate(const GraderValidateParams& params) const;
  GraderValidateResponse validate(const GraderValidateParams& params, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class FineTuningAlphaResource {
public:
  explicit FineTuningAlphaResource(OpenAIClient& client) : graders_(client) {}

  FineTuningAlphaGradersResource& graders() { return graders_; }
  const FineTuningAlphaGradersResource& graders() const { return graders_; }

private:
  FineTuningAlphaGradersResource graders_;
};

struct FineTuningJobCheckpointMetrics {
  std::optional<double> full_valid_loss;
  std::optional<double> full_valid_mean_token_accuracy;
  std::optional<int> step;
  std::optional<double> train_loss;
  std::optional<double> train_mean_token_accuracy;
  std::optional<double> valid_loss;
  std::optional<double> valid_mean_token_accuracy;
};

struct FineTuningJobCheckpoint {
  std::string id;
  int created_at = 0;
  std::string fine_tuned_model_checkpoint;
  std::string fine_tuning_job_id;
  FineTuningJobCheckpointMetrics metrics;
  std::string object;
  int step_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningJobCheckpointList {
  std::vector<FineTuningJobCheckpoint> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningCheckpointListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
};

struct FineTuningCheckpointPermission {
  std::string id;
  int created_at = 0;
  std::string object;
  std::string project_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningCheckpointPermissionList {
  std::vector<FineTuningCheckpointPermission> data;
  bool has_more = false;
  std::string object;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct FineTuningCheckpointPermissionCreateParams {
  std::vector<std::string> project_ids;
};

struct FineTuningCheckpointPermissionRetrieveParams {
  std::optional<std::string> after;
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> project_id;
};

struct FineTuningCheckpointPermissionDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

class FineTuningJobCheckpointPermissionsResource {
public:
  explicit FineTuningJobCheckpointPermissionsResource(OpenAIClient& client) : client_(client) {}

  FineTuningCheckpointPermissionList create(const std::string& checkpoint_id,
                                            const FineTuningCheckpointPermissionCreateParams& params) const;
  FineTuningCheckpointPermissionList create(const std::string& checkpoint_id,
                                            const FineTuningCheckpointPermissionCreateParams& params,
                                            const RequestOptions& options) const;

  FineTuningCheckpointPermissionList retrieve(const std::string& checkpoint_id,
                                              const FineTuningCheckpointPermissionRetrieveParams& params) const;
  FineTuningCheckpointPermissionList retrieve(const std::string& checkpoint_id,
                                              const FineTuningCheckpointPermissionRetrieveParams& params,
                                              const RequestOptions& options) const;
  FineTuningCheckpointPermissionList retrieve(const std::string& checkpoint_id) const;
  FineTuningCheckpointPermissionList retrieve(const std::string& checkpoint_id,
                                              const RequestOptions& options) const;

  FineTuningCheckpointPermissionDeleteResponse remove(const std::string& checkpoint_id,
                                                      const std::string& permission_id) const;
  FineTuningCheckpointPermissionDeleteResponse remove(const std::string& checkpoint_id,
                                                      const std::string& permission_id,
                                                      const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class FineTuningJobCheckpointsResource {
public:
  explicit FineTuningJobCheckpointsResource(OpenAIClient& client)
      : client_(client), permissions_(client) {}

  FineTuningJobCheckpointList list(const std::string& job_id,
                                   const FineTuningCheckpointListParams& params) const;
  FineTuningJobCheckpointList list(const std::string& job_id,
                                   const FineTuningCheckpointListParams& params,
                                   const RequestOptions& options) const;
  FineTuningJobCheckpointList list(const std::string& job_id) const;
  FineTuningJobCheckpointList list(const std::string& job_id, const RequestOptions& options) const;

  FineTuningJobCheckpointPermissionsResource& permissions() { return permissions_; }
  const FineTuningJobCheckpointPermissionsResource& permissions() const { return permissions_; }

private:
  OpenAIClient& client_;
  FineTuningJobCheckpointPermissionsResource permissions_;
};

class FineTuningJobsResource {
public:
  explicit FineTuningJobsResource(OpenAIClient& client) : client_(client), checkpoints_(client) {}

  FineTuningJob create(const JobCreateParams& request) const;
  FineTuningJob create(const JobCreateParams& request, const RequestOptions& options) const;

  FineTuningJob retrieve(const std::string& job_id) const;
  FineTuningJob retrieve(const std::string& job_id, const RequestOptions& options) const;

  FineTuningJobList list(const JobListParams& params) const;
  FineTuningJobList list(const JobListParams& params, const RequestOptions& options) const;
  FineTuningJobList list() const;
  FineTuningJobList list(const RequestOptions& options) const;

  CursorPage<FineTuningJob> list_page(const JobListParams& params) const;
  CursorPage<FineTuningJob> list_page(const JobListParams& params, const RequestOptions& options) const;
  CursorPage<FineTuningJob> list_page() const;
  CursorPage<FineTuningJob> list_page(const RequestOptions& options) const;

  FineTuningJob cancel(const std::string& job_id) const;
  FineTuningJob cancel(const std::string& job_id, const RequestOptions& options) const;

  FineTuningJob pause(const std::string& job_id) const;
  FineTuningJob pause(const std::string& job_id, const RequestOptions& options) const;

  FineTuningJob resume(const std::string& job_id) const;
  FineTuningJob resume(const std::string& job_id, const RequestOptions& options) const;

  FineTuningJobEventsList list_events(const std::string& job_id, const JobListEventsParams& params) const;
  FineTuningJobEventsList list_events(const std::string& job_id,
                                      const JobListEventsParams& params,
                                      const RequestOptions& options) const;
  FineTuningJobEventsList list_events(const std::string& job_id) const;
  FineTuningJobEventsList list_events(const std::string& job_id, const RequestOptions& options) const;

  CursorPage<FineTuningJobEvent> list_events_page(const std::string& job_id,
                                                  const JobListEventsParams& params) const;
  CursorPage<FineTuningJobEvent> list_events_page(const std::string& job_id,
                                                  const JobListEventsParams& params,
                                                  const RequestOptions& options) const;
  CursorPage<FineTuningJobEvent> list_events_page(const std::string& job_id) const;

  FineTuningJobCheckpointsResource& checkpoints() { return checkpoints_; }
  const FineTuningJobCheckpointsResource& checkpoints() const { return checkpoints_; }

private:
  OpenAIClient& client_;
  FineTuningJobCheckpointsResource checkpoints_;
};

class FineTuningResource {
public:
  explicit FineTuningResource(OpenAIClient& client) : client_(client), jobs_(client), alpha_(client) {}

  FineTuningJobsResource& jobs() { return jobs_; }
  const FineTuningJobsResource& jobs() const { return jobs_; }

  FineTuningAlphaResource& alpha() { return alpha_; }
  const FineTuningAlphaResource& alpha() const { return alpha_; }

private:
  OpenAIClient& client_;
  FineTuningJobsResource jobs_;
  FineTuningAlphaResource alpha_;
};

}  // namespace openai

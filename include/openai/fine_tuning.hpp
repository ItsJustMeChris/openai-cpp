#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

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

class FineTuningJobCheckpointsResource {
public:
  explicit FineTuningJobCheckpointsResource(OpenAIClient& client) : client_(client) {}

  FineTuningJobCheckpointList list(const std::string& job_id,
                                   const FineTuningCheckpointListParams& params) const;
  FineTuningJobCheckpointList list(const std::string& job_id,
                                   const FineTuningCheckpointListParams& params,
                                   const RequestOptions& options) const;
  FineTuningJobCheckpointList list(const std::string& job_id) const;
  FineTuningJobCheckpointList list(const std::string& job_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
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
  explicit FineTuningResource(OpenAIClient& client) : client_(client), jobs_(client) {}

  FineTuningJobsResource& jobs() { return jobs_; }
  const FineTuningJobsResource& jobs() const { return jobs_; }

private:
  OpenAIClient& client_;
  FineTuningJobsResource jobs_;
};

}  // namespace openai

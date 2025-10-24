#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/graders.hpp"

namespace openai {

struct RequestOptions;
class OpenAIClient;

template <typename Item>
class CursorPage;

namespace evals {

using Metadata = std::map<std::string, std::string>;

struct CustomDataSourceConfig {
  nlohmann::json schema;
  std::string type = "custom";
  nlohmann::json raw;
};

struct LogsDataSourceConfig {
  nlohmann::json schema;
  std::optional<Metadata> metadata;
  std::string type = "logs";
  nlohmann::json raw;
};

struct StoredCompletionsDataSourceConfig {
  nlohmann::json schema;
  std::optional<Metadata> metadata;
  std::string type = "stored_completions";
  nlohmann::json raw;
};

using DataSourceConfig =
    std::variant<CustomDataSourceConfig, LogsDataSourceConfig, StoredCompletionsDataSourceConfig>;

struct LabelModelGrader {
  graders::LabelModelGrader grader;
  std::vector<graders::LabelModelGraderInput> input;
  std::vector<std::string> labels;
  std::string model;
  std::string name;
  std::vector<std::string> passing_labels;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct StringCheckGrader {
  graders::StringCheckGrader grader;
  std::string input;
  std::string name;
  std::string operation;
  std::string reference;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct TextSimilarityGrader {
  graders::TextSimilarityGrader grader;
  std::string evaluation_metric;
  std::string input;
  std::string name;
  std::string reference;
  std::string type;
  double pass_threshold = 0.0;
  nlohmann::json raw = nlohmann::json::object();
};

struct PythonGrader {
  graders::PythonGrader grader;
  std::string name;
  std::string source;
  std::string type;
  std::optional<double> pass_threshold;
  nlohmann::json raw = nlohmann::json::object();
};

struct ScoreModelGrader {
  graders::ScoreModelGrader grader;
  std::vector<graders::ScoreModelGraderInput> input;
  std::optional<std::vector<double>> range;
  std::string model;
  std::string name;
  std::string type;
  std::optional<graders::ScoreModelGraderSamplingParams> sampling_params;
  std::optional<double> pass_threshold;
  nlohmann::json raw = nlohmann::json::object();
};

using TestingCriterion =
    std::variant<LabelModelGrader, StringCheckGrader, TextSimilarityGrader, PythonGrader, ScoreModelGrader>;

struct Evaluation {
  std::string id;
  int created_at = 0;
  DataSourceConfig data_source_config;
  std::optional<Metadata> metadata;
  std::string name;
  std::string object;
  std::vector<TestingCriterion> testing_criteria;
  nlohmann::json raw = nlohmann::json::object();
};

struct EvaluationList {
  std::vector<Evaluation> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct EvaluationDeleteResponse {
  bool deleted = false;
  std::string eval_id;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct CreateCustomDataSourceConfig {
  nlohmann::json item_schema;
  std::string type = "custom";
  std::optional<bool> include_sample_schema;
};

struct CreateLogsDataSourceConfig {
  std::string type = "logs";
  std::optional<nlohmann::json> metadata;
};

struct CreateStoredCompletionsDataSourceConfig {
  std::string type = "stored_completions";
  std::optional<nlohmann::json> metadata;
};

using CreateDataSourceConfig =
    std::variant<CreateCustomDataSourceConfig, CreateLogsDataSourceConfig, CreateStoredCompletionsDataSourceConfig>;

struct EvaluationCreateParams {
  CreateDataSourceConfig data_source_config;
  std::vector<TestingCriterion> testing_criteria;
  std::optional<Metadata> metadata;
  std::optional<std::string> name;
};

struct EvaluationUpdateParams {
  std::optional<Metadata> metadata;
  std::optional<std::string> name;
};

struct EvaluationListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> order;
  std::optional<std::string> order_by;
};

struct JSONLContentRow {
  nlohmann::json item = nlohmann::json::object();
  std::optional<nlohmann::json> sample;
  std::optional<std::string> detail;
  std::optional<std::string> image_url;
};

struct RunFileContentSource {
  std::vector<JSONLContentRow> content;
};

struct RunFileIDSource {
  std::string id;
};

struct RunStoredCompletionsSource {
  std::string type = "stored_completions";
  std::optional<int> limit;
  std::optional<int> created_after;
  std::optional<int> created_before;
  std::optional<Metadata> metadata;
  std::optional<std::string> model;
};

struct RunResponsesSource {
  std::string type = "responses";
  std::optional<int> limit;
  std::optional<int> created_after;
  std::optional<int> created_before;
  std::optional<std::string> instructions_search;
  std::optional<nlohmann::json> metadata;
  std::optional<std::string> model;
  std::optional<std::string> reasoning_effort;
  std::optional<double> temperature;
  std::vector<std::string> tools;
  std::optional<double> top_p;
  std::vector<std::string> users;
};

struct RunItemReference {
  std::string item_reference;
};

struct RunTemplate {
  std::vector<nlohmann::json> template_messages;
  std::string type = "template";
  nlohmann::json raw = nlohmann::json::object();
};

struct RunSamplingParams {
  std::optional<int> max_completion_tokens;
  std::optional<std::string> reasoning_effort;
  std::optional<int> seed;
  std::optional<double> temperature;
  std::optional<nlohmann::json> text;
  std::vector<nlohmann::json> tools;
  std::optional<double> top_p;
  std::optional<nlohmann::json> response_format;
  std::optional<std::string> format;
};

struct CreateCompletionsRunDataSource {
  std::variant<RunFileContentSource, RunFileIDSource, RunStoredCompletionsSource> source;
  std::string type = "completions";
  std::optional<std::variant<RunTemplate, RunItemReference>> input_messages;
  std::optional<std::string> model;
  std::optional<RunSamplingParams> sampling_params;
  nlohmann::json raw = nlohmann::json::object();
};

struct CreateJSONLRunDataSource {
  std::variant<RunFileContentSource, RunFileIDSource> source;
  std::string type = "jsonl";
  nlohmann::json raw = nlohmann::json::object();
};

struct CreateResponsesRunDataSource {
  std::variant<RunFileContentSource, RunFileIDSource, RunResponsesSource> source;
  std::string type = "responses";
  std::optional<std::variant<RunTemplate, RunItemReference>> input_messages;
  std::optional<std::string> model;
  std::optional<RunSamplingParams> sampling_params;
  nlohmann::json raw = nlohmann::json::object();
};

using RunDataSource = std::variant<CreateJSONLRunDataSource, CreateCompletionsRunDataSource, CreateResponsesRunDataSource>;

struct RunPerModelUsage {
  std::string model_name;
  int cached_tokens = 0;
  int completion_tokens = 0;
  int invocation_count = 0;
  int prompt_tokens = 0;
  int total_tokens = 0;
};

struct RunPerTestingCriteriaResult {
  int failed = 0;
  int passed = 0;
  std::string testing_criteria;
};

struct RunResultCounts {
  int errored = 0;
  int failed = 0;
  int passed = 0;
  int total = 0;
};

struct RunOutputItem {
  nlohmann::json datasource_item;
  int datasource_item_id = 0;
  std::optional<nlohmann::json> item;
  std::optional<nlohmann::json> sample;
  nlohmann::json raw;
};

struct EvalAPIError {
  std::string code;
  std::string message;
};

struct Run {
  std::string id;
  int created_at = 0;
  RunDataSource data_source;
  EvalAPIError error;
  std::string eval_id;
  std::optional<Metadata> metadata;
  std::string model;
  std::string name;
  std::string object;
  std::vector<RunPerModelUsage> per_model_usage;
  std::vector<RunPerTestingCriteriaResult> per_testing_criteria_results;
  std::string report_url;
  RunResultCounts result_counts;
  std::string status;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunList {
  std::vector<Run> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunDeleteResponse {
  bool deleted = false;
  std::string eval_id;
  std::string object;
  std::string run_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunCancelResponse {
  std::string id;
  std::string object;
  std::string status;
  std::string eval_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunCreateParams {
  RunDataSource data_source;
  std::optional<Metadata> metadata;
  std::optional<std::string> name;
};

struct RunRetrieveParams {
  std::string eval_id;
};

struct RunListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> order;
  std::optional<std::string> status;
};

struct RunDeleteParams {
  std::string eval_id;
};

struct RunCancelParams {
  std::string eval_id;
};

struct OutputItemResult {
  std::string name;
  bool passed = false;
  double score = 0.0;
  std::optional<nlohmann::json> sample;
  std::optional<std::string> type;
};

struct OutputItemSampleUsage {
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  std::optional<nlohmann::json> details;
};

struct OutputItemSampleMessage {
  std::string role;
  std::string content;
};

struct OutputItemSample {
  EvalAPIError error;
  std::string finish_reason;
  std::vector<OutputItemSampleMessage> input;
  int max_completion_tokens = 0;
  std::string model;
  std::vector<OutputItemSampleMessage> output;
  int seed = 0;
  double temperature = 0.0;
  double top_p = 0.0;
  OutputItemSampleUsage usage;
  nlohmann::json raw;
};

struct OutputItem {
  std::string id;
  int created_at = 0;
  nlohmann::json datasource_item;
  int datasource_item_id = 0;
  std::string eval_id;
  std::string object;
  std::vector<OutputItemResult> results;
  std::string run_id;
  OutputItemSample sample;
  std::string status;
  nlohmann::json raw;
};

struct OutputItemList {
  std::vector<OutputItem> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw;
};

struct OutputItemListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> order;
};

}  // namespace evals

class EvalsRunsOutputItemsResource {
public:
  explicit EvalsRunsOutputItemsResource(OpenAIClient& client) : client_(client) {}

  evals::OutputItem retrieve(const std::string& eval_id,
                             const std::string& run_id,
                             const std::string& output_item_id) const;
  evals::OutputItem retrieve(const std::string& eval_id,
                             const std::string& run_id,
                             const std::string& output_item_id,
                             const RequestOptions& options) const;

  evals::OutputItemList list(const std::string& eval_id,
                             const std::string& run_id) const;
  evals::OutputItemList list(const std::string& eval_id,
                             const std::string& run_id,
                             const evals::OutputItemListParams& params) const;
  evals::OutputItemList list(const std::string& eval_id,
                             const std::string& run_id,
                             const RequestOptions& options) const;
  evals::OutputItemList list(const std::string& eval_id,
                             const std::string& run_id,
                             const evals::OutputItemListParams& params,
                             const RequestOptions& options) const;

  CursorPage<evals::OutputItem> list_page(const std::string& eval_id,
                                          const std::string& run_id) const;
  CursorPage<evals::OutputItem> list_page(const std::string& eval_id,
                                          const std::string& run_id,
                                          const evals::OutputItemListParams& params) const;
  CursorPage<evals::OutputItem> list_page(const std::string& eval_id,
                                          const std::string& run_id,
                                          const RequestOptions& options) const;
  CursorPage<evals::OutputItem> list_page(const std::string& eval_id,
                                          const std::string& run_id,
                                          const evals::OutputItemListParams& params,
                                          const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class EvalsRunsResource {
public:
  explicit EvalsRunsResource(OpenAIClient& client) : client_(client), output_items_(client) {}

  evals::Run create(const std::string& eval_id,
                    const evals::RunCreateParams& params) const;
  evals::Run create(const std::string& eval_id,
                    const evals::RunCreateParams& params,
                    const RequestOptions& options) const;

  evals::Run retrieve(const std::string& run_id,
                      const evals::RunRetrieveParams& params) const;
  evals::Run retrieve(const std::string& run_id,
                      const evals::RunRetrieveParams& params,
                      const RequestOptions& options) const;

  evals::RunList list(const std::string& eval_id) const;
  evals::RunList list(const std::string& eval_id, const evals::RunListParams& params) const;
  evals::RunList list(const std::string& eval_id, const RequestOptions& options) const;
  evals::RunList list(const std::string& eval_id,
                      const evals::RunListParams& params,
                      const RequestOptions& options) const;

  CursorPage<evals::Run> list_page(const std::string& eval_id) const;
  CursorPage<evals::Run> list_page(const std::string& eval_id, const evals::RunListParams& params) const;
  CursorPage<evals::Run> list_page(const std::string& eval_id, const RequestOptions& options) const;
  CursorPage<evals::Run> list_page(const std::string& eval_id,
                                   const evals::RunListParams& params,
                                   const RequestOptions& options) const;

  evals::RunDeleteResponse remove(const std::string& run_id,
                                  const evals::RunDeleteParams& params) const;
  evals::RunDeleteResponse remove(const std::string& run_id,
                                  const evals::RunDeleteParams& params,
                                  const RequestOptions& options) const;

  evals::RunCancelResponse cancel(const std::string& run_id,
                                  const evals::RunCancelParams& params) const;
  evals::RunCancelResponse cancel(const std::string& run_id,
                                  const evals::RunCancelParams& params,
                                  const RequestOptions& options) const;

  EvalsRunsOutputItemsResource& output_items() { return output_items_; }
  const EvalsRunsOutputItemsResource& output_items() const { return output_items_; }

private:
  OpenAIClient& client_;
  EvalsRunsOutputItemsResource output_items_;
};

class EvalsResource {
public:
  explicit EvalsResource(OpenAIClient& client) : client_(client), runs_(client) {}

  evals::Evaluation create(const evals::EvaluationCreateParams& params) const;
  evals::Evaluation create(const evals::EvaluationCreateParams& params, const RequestOptions& options) const;

  evals::Evaluation retrieve(const std::string& eval_id) const;
  evals::Evaluation retrieve(const std::string& eval_id, const RequestOptions& options) const;

  evals::Evaluation update(const std::string& eval_id,
                           const evals::EvaluationUpdateParams& params) const;
  evals::Evaluation update(const std::string& eval_id,
                           const evals::EvaluationUpdateParams& params,
                           const RequestOptions& options) const;

  evals::EvaluationDeleteResponse remove(const std::string& eval_id) const;
  evals::EvaluationDeleteResponse remove(const std::string& eval_id, const RequestOptions& options) const;

  evals::EvaluationList list() const;
  evals::EvaluationList list(const evals::EvaluationListParams& params) const;
  evals::EvaluationList list(const RequestOptions& options) const;
  evals::EvaluationList list(const evals::EvaluationListParams& params, const RequestOptions& options) const;

  CursorPage<evals::Evaluation> list_page() const;
  CursorPage<evals::Evaluation> list_page(const evals::EvaluationListParams& params) const;
  CursorPage<evals::Evaluation> list_page(const RequestOptions& options) const;
  CursorPage<evals::Evaluation> list_page(const evals::EvaluationListParams& params,
                                          const RequestOptions& options) const;

  EvalsRunsResource& runs() { return runs_; }
  const EvalsRunsResource& runs() const { return runs_; }

private:
  OpenAIClient& client_;
  EvalsRunsResource runs_;
};

}  // namespace openai

#include "openai/evals.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <utility>
#include <variant>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kEvalsPath = "/evals";
constexpr const char* kRunsSuffix = "/runs";
constexpr const char* kOutputItemsSuffix = "/output_items";

json metadata_to_json(const evals::Metadata& metadata) {
  json value = json::object();
  for (const auto& entry : metadata) {
    value[entry.first] = entry.second;
  }
  return value;
}

std::optional<evals::Metadata> parse_metadata(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  evals::Metadata metadata;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    if (it.value().is_string()) {
      metadata[it.key()] = it.value().get<std::string>();
    }
  }
  return metadata;
}

json json_from_tools(const std::vector<nlohmann::json>& tools) {
  json array = json::array();
  for (const auto& tool : tools) {
    array.push_back(tool);
  }
  return array;
}
std::vector<evals::JSONLContentRow> parse_jsonl_rows(const json& payload) {
  std::vector<evals::JSONLContentRow> rows;
  if (!payload.is_array()) return rows;
  for (const auto& entry : payload) {
    evals::JSONLContentRow row;
    if (entry.contains("item")) {
      row.item = entry.at("item");
    }
    if (entry.contains("sample") && !entry.at("sample").is_null()) {
      row.sample = entry.at("sample");
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

graders::LabelModelGraderInput parse_label_input(const json& payload) {
  graders::LabelModelGraderInput input;
  input.role = payload.value("role", "");
  if (payload.contains("content")) {
    input.content = payload.at("content");
  }
  if (payload.contains("type") && payload.at("type").is_string()) {
    input.type = payload.at("type").get<std::string>();
  }
  return input;
}

json label_input_to_json(const graders::LabelModelGraderInput& input) {
  json value;
  value["role"] = input.role;
  value["content"] = input.content;
  if (input.type) value["type"] = *input.type;
  return value;
}

graders::LabelModelGrader parse_label_model(const json& payload) {
  graders::LabelModelGrader grader;
  if (payload.contains("input") && payload.at("input").is_array()) {
    for (const auto& entry : payload.at("input")) {
      grader.input.push_back(parse_label_input(entry));
    }
  }
  if (payload.contains("labels") && payload.at("labels").is_array()) {
    for (const auto& item : payload.at("labels")) {
      if (item.is_string()) grader.labels.push_back(item.get<std::string>());
    }
  }
  grader.model = payload.value("model", "");
  grader.name = payload.value("name", "");
  if (payload.contains("passing_labels") && payload.at("passing_labels").is_array()) {
    for (const auto& item : payload.at("passing_labels")) {
      if (item.is_string()) grader.passing_labels.push_back(item.get<std::string>());
    }
  }
  grader.type = payload.value("type", "");
  return grader;
}

json label_model_to_json(const graders::LabelModelGrader& grader) {
  json value;
  json input = json::array();
  for (const auto& entry : grader.input) {
    input.push_back(label_input_to_json(entry));
  }
  value["input"] = std::move(input);
  value["labels"] = grader.labels;
  value["model"] = grader.model;
  value["name"] = grader.name;
  value["passing_labels"] = grader.passing_labels;
  value["type"] = grader.type;
  return value;
}

graders::StringCheckGrader parse_string_check(const json& payload) {
  graders::StringCheckGrader grader;
  grader.input = payload.value("input", "");
  grader.name = payload.value("name", "");
  grader.operation = payload.value("operation", "");
  grader.reference = payload.value("reference", "");
  grader.type = payload.value("type", "");
  return grader;
}

json string_check_to_json(const graders::StringCheckGrader& grader) {
  json value;
  value["input"] = grader.input;
  value["name"] = grader.name;
  value["operation"] = grader.operation;
  value["reference"] = grader.reference;
  value["type"] = grader.type;
  return value;
}

graders::TextSimilarityGrader parse_text_similarity_base(const json& payload) {
  graders::TextSimilarityGrader grader;
  grader.evaluation_metric = payload.value("evaluation_metric", "");
  grader.input = payload.value("input", "");
  grader.name = payload.value("name", "");
  grader.reference = payload.value("reference", "");
  grader.type = payload.value("type", "");
  return grader;
}

json text_similarity_base_to_json(const graders::TextSimilarityGrader& grader) {
  json value;
  value["evaluation_metric"] = grader.evaluation_metric;
  value["input"] = grader.input;
  value["name"] = grader.name;
  value["reference"] = grader.reference;
  value["type"] = grader.type;
  return value;
}

graders::PythonGrader parse_python_grader_base(const json& payload) {
  graders::PythonGrader grader;
  grader.name = payload.value("name", "");
  grader.source = payload.value("source", "");
  grader.type = payload.value("type", "");
  if (payload.contains("image_tag") && payload.at("image_tag").is_string()) {
    grader.image_tag = payload.at("image_tag").get<std::string>();
  }
  return grader;
}

json python_grader_base_to_json(const graders::PythonGrader& grader) {
  json value;
  value["name"] = grader.name;
  value["source"] = grader.source;
  value["type"] = grader.type;
  if (grader.image_tag) value["image_tag"] = *grader.image_tag;
  return value;
}

graders::ScoreModelGraderSamplingParams parse_score_sampling_params(const json& payload) {
  graders::ScoreModelGraderSamplingParams params;
  if (payload.contains("max_tokens") && payload.at("max_tokens").is_number_integer()) {
    params.max_tokens = payload.at("max_tokens").get<int>();
  }
  if (payload.contains("temperature") && payload.at("temperature").is_number()) {
    params.temperature = payload.at("temperature").get<double>();
  }
  if (payload.contains("top_p") && payload.at("top_p").is_number()) {
    params.top_p = payload.at("top_p").get<double>();
  }
  return params;
}

json score_sampling_params_to_json(const graders::ScoreModelGraderSamplingParams& params) {
  json value;
  if (params.max_tokens) value["max_tokens"] = *params.max_tokens;
  if (params.temperature) value["temperature"] = *params.temperature;
  if (params.top_p) value["top_p"] = *params.top_p;
  return value;
}

graders::ScoreModelGraderInput parse_score_model_input(const json& payload) {
  graders::ScoreModelGraderInput input;
  input.role = payload.value("role", "");
  if (payload.contains("content")) {
    input.content = payload.at("content");
  }
  if (payload.contains("type") && payload.at("type").is_string()) {
    input.type = payload.at("type").get<std::string>();
  }
  return input;
}

json score_model_input_to_json(const graders::ScoreModelGraderInput& input) {
  json value;
  value["role"] = input.role;
  value["content"] = input.content;
  if (input.type) value["type"] = *input.type;
  return value;
}

graders::ScoreModelGrader parse_score_model_base(const json& payload) {
  graders::ScoreModelGrader grader;
  if (payload.contains("input") && payload.at("input").is_array()) {
    for (const auto& entry : payload.at("input")) {
      grader.input.push_back(parse_score_model_input(entry));
    }
  }
  grader.model = payload.value("model", "");
  grader.name = payload.value("name", "");
  grader.type = payload.value("type", "");
  if (payload.contains("range") && payload.at("range").is_array()) {
    grader.range = std::vector<double>();
    for (const auto& item : payload.at("range")) {
      if (item.is_number()) grader.range->push_back(item.get<double>());
    }
  }
  if (payload.contains("sampling_params") && payload.at("sampling_params").is_object()) {
    grader.sampling_params = parse_score_sampling_params(payload.at("sampling_params"));
  }
  return grader;
}

json score_model_to_json(const graders::ScoreModelGrader& grader) {
  json value;
  json input = json::array();
  for (const auto& entry : grader.input) {
    input.push_back(score_model_input_to_json(entry));
  }
  value["input"] = std::move(input);
  value["model"] = grader.model;
  value["name"] = grader.name;
  value["type"] = grader.type;
  if (grader.range) value["range"] = *grader.range;
  if (grader.sampling_params) value["sampling_params"] = score_sampling_params_to_json(*grader.sampling_params);
  return value;
}

evals::LabelModelGrader parse_label_model_grader(const json& payload) {
  evals::LabelModelGrader result;
  result.raw = payload;
  result.grader = parse_label_model(payload);
  return result;
}

evals::StringCheckGrader parse_string_check_grader(const json& payload) {
  evals::StringCheckGrader grader;
  grader.raw = payload;
  grader.grader = parse_string_check(payload);
  return grader;
}

evals::TextSimilarityGrader parse_text_similarity_grader(const json& payload) {
  evals::TextSimilarityGrader grader;
  grader.raw = payload;
  grader.grader = parse_text_similarity_base(payload);
  if (payload.contains("pass_threshold") && payload.at("pass_threshold").is_number()) {
    grader.pass_threshold = payload.at("pass_threshold").get<double>();
  }
  return grader;
}

evals::PythonGrader parse_python_grader(const json& payload) {
  evals::PythonGrader grader;
  grader.raw = payload;
  grader.grader = parse_python_grader_base(payload);
  if (payload.contains("pass_threshold") && payload.at("pass_threshold").is_number()) {
    grader.pass_threshold = payload.at("pass_threshold").get<double>();
  }
  return grader;
}

evals::ScoreModelGrader parse_score_model_grader(const json& payload) {
  evals::ScoreModelGrader grader;
  grader.raw = payload;
  grader.grader = parse_score_model_base(payload);
  if (payload.contains("pass_threshold") && payload.at("pass_threshold").is_number()) {
    grader.pass_threshold = payload.at("pass_threshold").get<double>();
  }
  return grader;
}

evals::TestingCriterion parse_testing_criterion(const json& payload) {
  const std::string type = payload.value("type", "");
  if (type == "label_model") {
    return parse_label_model_grader(payload);
  }
  if (type == "string_check") {
    return parse_string_check_grader(payload);
  }
  if (type == "text_similarity") {
    return parse_text_similarity_grader(payload);
  }
  if (type == "python") {
    return parse_python_grader(payload);
  }
  if (type == "score_model") {
    return parse_score_model_grader(payload);
  }
  return parse_string_check_grader(payload);
}

json testing_criterion_to_json(const evals::TestingCriterion& criterion) {
  return std::visit(
      [](const auto& value) -> json {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, evals::LabelModelGrader>) {
          json payload = label_model_to_json(value.grader);
          return payload;
        } else if constexpr (std::is_same_v<T, evals::StringCheckGrader>) {
          json payload = string_check_to_json(value.grader);
          return payload;
        } else if constexpr (std::is_same_v<T, evals::TextSimilarityGrader>) {
          json payload = text_similarity_base_to_json(value.grader);
          payload["pass_threshold"] = value.pass_threshold;
          return payload;
        } else if constexpr (std::is_same_v<T, evals::PythonGrader>) {
          json payload = python_grader_base_to_json(value.grader);
          if (value.pass_threshold) payload["pass_threshold"] = *value.pass_threshold;
          return payload;
        } else if constexpr (std::is_same_v<T, evals::ScoreModelGrader>) {
          json payload = score_model_to_json(value.grader);
          if (value.pass_threshold) payload["pass_threshold"] = *value.pass_threshold;
          return payload;
        }
        return json::object();
      },
      criterion);
}

evals::CustomDataSourceConfig parse_custom_config(const json& payload) {
  evals::CustomDataSourceConfig config;
  config.raw = payload;
  if (payload.contains("schema")) {
    config.schema = payload.at("schema");
  }
  config.type = payload.value("type", "custom");
  return config;
}

evals::LogsDataSourceConfig parse_logs_config(const json& payload) {
  evals::LogsDataSourceConfig config;
  config.raw = payload;
  if (payload.contains("schema")) {
    config.schema = payload.at("schema");
  }
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    config.metadata = parse_metadata(payload.at("metadata"));
  }
  config.type = payload.value("type", "logs");
  return config;
}

evals::StoredCompletionsDataSourceConfig parse_stored_config(const json& payload) {
  evals::StoredCompletionsDataSourceConfig config;
  config.raw = payload;
  if (payload.contains("schema")) {
    config.schema = payload.at("schema");
  }
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    config.metadata = parse_metadata(payload.at("metadata"));
  }
  config.type = payload.value("type", "stored_completions");
  return config;
}

evals::DataSourceConfig parse_data_source_config(const json& payload) {
  const std::string type = payload.value("type", "");
  if (type == "custom") return parse_custom_config(payload);
  if (type == "logs") return parse_logs_config(payload);
  if (type == "stored_completions") return parse_stored_config(payload);
  return parse_custom_config(payload);
}

evals::Evaluation parse_evaluation(const json& payload) {
  evals::Evaluation evaluation;
  evaluation.raw = payload;
  evaluation.id = payload.value("id", "");
  evaluation.created_at = payload.value("created_at", 0);
  if (payload.contains("data_source_config") && payload.at("data_source_config").is_object()) {
    evaluation.data_source_config = parse_data_source_config(payload.at("data_source_config"));
  } else {
    evaluation.data_source_config = evals::CustomDataSourceConfig{};
  }
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    evaluation.metadata = parse_metadata(payload.at("metadata"));
  } else if (payload.contains("metadata") && payload.at("metadata").is_null()) {
    evaluation.metadata = std::nullopt;
  }
  evaluation.name = payload.value("name", "");
  evaluation.object = payload.value("object", "");
  if (payload.contains("testing_criteria") && payload.at("testing_criteria").is_array()) {
    for (const auto& item : payload.at("testing_criteria")) {
      evaluation.testing_criteria.push_back(parse_testing_criterion(item));
    }
  }
  return evaluation;
}

evals::EvaluationList parse_evaluation_list(const json& payload) {
  evals::EvaluationList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_evaluation(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("first_id") && payload.at("first_id").is_string()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  return list;
}

evals::EvaluationDeleteResponse parse_delete_response(const json& payload) {
  evals::EvaluationDeleteResponse response;
  response.raw = payload;
  response.deleted = payload.value("deleted", false);
  response.eval_id = payload.value("eval_id", "");
  response.object = payload.value("object", "");
  return response;
}

json create_data_source_config_to_json(const evals::CreateDataSourceConfig& config) {
  return std::visit(
      [](const auto& value) -> json {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, evals::CreateCustomDataSourceConfig>) {
          json payload;
          payload["type"] = value.type;
          payload["item_schema"] = value.item_schema;
          if (value.include_sample_schema) payload["include_sample_schema"] = *value.include_sample_schema;
          return payload;
        } else if constexpr (std::is_same_v<T, evals::CreateLogsDataSourceConfig>) {
          json payload;
          payload["type"] = value.type;
          if (value.metadata) payload["metadata"] = *value.metadata;
          return payload;
        } else if constexpr (std::is_same_v<T, evals::CreateStoredCompletionsDataSourceConfig>) {
          json payload;
          payload["type"] = value.type;
          if (value.metadata) payload["metadata"] = *value.metadata;
          return payload;
        }
        return json::object();
      },
      config);
}

json evaluation_create_body(const evals::EvaluationCreateParams& params) {
  json body;
  body["data_source_config"] = create_data_source_config_to_json(params.data_source_config);
  json criteria = json::array();
  for (const auto& criterion : params.testing_criteria) {
    criteria.push_back(testing_criterion_to_json(criterion));
  }
  body["testing_criteria"] = std::move(criteria);
  if (params.metadata && !params.metadata->empty()) {
    body["metadata"] = metadata_to_json(*params.metadata);
  }
  if (params.name) body["name"] = *params.name;
  return body;
}

void apply_evaluation_list_params(const evals::EvaluationListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) options.query_params["after"] = *params.after;
  if (params.order) options.query_params["order"] = *params.order;
  if (params.order_by) options.query_params["order_by"] = *params.order_by;
}

evals::RunFileContentSource parse_file_content_source(const json& payload) {
  evals::RunFileContentSource source;
  if (payload.contains("content")) {
    source.content = parse_jsonl_rows(payload.at("content"));
  }
  return source;
}

evals::RunFileIDSource parse_file_id_source(const json& payload) {
  evals::RunFileIDSource source;
  source.id = payload.value("id", "");
  return source;
}

evals::RunStoredCompletionsSource parse_stored_completions_source(const json& payload) {
  evals::RunStoredCompletionsSource source;
  if (payload.contains("limit") && payload.at("limit").is_number_integer()) {
    source.limit = payload.at("limit").get<int>();
  }
  if (payload.contains("created_after") && payload.at("created_after").is_number_integer()) {
    source.created_after = payload.at("created_after").get<int>();
  }
  if (payload.contains("created_before") && payload.at("created_before").is_number_integer()) {
    source.created_before = payload.at("created_before").get<int>();
  }
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    source.metadata = parse_metadata(payload.at("metadata"));
  }
  if (payload.contains("model") && payload.at("model").is_string()) {
    source.model = payload.at("model").get<std::string>();
  }
  return source;
}

evals::RunResponsesSource parse_responses_source(const json& payload) {
  evals::RunResponsesSource source;
  if (payload.contains("limit") && payload.at("limit").is_number_integer()) {
    source.limit = payload.at("limit").get<int>();
  }
  if (payload.contains("created_after") && payload.at("created_after").is_number_integer()) {
    source.created_after = payload.at("created_after").get<int>();
  }
  if (payload.contains("created_before") && payload.at("created_before").is_number_integer()) {
    source.created_before = payload.at("created_before").get<int>();
  }
  if (payload.contains("instructions_search") && payload.at("instructions_search").is_string()) {
    source.instructions_search = payload.at("instructions_search").get<std::string>();
  }
  if (payload.contains("metadata") && !payload.at("metadata").is_null()) {
    source.metadata = payload.at("metadata");
  }
  if (payload.contains("model") && payload.at("model").is_string()) {
    source.model = payload.at("model").get<std::string>();
  }
  if (payload.contains("reasoning_effort") && payload.at("reasoning_effort").is_string()) {
    source.reasoning_effort = payload.at("reasoning_effort").get<std::string>();
  }
  if (payload.contains("temperature") && payload.at("temperature").is_number()) {
    source.temperature = payload.at("temperature").get<double>();
  }
  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool : payload.at("tools")) {
      if (tool.is_string()) source.tools.push_back(tool.get<std::string>());
    }
  }
  if (payload.contains("top_p") && payload.at("top_p").is_number()) {
    source.top_p = payload.at("top_p").get<double>();
  }
  if (payload.contains("users") && payload.at("users").is_array()) {
    for (const auto& user : payload.at("users")) {
      if (user.is_string()) source.users.push_back(user.get<std::string>());
    }
  }
  return source;
}

evals::RunTemplate parse_template(const json& payload) {
  evals::RunTemplate templ;
  if (payload.contains("template")) {
    templ.entries = payload.at("template");
  }
  return templ;
}

evals::RunItemReference parse_item_reference(const json& payload) {
  evals::RunItemReference reference;
  reference.item_reference = payload.value("item_reference", "");
  return reference;
}

evals::RunSamplingParams parse_sampling_params(const json& payload) {
  evals::RunSamplingParams params;
  if (payload.contains("max_completion_tokens") && payload.at("max_completion_tokens").is_number_integer()) {
    params.max_completion_tokens = payload.at("max_completion_tokens").get<int>();
  }
  if (payload.contains("reasoning_effort") && payload.at("reasoning_effort").is_string()) {
    params.reasoning_effort = payload.at("reasoning_effort").get<std::string>();
  }
  if (payload.contains("seed") && payload.at("seed").is_number_integer()) {
    params.seed = payload.at("seed").get<int>();
  }
  if (payload.contains("temperature") && payload.at("temperature").is_number()) {
    params.temperature = payload.at("temperature").get<double>();
  }
  if (payload.contains("top_p") && payload.at("top_p").is_number()) {
    params.top_p = payload.at("top_p").get<double>();
  }
  if (payload.contains("text") && payload.at("text").is_object()) {
    params.text = payload.at("text");
  }
  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool : payload.at("tools")) {
      params.tools.push_back(tool);
    }
  }
  if (payload.contains("response_format") && !payload.at("response_format").is_null()) {
    params.response_format = payload.at("response_format");
  }
  return params;
}

evals::RunDataSource parse_run_data_source(const json& payload) {
  const std::string type = payload.value("type", "");
  if (type == "jsonl") {
    evals::CreateJSONLRunDataSource source;
    source.raw = payload;
    if (payload.contains("source") && payload.at("source").is_object()) {
      const auto& src = payload.at("source");
      const std::string source_type = src.value("type", "");
      if (source_type == "file_content") {
        source.source = parse_file_content_source(src);
      } else {
        source.source = parse_file_id_source(src);
      }
    }
    return source;
  }
  if (type == "completions") {
    evals::CreateCompletionsRunDataSource source;
    source.raw = payload;
    if (payload.contains("source") && payload.at("source").is_object()) {
      const auto& src = payload.at("source");
      const std::string source_type = src.value("type", "");
      if (source_type == "file_content") {
        source.source = parse_file_content_source(src);
      } else if (source_type == "file_id") {
        source.source = parse_file_id_source(src);
      } else {
        source.source = parse_stored_completions_source(src);
      }
    }
    if (payload.contains("input_messages") && payload.at("input_messages").is_object()) {
      const auto& input = payload.at("input_messages");
      const std::string input_type = input.value("type", "");
      if (input_type == "template") {
        source.input_messages = parse_template(input);
      } else if (input_type == "item_reference") {
        source.input_messages = parse_item_reference(input);
      }
    }
    if (payload.contains("model") && payload.at("model").is_string()) {
      source.model = payload.at("model").get<std::string>();
    }
    if (payload.contains("sampling_params") && payload.at("sampling_params").is_object()) {
      source.sampling_params = parse_sampling_params(payload.at("sampling_params"));
    }
    return source;
  }
  evals::CreateResponsesRunDataSource source;
  source.raw = payload;
  if (payload.contains("source") && payload.at("source").is_object()) {
    const auto& src = payload.at("source");
    const std::string source_type = src.value("type", "");
    if (source_type == "file_content") {
      source.source = parse_file_content_source(src);
    } else if (source_type == "file_id") {
      source.source = parse_file_id_source(src);
    } else {
      source.source = parse_responses_source(src);
    }
  }
  if (payload.contains("input_messages") && payload.at("input_messages").is_object()) {
    const auto& input = payload.at("input_messages");
    const std::string input_type = input.value("type", "");
    if (input_type == "template") {
      source.input_messages = parse_template(input);
    } else if (input_type == "item_reference") {
      source.input_messages = parse_item_reference(input);
    }
  }
  if (payload.contains("model") && payload.at("model").is_string()) {
    source.model = payload.at("model").get<std::string>();
  }
  if (payload.contains("sampling_params") && payload.at("sampling_params").is_object()) {
    source.sampling_params = parse_sampling_params(payload.at("sampling_params"));
  }
  return source;
}

json run_file_source_to_json(const evals::RunFileContentSource& source) {
  json payload;
  payload["type"] = "file_content";
  json content = json::array();
  for (const auto& row : source.content) {
    json entry;
    entry["item"] = row.item;
    if (row.sample) entry["sample"] = *row.sample;
    content.push_back(std::move(entry));
  }
  payload["content"] = std::move(content);
  return payload;
}

json run_file_source_to_json(const evals::RunFileIDSource& source) {
  json payload;
  payload["type"] = "file_id";
  payload["id"] = source.id;
  return payload;
}

json run_stored_completions_to_json(const evals::RunStoredCompletionsSource& source) {
  json payload;
  payload["type"] = "stored_completions";
  if (source.limit) payload["limit"] = *source.limit;
  if (source.created_after) payload["created_after"] = *source.created_after;
  if (source.created_before) payload["created_before"] = *source.created_before;
  if (source.metadata) payload["metadata"] = metadata_to_json(*source.metadata);
  if (source.model) payload["model"] = *source.model;
  return payload;
}

json run_responses_source_to_json(const evals::RunResponsesSource& source) {
  json payload;
  payload["type"] = "responses";
  if (source.limit) payload["limit"] = *source.limit;
  if (source.created_after) payload["created_after"] = *source.created_after;
  if (source.created_before) payload["created_before"] = *source.created_before;
  if (source.instructions_search) payload["instructions_search"] = *source.instructions_search;
  if (source.metadata) payload["metadata"] = *source.metadata;
  if (source.model) payload["model"] = *source.model;
  if (source.reasoning_effort) payload["reasoning_effort"] = *source.reasoning_effort;
  if (source.temperature) payload["temperature"] = *source.temperature;
  if (!source.tools.empty()) payload["tools"] = source.tools;
  if (source.top_p) payload["top_p"] = *source.top_p;
  if (!source.users.empty()) payload["users"] = source.users;
  return payload;
}

json run_template_to_json(const evals::RunTemplate& templ) {
  json payload;
  payload["type"] = "template";
  payload["template"] = templ.entries;
  return payload;
}

json run_item_reference_to_json(const evals::RunItemReference& reference) {
  json payload;
  payload["type"] = "item_reference";
  payload["item_reference"] = reference.item_reference;
  return payload;
}

json run_sampling_params_to_json(const evals::RunSamplingParams& params) {
  json payload;
  if (params.max_completion_tokens) payload["max_completion_tokens"] = *params.max_completion_tokens;
  if (params.reasoning_effort) payload["reasoning_effort"] = *params.reasoning_effort;
  if (params.seed) payload["seed"] = *params.seed;
  if (params.temperature) payload["temperature"] = *params.temperature;
  if (params.top_p) payload["top_p"] = *params.top_p;
  if (params.text) payload["text"] = *params.text;
  if (!params.tools.empty()) payload["tools"] = json_from_tools(params.tools);
  if (params.response_format) payload["response_format"] = *params.response_format;
  return payload;
}

json run_data_source_to_json(const evals::RunDataSource& source) {
  return std::visit(
      [](const auto& value) -> json {
        using T = std::decay_t<decltype(value)>;
        json payload;
        payload["type"] = value.type;
        if constexpr (std::is_same_v<T, evals::CreateJSONLRunDataSource>) {
          payload["source"] = std::visit([](const auto& src) { return run_file_source_to_json(src); }, value.source);
        } else if constexpr (std::is_same_v<T, evals::CreateCompletionsRunDataSource>) {
          payload["source"] = std::visit(
              [](const auto& src) -> json {
                using S = std::decay_t<decltype(src)>;
                if constexpr (std::is_same_v<S, evals::RunFileContentSource>) {
                  return run_file_source_to_json(src);
                } else if constexpr (std::is_same_v<S, evals::RunFileIDSource>) {
                  return run_file_source_to_json(src);
                } else {
                  return run_stored_completions_to_json(src);
                }
              },
              value.source);
          if (value.input_messages) {
            payload["input_messages"] = std::visit(
                [](const auto& item) -> json {
                  using S = std::decay_t<decltype(item)>;
                  if constexpr (std::is_same_v<S, evals::RunTemplate>) {
                    return run_template_to_json(item);
                  } else {
                    return run_item_reference_to_json(item);
                  }
                },
                *value.input_messages);
          }
          if (value.model) payload["model"] = *value.model;
          if (value.sampling_params) payload["sampling_params"] = run_sampling_params_to_json(*value.sampling_params);
        } else if constexpr (std::is_same_v<T, evals::CreateResponsesRunDataSource>) {
          payload["source"] = std::visit(
              [](const auto& src) -> json {
                using S = std::decay_t<decltype(src)>;
                if constexpr (std::is_same_v<S, evals::RunFileContentSource>) {
                  return run_file_source_to_json(src);
                } else if constexpr (std::is_same_v<S, evals::RunFileIDSource>) {
                  return run_file_source_to_json(src);
                } else {
                  return run_responses_source_to_json(src);
                }
              },
              value.source);
          if (value.input_messages) {
            payload["input_messages"] = std::visit(
                [](const auto& item) -> json {
                  using S = std::decay_t<decltype(item)>;
                  if constexpr (std::is_same_v<S, evals::RunTemplate>) {
                    return run_template_to_json(item);
                  } else {
                    return run_item_reference_to_json(item);
                  }
                },
                *value.input_messages);
          }
          if (value.model) payload["model"] = *value.model;
          if (value.sampling_params) payload["sampling_params"] = run_sampling_params_to_json(*value.sampling_params);
        }
        return payload;
      },
      source);
}

json run_create_body(const evals::RunCreateParams& params) {
  json body;
  body["data_source"] = run_data_source_to_json(params.data_source);
  if (params.metadata && !params.metadata->empty()) {
    body["metadata"] = metadata_to_json(*params.metadata);
  }
  if (params.name) body["name"] = *params.name;
  return body;
}

evals::RunPerModelUsage parse_per_model_usage(const json& payload) {
  evals::RunPerModelUsage usage;
  usage.model_name = payload.value("model_name", "");
  usage.cached_tokens = payload.value("cached_tokens", 0);
  usage.completion_tokens = payload.value("completion_tokens", 0);
  usage.invocation_count = payload.value("invocation_count", 0);
  usage.prompt_tokens = payload.value("prompt_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  return usage;
}

evals::RunPerTestingCriteriaResult parse_per_testing_result(const json& payload) {
  evals::RunPerTestingCriteriaResult result;
  result.failed = payload.value("failed", 0);
  result.passed = payload.value("passed", 0);
  result.testing_criteria = payload.value("testing_criteria", "");
  return result;
}

evals::RunResultCounts parse_result_counts(const json& payload) {
  evals::RunResultCounts counts;
  counts.errored = payload.value("errored", 0);
  counts.failed = payload.value("failed", 0);
  counts.passed = payload.value("passed", 0);
  counts.total = payload.value("total", 0);
  return counts;
}

evals::EvalAPIError parse_eval_error(const json& payload) {
  evals::EvalAPIError error;
  error.code = payload.value("code", "");
  error.message = payload.value("message", "");
  return error;
}

evals::Run parse_run(const json& payload) {
  evals::Run run;
  run.raw = payload;
  run.id = payload.value("id", "");
  run.created_at = payload.value("created_at", 0);
  if (payload.contains("data_source") && payload.at("data_source").is_object()) {
    run.data_source = parse_run_data_source(payload.at("data_source"));
  }
  if (payload.contains("error") && payload.at("error").is_object()) {
    run.error = parse_eval_error(payload.at("error"));
  }
  run.eval_id = payload.value("eval_id", "");
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    run.metadata = parse_metadata(payload.at("metadata"));
  } else if (payload.contains("metadata") && payload.at("metadata").is_null()) {
    run.metadata = std::nullopt;
  }
  run.model = payload.value("model", "");
  run.name = payload.value("name", "");
  run.object = payload.value("object", "");
  if (payload.contains("per_model_usage") && payload.at("per_model_usage").is_array()) {
    for (const auto& item : payload.at("per_model_usage")) {
      run.per_model_usage.push_back(parse_per_model_usage(item));
    }
  }
  if (payload.contains("per_testing_criteria_results") && payload.at("per_testing_criteria_results").is_array()) {
    for (const auto& item : payload.at("per_testing_criteria_results")) {
      run.per_testing_criteria_results.push_back(parse_per_testing_result(item));
    }
  }
  run.report_url = payload.value("report_url", "");
  if (payload.contains("result_counts") && payload.at("result_counts").is_object()) {
    run.result_counts = parse_result_counts(payload.at("result_counts"));
  }
  run.status = payload.value("status", "");
  return run;
}

evals::RunList parse_run_list(const json& payload) {
  evals::RunList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_run(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("first_id") && payload.at("first_id").is_string()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  return list;
}

evals::RunDeleteResponse parse_run_delete_response(const json& payload) {
  evals::RunDeleteResponse response;
  response.raw = payload;
  response.deleted = payload.value("deleted", false);
  response.eval_id = payload.value("eval_id", "");
  response.object = payload.value("object", "");
  response.run_id = payload.value("run_id", "");
  return response;
}

evals::RunCancelResponse parse_run_cancel_response(const json& payload) {
  evals::RunCancelResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.object = payload.value("object", "");
  response.status = payload.value("status", "");
  response.eval_id = payload.value("eval_id", "");
  return response;
}

void apply_run_list_params(const evals::RunListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) options.query_params["after"] = *params.after;
  if (params.order) options.query_params["order"] = *params.order;
  if (params.status) options.query_params["status"] = *params.status;
}

void apply_output_item_list_params(const evals::OutputItemListParams& params, RequestOptions& options) {
  if (params.limit) options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) options.query_params["after"] = *params.after;
  if (params.order) options.query_params["order"] = *params.order;
}

evals::OutputItemResult parse_output_item_result(const json& payload) {
  evals::OutputItemResult result;
  result.name = payload.value("name", "");
  result.passed = payload.value("passed", false);
  if (payload.contains("score") && payload.at("score").is_number()) {
    result.score = payload.at("score").get<double>();
  }
  if (payload.contains("sample") && !payload.at("sample").is_null()) {
    result.sample = payload.at("sample");
  }
  if (payload.contains("type") && payload.at("type").is_string()) {
    result.type = payload.at("type").get<std::string>();
  }
  return result;
}

evals::OutputItemSampleUsage parse_output_item_usage(const json& payload) {
  evals::OutputItemSampleUsage usage;
  usage.input_tokens = payload.value("input_tokens", 0);
  usage.output_tokens = payload.value("output_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  if (payload.contains("details") && !payload.at("details").is_null()) {
    usage.details = payload.at("details");
  }
  return usage;
}

evals::OutputItemSampleMessage parse_output_item_message(const json& payload) {
  evals::OutputItemSampleMessage message;
  message.content = payload.value("content", "");
  message.role = payload.value("role", "");
  return message;
}

evals::OutputItemSample parse_output_item_sample(const json& payload) {
  evals::OutputItemSample sample;
  sample.raw = payload;
  if (payload.contains("error") && payload.at("error").is_object()) {
    sample.error = parse_eval_error(payload.at("error"));
  }
  sample.finish_reason = payload.value("finish_reason", "");
  if (payload.contains("input") && payload.at("input").is_array()) {
    for (const auto& entry : payload.at("input")) {
      sample.input.push_back(parse_output_item_message(entry));
    }
  }
  sample.max_completion_tokens = payload.value("max_completion_tokens", 0);
  sample.model = payload.value("model", "");
  if (payload.contains("output") && payload.at("output").is_array()) {
    for (const auto& entry : payload.at("output")) {
      sample.output.push_back(parse_output_item_message(entry));
    }
  }
  sample.seed = payload.value("seed", 0);
  sample.temperature = payload.value("temperature", 0.0);
  sample.top_p = payload.value("top_p", 0.0);
  if (payload.contains("usage") && payload.at("usage").is_object()) {
    sample.usage = parse_output_item_usage(payload.at("usage"));
  }
  return sample;
}

evals::OutputItem parse_output_item(const json& payload) {
  evals::OutputItem item;
  item.raw = payload;
  item.id = payload.value("id", "");
  item.created_at = payload.value("created_at", 0);
  if (payload.contains("datasource_item")) {
    item.datasource_item = payload.at("datasource_item");
  }
  item.datasource_item_id = payload.value("datasource_item_id", 0);
  item.eval_id = payload.value("eval_id", "");
  item.object = payload.value("object", "");
  if (payload.contains("results") && payload.at("results").is_array()) {
    for (const auto& result : payload.at("results")) {
      item.results.push_back(parse_output_item_result(result));
    }
  }
  item.run_id = payload.value("run_id", "");
  if (payload.contains("sample") && payload.at("sample").is_object()) {
    item.sample = parse_output_item_sample(payload.at("sample"));
  }
  item.status = payload.value("status", "");
  return item;
}

evals::OutputItemList parse_output_item_list(const json& payload) {
  evals::OutputItemList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_output_item(entry));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  return list;
}

CursorPage<evals::Evaluation> make_evaluation_cursor_page(evals::EvaluationList list,
                                                          const PageRequestOptions& request_options,
                                                          const std::shared_ptr<std::function<CursorPage<evals::Evaluation>(
                                                              const PageRequestOptions&)>>& fetch_impl) {
  std::optional<std::string> cursor = list.next_cursor;
  if (!cursor && !list.data.empty()) {
    cursor = list.data.back().id;
  }
  return CursorPage<evals::Evaluation>(
      std::move(list.data),
      list.has_more,
      std::move(cursor),
      request_options,
      *fetch_impl,
      "after",
      std::move(list.raw));
}

CursorPage<evals::Run> make_run_cursor_page(evals::RunList list,
                                            const PageRequestOptions& request_options,
                                            const std::shared_ptr<std::function<CursorPage<evals::Run>(
                                                const PageRequestOptions&)>>& fetch_impl) {
  std::optional<std::string> cursor = list.next_cursor;
  if (!cursor && !list.data.empty()) {
    cursor = list.data.back().id;
  }
  return CursorPage<evals::Run>(
      std::move(list.data),
      list.has_more,
      std::move(cursor),
      request_options,
      *fetch_impl,
      "after",
      std::move(list.raw));
}

CursorPage<evals::OutputItem> make_output_item_cursor_page(
    evals::OutputItemList list,
    const PageRequestOptions& request_options,
    const std::shared_ptr<std::function<CursorPage<evals::OutputItem>(const PageRequestOptions&)>>& fetch_impl) {
  std::optional<std::string> cursor = list.next_cursor;
  if (!cursor && !list.data.empty()) {
    cursor = list.data.back().id;
  }
  return CursorPage<evals::OutputItem>(
      std::move(list.data),
      list.has_more,
      std::move(cursor),
      request_options,
      *fetch_impl,
      "after",
      std::move(list.raw));
}

}  // namespace

evals::Evaluation EvalsResource::create(const evals::EvaluationCreateParams& params) const {
  return create(params, RequestOptions{});
}

evals::Evaluation EvalsResource::create(const evals::EvaluationCreateParams& params,
                                        const RequestOptions& options) const {
  auto body = evaluation_create_body(params);
  RequestOptions request_options = options;
  auto response = client_.perform_request("POST", kEvalsPath, body.dump(), request_options);
  try {
    return parse_evaluation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation create response: ") + ex.what());
  }
}

evals::Evaluation EvalsResource::retrieve(const std::string& eval_id) const {
  return retrieve(eval_id, RequestOptions{});
}

evals::Evaluation EvalsResource::retrieve(const std::string& eval_id, const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_evaluation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation retrieve response: ") + ex.what());
  }
}

evals::Evaluation EvalsResource::update(const std::string& eval_id,
                                        const evals::EvaluationUpdateParams& params) const {
  return update(eval_id, params, RequestOptions{});
}

evals::Evaluation EvalsResource::update(const std::string& eval_id,
                                        const evals::EvaluationUpdateParams& params,
                                        const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id;
  json body;
  if (params.metadata && !params.metadata->empty()) {
    body["metadata"] = metadata_to_json(*params.metadata);
  }
  if (params.name) body["name"] = *params.name;
  auto response = client_.perform_request("POST", path, body.dump(), options);
  try {
    return parse_evaluation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation update response: ") + ex.what());
  }
}

evals::EvaluationDeleteResponse EvalsResource::remove(const std::string& eval_id) const {
  return remove(eval_id, RequestOptions{});
}

evals::EvaluationDeleteResponse EvalsResource::remove(const std::string& eval_id,
                                                      const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_delete_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation delete response: ") + ex.what());
  }
}

evals::EvaluationList EvalsResource::list() const {
  return list(evals::EvaluationListParams{}, RequestOptions{});
}

evals::EvaluationList EvalsResource::list(const evals::EvaluationListParams& params) const {
  return list(params, RequestOptions{});
}

evals::EvaluationList EvalsResource::list(const RequestOptions& options) const {
  return list(evals::EvaluationListParams{}, options);
}

evals::EvaluationList EvalsResource::list(const evals::EvaluationListParams& params,
                                          const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_evaluation_list_params(params, request_options);
  auto response = client_.perform_request("GET", kEvalsPath, "", request_options);
  try {
    return parse_evaluation_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation list response: ") + ex.what());
  }
}

CursorPage<evals::Evaluation> EvalsResource::list_page() const {
  return list_page(evals::EvaluationListParams{}, RequestOptions{});
}

CursorPage<evals::Evaluation> EvalsResource::list_page(const evals::EvaluationListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<evals::Evaluation> EvalsResource::list_page(const RequestOptions& options) const {
  return list_page(evals::EvaluationListParams{}, options);
}

CursorPage<evals::Evaluation> EvalsResource::list_page(const evals::EvaluationListParams& params,
                                                       const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_evaluation_list_params(params, request_options);
  auto fetch_impl = std::make_shared<std::function<CursorPage<evals::Evaluation>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<evals::Evaluation> {
    RequestOptions next = to_request_options(request_options);
    auto response = client_.perform_request(request_options.method, request_options.path, request_options.body, next);
    try {
      auto payload = json::parse(response.body);
      auto list = parse_evaluation_list(payload);
      return make_evaluation_cursor_page(std::move(list), request_options, fetch_impl);
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse evaluation page: ") + ex.what());
    }
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kEvalsPath;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);
  initial.body.clear();

  RequestOptions first_options = to_request_options(initial);
  auto response = client_.perform_request("GET", kEvalsPath, "", request_options);
  try {
    auto payload = json::parse(response.body);
    auto list = parse_evaluation_list(payload);
    return make_evaluation_cursor_page(std::move(list), initial, fetch_impl);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse evaluation page: ") + ex.what());
  }
}

evals::Run EvalsRunsResource::create(const std::string& eval_id,
                                     const evals::RunCreateParams& params) const {
  return create(eval_id, params, RequestOptions{});
}

evals::Run EvalsRunsResource::create(const std::string& eval_id,
                                     const evals::RunCreateParams& params,
                                     const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix;
  auto response = client_.perform_request("POST", path, run_create_body(params).dump(), options);
  try {
    return parse_run(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run create response: ") + ex.what());
  }
}

evals::Run EvalsRunsResource::retrieve(const std::string& run_id,
                                       const evals::RunRetrieveParams& params) const {
  return retrieve(run_id, params, RequestOptions{});
}

evals::Run EvalsRunsResource::retrieve(const std::string& run_id,
                                       const evals::RunRetrieveParams& params,
                                       const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + params.eval_id + kRunsSuffix + "/" + run_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_run(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run retrieve response: ") + ex.what());
  }
}

evals::RunList EvalsRunsResource::list(const std::string& eval_id) const {
  return list(eval_id, evals::RunListParams{}, RequestOptions{});
}

evals::RunList EvalsRunsResource::list(const std::string& eval_id,
                                       const evals::RunListParams& params) const {
  return list(eval_id, params, RequestOptions{});
}

evals::RunList EvalsRunsResource::list(const std::string& eval_id,
                                       const RequestOptions& options) const {
  return list(eval_id, evals::RunListParams{}, options);
}

evals::RunList EvalsRunsResource::list(const std::string& eval_id,
                                       const evals::RunListParams& params,
                                       const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix;
  RequestOptions request_options = options;
  apply_run_list_params(params, request_options);
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_run_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run list response: ") + ex.what());
  }
}

CursorPage<evals::Run> EvalsRunsResource::list_page(const std::string& eval_id) const {
  return list_page(eval_id, evals::RunListParams{}, RequestOptions{});
}

CursorPage<evals::Run> EvalsRunsResource::list_page(const std::string& eval_id,
                                                    const evals::RunListParams& params) const {
  return list_page(eval_id, params, RequestOptions{});
}

CursorPage<evals::Run> EvalsRunsResource::list_page(const std::string& eval_id,
                                                    const RequestOptions& options) const {
  return list_page(eval_id, evals::RunListParams{}, options);
}

CursorPage<evals::Run> EvalsRunsResource::list_page(const std::string& eval_id,
                                                    const evals::RunListParams& params,
                                                    const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix;
  RequestOptions request_options = options;
  apply_run_list_params(params, request_options);

  auto fetch_impl = std::make_shared<std::function<CursorPage<evals::Run>(const PageRequestOptions&)>>();
  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<evals::Run> {
    RequestOptions next = to_request_options(request_options);
    auto response = client_.perform_request(request_options.method, request_options.path, request_options.body, next);
    try {
      auto payload = json::parse(response.body);
      auto list = parse_run_list(payload);
      return make_run_cursor_page(std::move(list), request_options, fetch_impl);
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse eval run page: ") + ex.what());
    }
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = path;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);
  initial.body.clear();

  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    auto payload = json::parse(response.body);
    auto list = parse_run_list(payload);
    return make_run_cursor_page(std::move(list), initial, fetch_impl);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run page: ") + ex.what());
  }
}

evals::RunDeleteResponse EvalsRunsResource::remove(const std::string& run_id,
                                                   const evals::RunDeleteParams& params) const {
  return remove(run_id, params, RequestOptions{});
}

evals::RunDeleteResponse EvalsRunsResource::remove(const std::string& run_id,
                                                   const evals::RunDeleteParams& params,
                                                   const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + params.eval_id + kRunsSuffix + "/" + run_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_run_delete_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run delete response: ") + ex.what());
  }
}

evals::RunCancelResponse EvalsRunsResource::cancel(const std::string& run_id,
                                                   const evals::RunCancelParams& params) const {
  return cancel(run_id, params, RequestOptions{});
}

evals::RunCancelResponse EvalsRunsResource::cancel(const std::string& run_id,
                                                   const evals::RunCancelParams& params,
                                                   const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + params.eval_id + kRunsSuffix + "/" + run_id;
  auto response = client_.perform_request("POST", path, "", options);
  try {
    return parse_run_cancel_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval run cancel response: ") + ex.what());
  }
}

evals::OutputItem EvalsRunsOutputItemsResource::retrieve(const std::string& eval_id,
                                                         const std::string& run_id,
                                                         const std::string& output_item_id) const {
  return retrieve(eval_id, run_id, output_item_id, RequestOptions{});
}

evals::OutputItem EvalsRunsOutputItemsResource::retrieve(const std::string& eval_id,
                                                         const std::string& run_id,
                                                         const std::string& output_item_id,
                                                         const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix + "/" + run_id + kOutputItemsSuffix +
                           "/" + output_item_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_output_item(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval output item retrieve response: ") + ex.what());
  }
}

evals::OutputItemList EvalsRunsOutputItemsResource::list(const std::string& eval_id,
                                                         const std::string& run_id) const {
  return list(eval_id, run_id, evals::OutputItemListParams{}, RequestOptions{});
}

evals::OutputItemList EvalsRunsOutputItemsResource::list(const std::string& eval_id,
                                                         const std::string& run_id,
                                                         const evals::OutputItemListParams& params) const {
  return list(eval_id, run_id, params, RequestOptions{});
}

evals::OutputItemList EvalsRunsOutputItemsResource::list(const std::string& eval_id,
                                                         const std::string& run_id,
                                                         const RequestOptions& options) const {
  return list(eval_id, run_id, evals::OutputItemListParams{}, options);
}

evals::OutputItemList EvalsRunsOutputItemsResource::list(const std::string& eval_id,
                                                         const std::string& run_id,
                                                         const evals::OutputItemListParams& params,
                                                         const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix + "/" + run_id + kOutputItemsSuffix;
  RequestOptions request_options = options;
  apply_output_item_list_params(params, request_options);
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_output_item_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval output item list response: ") + ex.what());
  }
}

CursorPage<evals::OutputItem> EvalsRunsOutputItemsResource::list_page(const std::string& eval_id,
                                                                      const std::string& run_id) const {
  return list_page(eval_id, run_id, evals::OutputItemListParams{}, RequestOptions{});
}

CursorPage<evals::OutputItem> EvalsRunsOutputItemsResource::list_page(const std::string& eval_id,
                                                                      const std::string& run_id,
                                                                      const evals::OutputItemListParams& params) const {
  return list_page(eval_id, run_id, params, RequestOptions{});
}

CursorPage<evals::OutputItem> EvalsRunsOutputItemsResource::list_page(const std::string& eval_id,
                                                                      const std::string& run_id,
                                                                      const RequestOptions& options) const {
  return list_page(eval_id, run_id, evals::OutputItemListParams{}, options);
}

CursorPage<evals::OutputItem> EvalsRunsOutputItemsResource::list_page(const std::string& eval_id,
                                                                      const std::string& run_id,
                                                                      const evals::OutputItemListParams& params,
                                                                      const RequestOptions& options) const {
  const std::string path = std::string(kEvalsPath) + "/" + eval_id + kRunsSuffix + "/" + run_id + kOutputItemsSuffix;
  RequestOptions request_options = options;
  apply_output_item_list_params(params, request_options);

  auto fetch_impl =
      std::make_shared<std::function<CursorPage<evals::OutputItem>(const PageRequestOptions&)>>();
  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<evals::OutputItem> {
    RequestOptions next = to_request_options(request_options);
    auto response = client_.perform_request(request_options.method, request_options.path, request_options.body, next);
    try {
      auto payload = json::parse(response.body);
      auto list = parse_output_item_list(payload);
      return make_output_item_cursor_page(std::move(list), request_options, fetch_impl);
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse eval output item page: ") + ex.what());
    }
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = path;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);
  initial.body.clear();

  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    auto payload = json::parse(response.body);
    auto list = parse_output_item_list(payload);
    return make_output_item_cursor_page(std::move(list), initial, fetch_impl);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse eval output item page: ") + ex.what());
  }
}

}  // namespace openai

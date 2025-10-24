#include "openai/fine_tuning.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <type_traits>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kFineTuningJobsPath = "/fine_tuning/jobs";

double as_double(const json& value) {
  if (value.is_number_float()) {
    return value.get<double>();
  }
  if (value.is_number_integer()) {
    return static_cast<double>(value.get<long long>());
  }
  return 0.0;
}

AutoNumber parse_auto_number(const json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  return as_double(value);
}

AutoInteger parse_auto_int(const json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_number_integer()) {
    return value.get<int>();
  }
  return static_cast<int>(as_double(value));
}

json auto_number_to_json(const AutoNumber& value) {
  if (std::holds_alternative<std::string>(value)) {
    return std::get<std::string>(value);
  }
  return json(std::get<double>(value));
}

json auto_int_to_json(const AutoInteger& value) {
  if (std::holds_alternative<std::string>(value)) {
    return std::get<std::string>(value);
  }
  return json(std::get<int>(value));
}

json hyperparameters_to_json(const SupervisedHyperparameters& params) {
  json body = json::object();
  if (params.batch_size) body["batch_size"] = auto_int_to_json(*params.batch_size);
  if (params.learning_rate_multiplier) body["learning_rate_multiplier"] = auto_number_to_json(*params.learning_rate_multiplier);
  if (params.n_epochs) body["n_epochs"] = auto_int_to_json(*params.n_epochs);
  return body;
}

json dpo_hyperparameters_to_json(const DpoHyperparameters& params) {
  json body = json::object();
  if (params.batch_size) body["batch_size"] = auto_int_to_json(*params.batch_size);
  if (params.beta) body["beta"] = auto_number_to_json(*params.beta);
  if (params.learning_rate_multiplier) body["learning_rate_multiplier"] = auto_number_to_json(*params.learning_rate_multiplier);
  if (params.n_epochs) body["n_epochs"] = auto_int_to_json(*params.n_epochs);
  return body;
}

json reinforcement_hyperparameters_to_json(const ReinforcementHyperparameters& params) {
  json body = json::object();
  if (params.batch_size) body["batch_size"] = auto_int_to_json(*params.batch_size);
  if (params.compute_multiplier) body["compute_multiplier"] = auto_number_to_json(*params.compute_multiplier);
  if (params.eval_interval) body["eval_interval"] = auto_int_to_json(*params.eval_interval);
  if (params.eval_samples) body["eval_samples"] = auto_int_to_json(*params.eval_samples);
  if (params.learning_rate_multiplier) body["learning_rate_multiplier"] = auto_number_to_json(*params.learning_rate_multiplier);
  if (params.n_epochs) body["n_epochs"] = auto_int_to_json(*params.n_epochs);
  if (params.reasoning_effort) body["reasoning_effort"] = *params.reasoning_effort;
  return body;
}

json method_to_json(const FineTuningMethod& method) {
  json body = json::object();
  switch (method.type) {
    case FineTuningMethod::Type::Supervised:
      body["type"] = "supervised";
      if (method.supervised && method.supervised->hyperparameters) {
        body["supervised"] = json::object();
        body["supervised"]["hyperparameters"] = hyperparameters_to_json(*method.supervised->hyperparameters);
      }
      break;
    case FineTuningMethod::Type::DPO:
      body["type"] = "dpo";
      if (method.dpo && method.dpo->hyperparameters) {
        body["dpo"] = json::object();
        body["dpo"]["hyperparameters"] = dpo_hyperparameters_to_json(*method.dpo->hyperparameters);
      }
      break;
    case FineTuningMethod::Type::Reinforcement:
      body["type"] = "reinforcement";
      if (method.reinforcement) {
        json reinforcement = json::object();
        if (method.reinforcement->grader) reinforcement["grader"] = *method.reinforcement->grader;
        if (method.reinforcement->hyperparameters) {
          reinforcement["hyperparameters"] = reinforcement_hyperparameters_to_json(*method.reinforcement->hyperparameters);
        }
        body["reinforcement"] = std::move(reinforcement);
      }
      break;
  }
  return body;
}

FineTuningMethod::Type parse_method_type(const std::string& value) {
  if (value == "dpo") return FineTuningMethod::Type::DPO;
  if (value == "reinforcement") return FineTuningMethod::Type::Reinforcement;
  return FineTuningMethod::Type::Supervised;
}

std::optional<SupervisedHyperparameters> parse_supervised_hyperparameters(const json& payload) {
  if (!payload.is_object()) return std::nullopt;
  SupervisedHyperparameters params;
  if (payload.contains("batch_size")) params.batch_size = parse_auto_int(payload.at("batch_size"));
  if (payload.contains("learning_rate_multiplier")) params.learning_rate_multiplier = parse_auto_number(payload.at("learning_rate_multiplier"));
  if (payload.contains("n_epochs")) params.n_epochs = parse_auto_int(payload.at("n_epochs"));
  return params;
}

std::optional<DpoHyperparameters> parse_dpo_hyperparameters(const json& payload) {
  if (!payload.is_object()) return std::nullopt;
  DpoHyperparameters params;
  if (payload.contains("batch_size")) params.batch_size = parse_auto_int(payload.at("batch_size"));
  if (payload.contains("beta")) params.beta = parse_auto_number(payload.at("beta"));
  if (payload.contains("learning_rate_multiplier")) params.learning_rate_multiplier = parse_auto_number(payload.at("learning_rate_multiplier"));
  if (payload.contains("n_epochs")) params.n_epochs = parse_auto_int(payload.at("n_epochs"));
  return params;
}

std::optional<ReinforcementHyperparameters> parse_reinforcement_hyperparameters(const json& payload) {
  if (!payload.is_object()) return std::nullopt;
  ReinforcementHyperparameters params;
  if (payload.contains("batch_size")) params.batch_size = parse_auto_int(payload.at("batch_size"));
  if (payload.contains("compute_multiplier")) params.compute_multiplier = parse_auto_number(payload.at("compute_multiplier"));
  if (payload.contains("eval_interval")) params.eval_interval = parse_auto_int(payload.at("eval_interval"));
  if (payload.contains("eval_samples")) params.eval_samples = parse_auto_int(payload.at("eval_samples"));
  if (payload.contains("learning_rate_multiplier")) params.learning_rate_multiplier = parse_auto_number(payload.at("learning_rate_multiplier"));
  if (payload.contains("n_epochs")) params.n_epochs = parse_auto_int(payload.at("n_epochs"));
  if (payload.contains("reasoning_effort")) params.reasoning_effort = payload.at("reasoning_effort").get<std::string>();
  return params;
}

WandbIntegrationParams parse_wandb_integration(const json& payload) {
  WandbIntegrationParams params;
  params.project = payload.value("project", "");
  if (payload.contains("entity") && !payload.at("entity").is_null()) {
    params.entity = payload.at("entity").get<std::string>();
  }
  if (payload.contains("name") && !payload.at("name").is_null()) {
    params.name = payload.at("name").get<std::string>();
  }
  if (payload.contains("tags") && payload.at("tags").is_array()) {
    for (const auto& tag : payload.at("tags")) {
      params.tags.push_back(tag.get<std::string>());
    }
  }
  return params;
}

FineTuningJob parse_job(const json& payload) {
  FineTuningJob job;
  job.raw = payload;
  job.id = payload.value("id", "");
  job.created_at = payload.value("created_at", 0);
  if (payload.contains("error") && !payload.at("error").is_null()) {
    FineTuningJobError error;
    const auto& err_json = payload.at("error");
    error.code = err_json.value("code", "");
    error.message = err_json.value("message", "");
    if (err_json.contains("param") && !err_json.at("param").is_null()) {
      error.param = err_json.at("param").get<std::string>();
    }
    job.error = error;
  }
  if (payload.contains("fine_tuned_model") && !payload.at("fine_tuned_model").is_null()) {
    job.fine_tuned_model = payload.at("fine_tuned_model").get<std::string>();
  }
  if (payload.contains("finished_at") && !payload.at("finished_at").is_null()) {
    job.finished_at = payload.at("finished_at").get<int>();
  }
  if (payload.contains("hyperparameters") && payload.at("hyperparameters").is_object()) {
    FineTuningJobHyperparameters hyper;
    const auto& hyper_json = payload.at("hyperparameters");
    if (hyper_json.contains("batch_size")) hyper.batch_size = parse_auto_int(hyper_json.at("batch_size"));
    if (hyper_json.contains("learning_rate_multiplier")) hyper.learning_rate_multiplier = parse_auto_number(hyper_json.at("learning_rate_multiplier"));
    if (hyper_json.contains("n_epochs")) hyper.n_epochs = parse_auto_int(hyper_json.at("n_epochs"));
    job.hyperparameters = hyper;
  }
  job.model = payload.value("model", "");
  job.object = payload.value("object", "");
  job.organization_id = payload.value("organization_id", "");
  if (payload.contains("result_files") && payload.at("result_files").is_array()) {
    for (const auto& entry : payload.at("result_files")) {
      job.result_files.push_back(entry.get<std::string>());
    }
  }
  job.seed = payload.value("seed", 0);
  job.status = payload.value("status", "");
  if (payload.contains("trained_tokens") && !payload.at("trained_tokens").is_null()) {
    job.trained_tokens = payload.at("trained_tokens").get<int>();
  }
  job.training_file = payload.value("training_file", "");
  if (payload.contains("validation_file") && !payload.at("validation_file").is_null()) {
    job.validation_file = payload.at("validation_file").get<std::string>();
  }
  if (payload.contains("estimated_finish") && !payload.at("estimated_finish").is_null()) {
    job.estimated_finish = payload.at("estimated_finish").get<int>();
  }
  if (payload.contains("integrations") && payload.at("integrations").is_array()) {
    for (const auto& integration_json : payload.at("integrations")) {
      FineTuningJobIntegration integration;
      integration.type = integration_json.value("type", "");
      if (integration_json.contains("wandb") && integration_json.at("wandb").is_object()) {
        integration.wandb = parse_wandb_integration(integration_json.at("wandb"));
      }
      job.integrations.push_back(std::move(integration));
    }
  }
  if (payload.contains("metadata")) {
    if (payload.at("metadata").is_object()) {
      std::map<std::string, std::string> metadata;
      for (const auto& item : payload.at("metadata").items()) {
        if (item.value().is_string()) {
          metadata[item.key()] = item.value().get<std::string>();
        }
      }
      job.metadata = std::move(metadata);
    } else if (payload.at("metadata").is_null()) {
      job.metadata = std::nullopt;
    }
  }
  if (payload.contains("method") && payload.at("method").is_object()) {
    FineTuningJobMethod method;
    const auto& method_json = payload.at("method");
    method.type = parse_method_type(method_json.value("type", "supervised"));
    if (method_json.contains("supervised") && method_json.at("supervised").is_object()) {
      SupervisedMethodConfig cfg;
      cfg.hyperparameters = parse_supervised_hyperparameters(method_json.at("supervised").value("hyperparameters", json::object()));
      method.supervised = std::move(cfg);
    }
    if (method_json.contains("dpo") && method_json.at("dpo").is_object()) {
      DpoMethodConfig cfg;
      cfg.hyperparameters = parse_dpo_hyperparameters(method_json.at("dpo").value("hyperparameters", json::object()));
      method.dpo = std::move(cfg);
    }
    if (method_json.contains("reinforcement") && method_json.at("reinforcement").is_object()) {
      ReinforcementMethodConfig cfg;
      const auto& reinforcement_json = method_json.at("reinforcement");
      if (reinforcement_json.contains("grader") && reinforcement_json.at("grader").is_string()) {
        cfg.grader = reinforcement_json.at("grader").get<std::string>();
      }
      cfg.hyperparameters = parse_reinforcement_hyperparameters(reinforcement_json.value("hyperparameters", json::object()));
      method.reinforcement = std::move(cfg);
    }
    job.method = std::move(method);
  }
  return job;
}

json label_model_input_to_json(const graders::LabelModelGraderInput& input) {
  json payload = json::object();
  payload["role"] = input.role;
  payload["content"] = input.content;
  if (input.type) payload["type"] = *input.type;
  return payload;
}

json score_model_input_to_json(const graders::ScoreModelGraderInput& input) {
  json payload = json::object();
  payload["content"] = input.content;
  payload["role"] = input.role;
  if (input.type) payload["type"] = *input.type;
  return payload;
}

json string_check_to_json(const graders::StringCheckGrader& grader) {
  json payload = json::object();
  payload["input"] = grader.input;
  payload["name"] = grader.name;
  payload["operation"] = grader.operation;
  payload["reference"] = grader.reference;
  payload["type"] = grader.type;
  return payload;
}

json text_similarity_to_json(const graders::TextSimilarityGrader& grader) {
  json payload = json::object();
  payload["evaluation_metric"] = grader.evaluation_metric;
  payload["input"] = grader.input;
  payload["name"] = grader.name;
  payload["reference"] = grader.reference;
  payload["type"] = grader.type;
  return payload;
}

json python_grader_to_json(const graders::PythonGrader& grader) {
  json payload = json::object();
  payload["name"] = grader.name;
  payload["source"] = grader.source;
  payload["type"] = grader.type;
  if (grader.image_tag) payload["image_tag"] = *grader.image_tag;
  return payload;
}

json score_model_to_json(const graders::ScoreModelGrader& grader) {
  json payload = json::object();
  payload["input"] = json::array();
  for (const auto& item : grader.input) {
    payload["input"].push_back(score_model_input_to_json(item));
  }
  payload["model"] = grader.model;
  payload["name"] = grader.name;
  payload["type"] = grader.type;
  if (grader.range) payload["range"] = *grader.range;
  if (grader.sampling_params) {
    json sampling = json::object();
    if (grader.sampling_params->max_tokens) sampling["max_tokens"] = *grader.sampling_params->max_tokens;
    if (grader.sampling_params->temperature) sampling["temperature"] = *grader.sampling_params->temperature;
    if (grader.sampling_params->top_p) sampling["top_p"] = *grader.sampling_params->top_p;
    payload["sampling_params"] = std::move(sampling);
  }
  return payload;
}

json label_model_to_json(const graders::LabelModelGrader& grader) {
  json payload = json::object();
  json inputs = json::array();
  for (const auto& item : grader.input) {
    inputs.push_back(label_model_input_to_json(item));
  }
  payload["input"] = std::move(inputs);
  payload["labels"] = grader.labels;
  payload["model"] = grader.model;
  payload["name"] = grader.name;
  payload["passing_labels"] = grader.passing_labels;
  payload["type"] = grader.type;
  return payload;
}

json multi_grader_to_json(const graders::MultiGrader& grader);

template <typename Variant>
json graders_variant_to_json(const Variant& grader_variant) {
  return std::visit([](const auto& value) -> json {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, graders::StringCheckGrader>) {
      return string_check_to_json(value);
    } else if constexpr (std::is_same_v<T, graders::TextSimilarityGrader>) {
      return text_similarity_to_json(value);
    } else if constexpr (std::is_same_v<T, graders::PythonGrader>) {
      return python_grader_to_json(value);
    } else if constexpr (std::is_same_v<T, graders::ScoreModelGrader>) {
      return score_model_to_json(value);
    } else if constexpr (std::is_same_v<T, graders::MultiGrader>) {
      return multi_grader_to_json(value);
    } else {
      return label_model_to_json(value);
    }
  }, grader_variant);
}

json multi_grader_to_json(const graders::MultiGrader& grader) {
  json payload = json::object();
  payload["calculate_output"] = grader.calculate_output;
  payload["graders"] = graders_variant_to_json(grader.graders);
  payload["name"] = grader.name;
  payload["type"] = grader.type;
  return payload;
}

graders::StringCheckGrader parse_string_check_grader(const json& payload) {
  graders::StringCheckGrader grader;
  grader.input = payload.value("input", "");
  grader.name = payload.value("name", "");
  grader.operation = payload.value("operation", "");
  grader.reference = payload.value("reference", "");
  grader.type = payload.value("type", "string_check");
  return grader;
}

graders::TextSimilarityGrader parse_text_similarity_grader(const json& payload) {
  graders::TextSimilarityGrader grader;
  grader.evaluation_metric = payload.value("evaluation_metric", "");
  grader.input = payload.value("input", "");
  grader.name = payload.value("name", "");
  grader.reference = payload.value("reference", "");
  grader.type = payload.value("type", "text_similarity");
  return grader;
}

graders::PythonGrader parse_python_grader(const json& payload) {
  graders::PythonGrader grader;
  grader.name = payload.value("name", "");
  grader.source = payload.value("source", "");
  grader.type = payload.value("type", "python");
  if (payload.contains("image_tag") && payload.at("image_tag").is_string()) {
    grader.image_tag = payload.at("image_tag").get<std::string>();
  }
  return grader;
}

graders::ScoreModelGraderInput parse_score_model_input(const json& payload) {
  graders::ScoreModelGraderInput input;
  if (payload.contains("content")) input.content = payload.at("content");
  input.role = payload.value("role", "");
  if (payload.contains("type") && payload.at("type").is_string()) input.type = payload.at("type").get<std::string>();
  return input;
}

graders::ScoreModelGrader parse_score_model_grader(const json& payload) {
  graders::ScoreModelGrader grader;
  grader.type = payload.value("type", "score_model");
  grader.model = payload.value("model", "");
  grader.name = payload.value("name", "");
  if (payload.contains("input") && payload.at("input").is_array()) {
    for (const auto& item : payload.at("input")) {
      grader.input.push_back(parse_score_model_input(item));
    }
  }
  if (payload.contains("range") && payload.at("range").is_array()) {
    grader.range = payload.at("range").get<std::vector<double>>();
  }
  if (payload.contains("sampling_params") && payload.at("sampling_params").is_object()) {
    graders::ScoreModelGraderSamplingParams params;
    const auto& sampling = payload.at("sampling_params");
    if (sampling.contains("max_tokens") && sampling.at("max_tokens").is_number_integer()) {
      params.max_tokens = sampling.at("max_tokens").get<int>();
    }
    if (sampling.contains("temperature") && sampling.at("temperature").is_number()) {
      params.temperature = sampling.at("temperature").get<double>();
    }
    if (sampling.contains("top_p") && sampling.at("top_p").is_number()) {
      params.top_p = sampling.at("top_p").get<double>();
    }
    grader.sampling_params = params;
  }
  return grader;
}

graders::LabelModelGrader parse_label_model_grader(const json& payload) {
  graders::LabelModelGrader grader;
  grader.type = payload.value("type", "label_model");
  grader.model = payload.value("model", "");
  grader.name = payload.value("name", "");
  if (payload.contains("input") && payload.at("input").is_array()) {
    for (const auto& item : payload.at("input")) {
      graders::LabelModelGraderInput input;
      input.content = item.contains("content") ? item.at("content") : json::array();
      input.role = item.value("role", "");
      if (item.contains("type") && item.at("type").is_string()) input.type = item.at("type").get<std::string>();
      grader.input.push_back(std::move(input));
    }
  }
  if (payload.contains("labels") && payload.at("labels").is_array()) {
    grader.labels = payload.at("labels").get<std::vector<std::string>>();
  }
  if (payload.contains("passing_labels") && payload.at("passing_labels").is_array()) {
    grader.passing_labels = payload.at("passing_labels").get<std::vector<std::string>>();
  }
  return grader;
}

std::variant<graders::StringCheckGrader,
             graders::TextSimilarityGrader,
             graders::PythonGrader,
             graders::ScoreModelGrader,
             graders::LabelModelGrader> parse_nested_grader_variant(const json& payload) {
  const std::string type = payload.value("type", "");
  if (type == "string_check") return parse_string_check_grader(payload);
  if (type == "text_similarity") return parse_text_similarity_grader(payload);
  if (type == "python") return parse_python_grader(payload);
  if (type == "score_model") return parse_score_model_grader(payload);
  return parse_label_model_grader(payload);
}

graders::MultiGrader parse_multi_grader(const json& payload);

std::variant<graders::StringCheckGrader,
             graders::TextSimilarityGrader,
             graders::PythonGrader,
             graders::ScoreModelGrader,
             graders::MultiGrader,
             graders::LabelModelGrader> parse_grader_variant(const json& payload) {
  const std::string type = payload.value("type", "");
  if (type == "string_check") return parse_string_check_grader(payload);
  if (type == "text_similarity") return parse_text_similarity_grader(payload);
  if (type == "python") return parse_python_grader(payload);
  if (type == "score_model") return parse_score_model_grader(payload);
  if (type == "multi") return parse_multi_grader(payload);
  if (type == "label_model") return parse_label_model_grader(payload);
  return parse_string_check_grader(payload);
}

graders::MultiGrader parse_multi_grader(const json& payload) {
  graders::MultiGrader grader;
  grader.type = payload.value("type", "multi");
  grader.calculate_output = payload.value("calculate_output", "");
  grader.name = payload.value("name", "");
  if (payload.contains("graders") && payload.at("graders").is_object()) {
    grader.graders = parse_nested_grader_variant(payload.at("graders"));
  }
  return grader;
}

GraderRunMetadataErrors parse_grader_run_errors(const json& payload) {
  GraderRunMetadataErrors errors;
  errors.formula_parse_error = payload.value("formula_parse_error", false);
  errors.invalid_variable_error = payload.value("invalid_variable_error", false);
  errors.model_grader_parse_error = payload.value("model_grader_parse_error", false);
  errors.model_grader_refusal_error = payload.value("model_grader_refusal_error", false);
  errors.model_grader_server_error = payload.value("model_grader_server_error", false);
  if (payload.contains("model_grader_server_error_details") && !payload.at("model_grader_server_error_details").is_null()) {
    errors.model_grader_server_error_details = payload.at("model_grader_server_error_details").get<std::string>();
  }
  errors.other_error = payload.value("other_error", false);
  errors.python_grader_runtime_error = payload.value("python_grader_runtime_error", false);
  if (payload.contains("python_grader_runtime_error_details") && !payload.at("python_grader_runtime_error_details").is_null()) {
    errors.python_grader_runtime_error_details = payload.at("python_grader_runtime_error_details").get<std::string>();
  }
  errors.python_grader_server_error = payload.value("python_grader_server_error", false);
  if (payload.contains("python_grader_server_error_type") && !payload.at("python_grader_server_error_type").is_null()) {
    errors.python_grader_server_error_type = payload.at("python_grader_server_error_type").get<std::string>();
  }
  errors.sample_parse_error = payload.value("sample_parse_error", false);
  errors.truncated_observation_error = payload.value("truncated_observation_error", false);
  errors.unresponsive_reward_error = payload.value("unresponsive_reward_error", false);
  return errors;
}

GraderRunMetadata parse_grader_run_metadata(const json& payload) {
  GraderRunMetadata metadata;
  metadata.execution_time = payload.value("execution_time", 0.0);
  metadata.name = payload.value("name", "");
  if (payload.contains("sampled_model_name") && !payload.at("sampled_model_name").is_null()) {
    metadata.sampled_model_name = payload.at("sampled_model_name").get<std::string>();
  }
  if (payload.contains("scores") && payload.at("scores").is_object()) {
    for (const auto& item : payload.at("scores").items()) {
      metadata.scores[item.key()] = item.value();
    }
  }
  if (payload.contains("token_usage") && !payload.at("token_usage").is_null()) {
    metadata.token_usage = payload.at("token_usage").get<double>();
  }
  metadata.type = payload.value("type", "");
  if (payload.contains("errors") && payload.at("errors").is_object()) {
    metadata.errors = parse_grader_run_errors(payload.at("errors"));
  }
  return metadata;
}

GraderRunResponse parse_grader_run_response(const json& payload) {
  GraderRunResponse response;
  response.raw = payload;
  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    response.metadata = parse_grader_run_metadata(payload.at("metadata"));
  }
  response.reward = payload.value("reward", 0.0);
  if (payload.contains("model_grader_token_usage_per_model") && payload.at("model_grader_token_usage_per_model").is_object()) {
    for (const auto& item : payload.at("model_grader_token_usage_per_model").items()) {
      response.model_grader_token_usage_per_model[item.key()] = item.value();
    }
  }
  if (payload.contains("sub_rewards") && payload.at("sub_rewards").is_object()) {
    for (const auto& item : payload.at("sub_rewards").items()) {
      response.sub_rewards[item.key()] = item.value();
    }
  }
  return response;
}

GraderValidateResponse parse_grader_validate_response(const json& payload) {
  GraderValidateResponse response;
  response.raw = payload;
  if (payload.contains("grader") && payload.at("grader").is_object()) {
    response.grader = parse_grader_variant(payload.at("grader"));
  }
  return response;
}

FineTuningJobEvent parse_job_event(const json& payload) {
  FineTuningJobEvent event;
  event.raw = payload;
  event.id = payload.value("id", "");
  event.created_at = payload.value("created_at", 0);
  event.level = payload.value("level", "");
  event.message = payload.value("message", "");
  event.object = payload.value("object", "");
  if (payload.contains("data")) {
    event.data = payload.at("data");
  }
  if (payload.contains("type") && !payload.at("type").is_null()) {
    event.type = payload.at("type").get<std::string>();
  }
  return event;
}

FineTuningJobList parse_job_list(const json& payload) {
  FineTuningJobList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_job(entry));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.next_cursor = payload.at("last_id").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

FineTuningJobEventsList parse_events_list(const json& payload) {
  FineTuningJobEventsList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_job_event(entry));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.next_cursor = payload.at("last_id").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

FineTuningJobCheckpoint parse_checkpoint(const json& payload) {
  FineTuningJobCheckpoint checkpoint;
  checkpoint.raw = payload;
  checkpoint.id = payload.value("id", "");
  checkpoint.created_at = payload.value("created_at", 0);
  checkpoint.fine_tuned_model_checkpoint = payload.value("fine_tuned_model_checkpoint", "");
  checkpoint.fine_tuning_job_id = payload.value("fine_tuning_job_id", "");
  checkpoint.object = payload.value("object", "");
  checkpoint.step_number = payload.value("step_number", 0);
  if (payload.contains("metrics") && payload.at("metrics").is_object()) {
    const auto& metrics = payload.at("metrics");
    if (metrics.contains("full_valid_loss") && !metrics.at("full_valid_loss").is_null()) {
      checkpoint.metrics.full_valid_loss = metrics.at("full_valid_loss").get<double>();
    }
    if (metrics.contains("full_valid_mean_token_accuracy") && !metrics.at("full_valid_mean_token_accuracy").is_null()) {
      checkpoint.metrics.full_valid_mean_token_accuracy = metrics.at("full_valid_mean_token_accuracy").get<double>();
    }
    if (metrics.contains("step") && !metrics.at("step").is_null()) {
      checkpoint.metrics.step = metrics.at("step").get<int>();
    }
    if (metrics.contains("train_loss") && !metrics.at("train_loss").is_null()) {
      checkpoint.metrics.train_loss = metrics.at("train_loss").get<double>();
    }
    if (metrics.contains("train_mean_token_accuracy") && !metrics.at("train_mean_token_accuracy").is_null()) {
      checkpoint.metrics.train_mean_token_accuracy = metrics.at("train_mean_token_accuracy").get<double>();
    }
    if (metrics.contains("valid_loss") && !metrics.at("valid_loss").is_null()) {
      checkpoint.metrics.valid_loss = metrics.at("valid_loss").get<double>();
    }
    if (metrics.contains("valid_mean_token_accuracy") && !metrics.at("valid_mean_token_accuracy").is_null()) {
      checkpoint.metrics.valid_mean_token_accuracy = metrics.at("valid_mean_token_accuracy").get<double>();
    }
  }
  return checkpoint;
}

FineTuningJobCheckpointList parse_checkpoint_list(const json& payload) {
  FineTuningJobCheckpointList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_checkpoint(entry));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.next_cursor = payload.at("last_id").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

FineTuningCheckpointPermission parse_checkpoint_permission(const json& payload) {
  FineTuningCheckpointPermission permission;
  permission.raw = payload;
  permission.id = payload.value("id", "");
  permission.created_at = payload.value("created_at", 0);
  permission.object = payload.value("object", "");
  permission.project_id = payload.value("project_id", "");
  return permission;
}

FineTuningCheckpointPermissionList parse_checkpoint_permission_list(const json& payload) {
  FineTuningCheckpointPermissionList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  list.object = payload.value("object", "");
  if (payload.contains("first_id") && payload.at("first_id").is_string()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_checkpoint_permission(entry));
    }
  }
  return list;
}

FineTuningCheckpointPermissionDeleteResponse parse_checkpoint_permission_delete_response(const json& payload) {
  FineTuningCheckpointPermissionDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

json checkpoint_permission_create_to_json(const FineTuningCheckpointPermissionCreateParams& params) {
  json body = json::object();
  json projects = json::array();
  for (const auto& project_id : params.project_ids) {
    projects.push_back(project_id);
  }
  body["project_ids"] = std::move(projects);
  return body;
}

json job_create_to_json(const JobCreateParams& request) {
  json body = json::object();
  body["model"] = request.model;
  body["training_file"] = request.training_file;
  if (request.hyperparameters) {
    body["hyperparameters"] = hyperparameters_to_json(*request.hyperparameters);
  }
  if (!request.integrations.empty()) {
    json integrations = json::array();
    for (const auto& integration : request.integrations) {
      json wandb = json::object();
      wandb["project"] = integration.wandb.project;
      if (integration.wandb.entity) wandb["entity"] = *integration.wandb.entity;
      if (integration.wandb.name) wandb["name"] = *integration.wandb.name;
      if (!integration.wandb.tags.empty()) {
        wandb["tags"] = integration.wandb.tags;
      }
      json integration_json = json::object();
      integration_json["type"] = "wandb";
      integration_json["wandb"] = std::move(wandb);
      integrations.push_back(std::move(integration_json));
    }
    body["integrations"] = std::move(integrations);
  }
  if (request.metadata) {
    json metadata = json::object();
    for (const auto& [key, value] : *request.metadata) {
      metadata[key] = value;
    }
    body["metadata"] = std::move(metadata);
  }
  if (request.method) {
    body["method"] = method_to_json(*request.method);
  }
  if (request.seed) body["seed"] = *request.seed;
  if (request.suffix) body["suffix"] = *request.suffix;
  if (request.validation_file) body["validation_file"] = *request.validation_file;
  return body;
}

std::string jobs_path(const std::string& job_id) {
  return std::string(kFineTuningJobsPath) + "/" + job_id;
}

std::string job_events_path(const std::string& job_id) {
  return jobs_path(job_id) + "/events";
}

std::string job_checkpoints_path(const std::string& job_id) {
  return jobs_path(job_id) + "/checkpoints";
}

std::string checkpoint_permissions_path(const std::string& checkpoint_id) {
  return "/fine_tuning/checkpoints/" + checkpoint_id + "/permissions";
}

}  // namespace

FineTuningJob FineTuningJobsResource::create(const JobCreateParams& request) const {
  return create(request, RequestOptions{});
}

FineTuningJob FineTuningJobsResource::create(const JobCreateParams& request, const RequestOptions& options) const {
  auto body = job_create_to_json(request).dump();
  auto response = client_.perform_request("POST", kFineTuningJobsPath, body, options);
  try {
    return parse_job(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job create response: ") + ex.what());
  }
}

FineTuningJob FineTuningJobsResource::retrieve(const std::string& job_id) const {
  return retrieve(job_id, RequestOptions{});
}

FineTuningJob FineTuningJobsResource::retrieve(const std::string& job_id, const RequestOptions& options) const {
  auto response = client_.perform_request("GET", jobs_path(job_id), "", options);
  try {
    return parse_job(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job retrieve response: ") + ex.what());
  }
}

FineTuningJobList FineTuningJobsResource::list(const JobListParams& params) const {
  return list(params, RequestOptions{});
}

FineTuningJobList FineTuningJobsResource::list(const JobListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.metadata) {
    for (const auto& [key, value] : *params.metadata) {
      request_options.query_params["metadata[" + key + "]"] = value;
    }
  } else if (params.metadata_null) {
    request_options.query_params["metadata"] = "null";
  }
  auto response = client_.perform_request("GET", kFineTuningJobsPath, "", request_options);
  try {
    return parse_job_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job list response: ") + ex.what());
  }
}

FineTuningJobList FineTuningJobsResource::list() const {
  return list(JobListParams{});
}

FineTuningJobList FineTuningJobsResource::list(const RequestOptions& options) const {
  return list(JobListParams{}, options);
}

CursorPage<FineTuningJob> FineTuningJobsResource::list_page(const JobListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<FineTuningJob> FineTuningJobsResource::list_page(const JobListParams& params,
                                                            const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.metadata) {
    for (const auto& [key, value] : *params.metadata) {
      request_options.query_params["metadata[" + key + "]"] = value;
    }
  } else if (params.metadata_null) {
    request_options.query_params["metadata"] = "null";
  }

  auto fetch_impl = std::make_shared<std::function<CursorPage<FineTuningJob>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<FineTuningJob> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    FineTuningJobList list;
    try {
      list = parse_job_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse fine-tuning job list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<FineTuningJob>(std::move(list.data),
                                     list.has_more,
                                     std::move(cursor),
                                     request_options,
                                     *fetch_impl,
                                     "after",
                                     std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kFineTuningJobsPath;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

CursorPage<FineTuningJob> FineTuningJobsResource::list_page() const {
  return list_page(JobListParams{});
}

CursorPage<FineTuningJob> FineTuningJobsResource::list_page(const RequestOptions& options) const {
  return list_page(JobListParams{}, options);
}

FineTuningJob FineTuningJobsResource::cancel(const std::string& job_id) const {
  return cancel(job_id, RequestOptions{});
}

FineTuningJob FineTuningJobsResource::cancel(const std::string& job_id, const RequestOptions& options) const {
  auto response = client_.perform_request("POST", jobs_path(job_id) + "/cancel", json::object().dump(), options);
  try {
    return parse_job(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job cancel response: ") + ex.what());
  }
}

FineTuningJob FineTuningJobsResource::pause(const std::string& job_id) const {
  return pause(job_id, RequestOptions{});
}

FineTuningJob FineTuningJobsResource::pause(const std::string& job_id, const RequestOptions& options) const {
  auto response = client_.perform_request("POST", jobs_path(job_id) + "/pause", json::object().dump(), options);
  try {
    return parse_job(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job pause response: ") + ex.what());
  }
}

FineTuningJob FineTuningJobsResource::resume(const std::string& job_id) const {
  return resume(job_id, RequestOptions{});
}

FineTuningJob FineTuningJobsResource::resume(const std::string& job_id, const RequestOptions& options) const {
  auto response = client_.perform_request("POST", jobs_path(job_id) + "/resume", json::object().dump(), options);
  try {
    return parse_job(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job resume response: ") + ex.what());
  }
}

FineTuningJobEventsList FineTuningJobsResource::list_events(const std::string& job_id,
                                                            const JobListEventsParams& params) const {
  return list_events(job_id, params, RequestOptions{});
}

FineTuningJobEventsList FineTuningJobsResource::list_events(const std::string& job_id,
                                                            const JobListEventsParams& params,
                                                            const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) request_options.query_params["after"] = *params.after;

  auto response = client_.perform_request("GET", job_events_path(job_id), "", request_options);
  try {
    return parse_events_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job events response: ") + ex.what());
  }
}

FineTuningJobEventsList FineTuningJobsResource::list_events(const std::string& job_id) const {
  return list_events(job_id, JobListEventsParams{});
}

FineTuningJobEventsList FineTuningJobsResource::list_events(const std::string& job_id,
                                                            const RequestOptions& options) const {
  return list_events(job_id, JobListEventsParams{}, options);
}

CursorPage<FineTuningJobEvent> FineTuningJobsResource::list_events_page(const std::string& job_id,
                                                                        const JobListEventsParams& params) const {
  return list_events_page(job_id, params, RequestOptions{});
}

CursorPage<FineTuningJobEvent> FineTuningJobsResource::list_events_page(const std::string& job_id,
                                                                        const JobListEventsParams& params,
                                                                        const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) request_options.query_params["after"] = *params.after;

  auto fetch_impl = std::make_shared<std::function<CursorPage<FineTuningJobEvent>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl, job_id](const PageRequestOptions& request_options) -> CursorPage<FineTuningJobEvent> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    FineTuningJobEventsList list;
    try {
      list = parse_events_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse fine-tuning job events response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<FineTuningJobEvent>(std::move(list.data),
                                          list.has_more,
                                          std::move(cursor),
                                          request_options,
                                          *fetch_impl,
                                          "after",
                                          std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = job_events_path(job_id);
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

CursorPage<FineTuningJobEvent> FineTuningJobsResource::list_events_page(const std::string& job_id) const {
  return list_events_page(job_id, JobListEventsParams{});
}

GraderRunResponse FineTuningAlphaGradersResource::run(const GraderRunParams& params) const {
  return run(params, RequestOptions{});
}

GraderRunResponse FineTuningAlphaGradersResource::run(const GraderRunParams& params,
                                                      const RequestOptions& options) const {
  json body = json::object();
  body["grader"] = graders_variant_to_json(params.grader);
  body["model_sample"] = params.model_sample;
  if (params.item) body["item"] = *params.item;

  auto response = client_.perform_request("POST", "/fine_tuning/alpha/graders/run", body.dump(), options);
  try {
    return parse_grader_run_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse grader run response: ") + ex.what());
  }
}

GraderValidateResponse FineTuningAlphaGradersResource::validate(const GraderValidateParams& params) const {
  return validate(params, RequestOptions{});
}

GraderValidateResponse FineTuningAlphaGradersResource::validate(const GraderValidateParams& params,
                                                                const RequestOptions& options) const {
  json body = json::object();
  body["grader"] = graders_variant_to_json(params.grader);
  auto response = client_.perform_request("POST", "/fine_tuning/alpha/graders/validate", body.dump(), options);
  try {
    return parse_grader_validate_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse grader validate response: ") + ex.what());
  }
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::create(
    const std::string& checkpoint_id, const FineTuningCheckpointPermissionCreateParams& params) const {
  return create(checkpoint_id, params, RequestOptions{});
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::create(
    const std::string& checkpoint_id,
    const FineTuningCheckpointPermissionCreateParams& params,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  auto body = checkpoint_permission_create_to_json(params).dump();
  auto response =
      client_.perform_request("POST", checkpoint_permissions_path(checkpoint_id), body, request_options);
  try {
    return parse_checkpoint_permission_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse checkpoint permission create response: ") + ex.what());
  }
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::retrieve(
    const std::string& checkpoint_id, const FineTuningCheckpointPermissionRetrieveParams& params) const {
  return retrieve(checkpoint_id, params, RequestOptions{});
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::retrieve(
    const std::string& checkpoint_id,
    const FineTuningCheckpointPermissionRetrieveParams& params,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.project_id) request_options.query_params["project_id"] = *params.project_id;

  auto response = client_.perform_request("GET", checkpoint_permissions_path(checkpoint_id), "", request_options);
  try {
    return parse_checkpoint_permission_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse checkpoint permission retrieve response: ") + ex.what());
  }
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::retrieve(
    const std::string& checkpoint_id) const {
  return retrieve(checkpoint_id, FineTuningCheckpointPermissionRetrieveParams{});
}

FineTuningCheckpointPermissionList FineTuningJobCheckpointPermissionsResource::retrieve(
    const std::string& checkpoint_id, const RequestOptions& options) const {
  return retrieve(checkpoint_id, FineTuningCheckpointPermissionRetrieveParams{}, options);
}

FineTuningCheckpointPermissionDeleteResponse FineTuningJobCheckpointPermissionsResource::remove(
    const std::string& checkpoint_id, const std::string& permission_id) const {
  return remove(checkpoint_id, permission_id, RequestOptions{});
}

FineTuningCheckpointPermissionDeleteResponse FineTuningJobCheckpointPermissionsResource::remove(
    const std::string& checkpoint_id,
    const std::string& permission_id,
    const RequestOptions& options) const {
  RequestOptions request_options = options;
  auto response = client_.perform_request("DELETE",
                                          checkpoint_permissions_path(checkpoint_id) + "/" + permission_id,
                                          "",
                                          request_options);
  try {
    return parse_checkpoint_permission_delete_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse checkpoint permission delete response: ") + ex.what());
  }
}

FineTuningJobCheckpointList FineTuningJobCheckpointsResource::list(const std::string& job_id,
                                                                   const FineTuningCheckpointListParams& params) const {
  return list(job_id, params, RequestOptions{});
}

FineTuningJobCheckpointList FineTuningJobCheckpointsResource::list(const std::string& job_id,
                                                                   const FineTuningCheckpointListParams& params,
                                                                   const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.after) request_options.query_params["after"] = *params.after;

  auto response = client_.perform_request("GET", job_checkpoints_path(job_id), "", request_options);
  try {
    return parse_checkpoint_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse fine-tuning job checkpoints response: ") + ex.what());
  }
}

FineTuningJobCheckpointList FineTuningJobCheckpointsResource::list(const std::string& job_id) const {
  return list(job_id, FineTuningCheckpointListParams{});
}

FineTuningJobCheckpointList FineTuningJobCheckpointsResource::list(const std::string& job_id,
                                                                   const RequestOptions& options) const {
  return list(job_id, FineTuningCheckpointListParams{}, options);
}

}  // namespace openai

#include "openai/fine_tuning.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>

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

#include "openai/beta.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/assistants.hpp"
#include "openai/assistant_stream.hpp"
#include "openai/messages.hpp"
#include "openai/run_steps.hpp"
#include "openai/runs.hpp"
#include "openai/threads.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

json tool_to_json(const beta::RealtimeSessionTool& tool) {
  json object = json::object();
  object["type"] = tool.type;
  if (!tool.definition.is_null()) {
    object["definition"] = tool.definition;
  }
  return object;
}

json tracing_to_json(const beta::RealtimeSessionTracingConfiguration& tracing) {
  json body = json::object();
  if (tracing.name) body["name"] = *tracing.name;
  if (tracing.group_id) body["group_id"] = *tracing.group_id;
  if (!tracing.metadata.empty()) body["metadata"] = tracing.metadata;
  return body;
}

json turn_detection_to_json(const beta::RealtimeSessionTurnDetection& detection) {
  json body = json::object();
  body["type"] = detection.type;
  if (detection.threshold) body["threshold"] = *detection.threshold;
  return body;
}

json session_create_body(const beta::RealtimeSessionCreateParams& params) {
  json body = json::object();
  if (params.model) body["model"] = *params.model;
  if (params.voice) body["voice"] = *params.voice;
  if (params.modalities) body["modalities"] = *params.modalities;
  if (params.instructions) body["instructions"] = *params.instructions;
  if (params.max_response_output_tokens) body["max_response_output_tokens"] = *params.max_response_output_tokens;
  if (params.tool_choice) body["tool_choice"] = *params.tool_choice;
  if (!params.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : params.tools) {
      tools.push_back(tool_to_json(tool));
    }
    body["tools"] = std::move(tools);
  }
  if (params.input_audio_format) body["input_audio_format"] = *params.input_audio_format;
  if (params.output_audio_format) body["output_audio_format"] = *params.output_audio_format;
  if (params.temperature) body["temperature"] = *params.temperature;
  if (params.speed) body["speed"] = *params.speed;
  if (params.input_audio_noise_reduction) {
    json nr = json::object();
    if (params.input_audio_noise_reduction->type) {
      nr["type"] = *params.input_audio_noise_reduction->type;
    }
    body["input_audio_noise_reduction"] = std::move(nr);
  }
  if (params.input_audio_transcription) {
    json transcription = json::object();
    if (params.input_audio_transcription->model) transcription["model"] = *params.input_audio_transcription->model;
    if (params.input_audio_transcription->language) transcription["language"] = *params.input_audio_transcription->language;
    if (params.input_audio_transcription->prompt) transcription["prompt"] = *params.input_audio_transcription->prompt;
    body["input_audio_transcription"] = std::move(transcription);
  }
  if (params.tracing) body["tracing"] = tracing_to_json(*params.tracing);
  if (params.turn_detection) body["turn_detection"] = turn_detection_to_json(*params.turn_detection);
  return body;
}

beta::RealtimeSession parse_session(const json& payload) {
  beta::RealtimeSession session;
  session.raw = payload;
  session.id = payload.value("id", "");
  if (payload.contains("model") && payload.at("model").is_string()) session.model = payload.at("model").get<std::string>();
  if (payload.contains("client_secret") && payload.at("client_secret").is_string()) session.client_secret = payload.at("client_secret").get<std::string>();
  if (payload.contains("voice") && payload.at("voice").is_string()) session.voice = payload.at("voice").get<std::string>();
  if (payload.contains("modalities") && payload.at("modalities").is_array()) {
    session.modalities = payload.at("modalities").get<std::vector<std::string>>();
  }
  if (payload.contains("instructions") && payload.at("instructions").is_string()) session.instructions = payload.at("instructions").get<std::string>();
  if (payload.contains("max_response_output_tokens") && payload.at("max_response_output_tokens").is_number_integer()) {
    session.max_response_output_tokens = payload.at("max_response_output_tokens").get<int>();
  }
  if (payload.contains("tool_choice") && payload.at("tool_choice").is_string()) session.tool_choice = payload.at("tool_choice").get<std::string>();
  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
      beta::RealtimeSessionTool tool;
      tool.type = tool_json.value("type", "");
      if (tool_json.contains("definition")) tool.definition = tool_json.at("definition");
      session.tools.push_back(std::move(tool));
    }
  }
  if (payload.contains("input_audio_format") && payload.at("input_audio_format").is_string()) session.input_audio_format = payload.at("input_audio_format").get<std::string>();
  if (payload.contains("output_audio_format") && payload.at("output_audio_format").is_string()) session.output_audio_format = payload.at("output_audio_format").get<std::string>();
  if (payload.contains("temperature") && payload.at("temperature").is_number()) session.temperature = payload.at("temperature").get<double>();
  if (payload.contains("speed") && payload.at("speed").is_number()) session.speed = payload.at("speed").get<double>();
  if (payload.contains("input_audio_noise_reduction") && payload.at("input_audio_noise_reduction").is_object()) {
    beta::RealtimeSessionInputAudioNoiseReduction nr;
    if (payload.at("input_audio_noise_reduction").contains("type") && payload.at("input_audio_noise_reduction").at("type").is_string()) {
      nr.type = payload.at("input_audio_noise_reduction").at("type").get<std::string>();
    }
    session.input_audio_noise_reduction = nr;
  }
  if (payload.contains("input_audio_transcription") && payload.at("input_audio_transcription").is_object()) {
    beta::RealtimeSessionInputAudioTranscription transcription;
    const auto& t_json = payload.at("input_audio_transcription");
    if (t_json.contains("model") && t_json.at("model").is_string()) transcription.model = t_json.at("model").get<std::string>();
    if (t_json.contains("language") && t_json.at("language").is_string()) transcription.language = t_json.at("language").get<std::string>();
    if (t_json.contains("prompt") && t_json.at("prompt").is_string()) transcription.prompt = t_json.at("prompt").get<std::string>();
    session.input_audio_transcription = transcription;
  }
  if (payload.contains("tracing") && payload.at("tracing").is_object()) {
    beta::RealtimeSessionTracingConfiguration tracing;
    auto& tr = payload.at("tracing");
    if (tr.contains("name") && tr.at("name").is_string()) tracing.name = tr.at("name").get<std::string>();
    if (tr.contains("group_id") && tr.at("group_id").is_string()) tracing.group_id = tr.at("group_id").get<std::string>();
    if (tr.contains("metadata") && tr.at("metadata").is_object()) {
      tracing.metadata = tr.at("metadata").get<std::map<std::string, std::string>>();
    }
    session.tracing = tracing;
  }
  if (payload.contains("turn_detection") && payload.at("turn_detection").is_object()) {
    beta::RealtimeSessionTurnDetection detection;
    auto& td = payload.at("turn_detection");
    detection.type = td.value("type", "");
    if (td.contains("threshold") && td.at("threshold").is_number()) detection.threshold = td.at("threshold").get<double>();
    session.turn_detection = detection;
  }
  return session;
}

json transcription_session_create_body(const beta::RealtimeTranscriptionSessionCreateParams& params) {
  json body = json::object();
  if (params.client_secret) {
    json client_secret = json::object();
    if (params.client_secret->expires_at) {
      json expires = json::object();
      if (params.client_secret->expires_at->anchor) {
        expires["anchor"] = *params.client_secret->expires_at->anchor;
      }
      if (params.client_secret->expires_at->seconds) {
        expires["seconds"] = *params.client_secret->expires_at->seconds;
      }
      if (!expires.empty()) {
        client_secret["expires_at"] = std::move(expires);
      }
    }
    if (!client_secret.empty()) {
      body["client_secret"] = std::move(client_secret);
    }
  }
  if (params.include && !params.include->empty()) {
    body["include"] = *params.include;
  }
  if (params.input_audio_format) body["input_audio_format"] = *params.input_audio_format;
  if (params.input_audio_noise_reduction) {
    json nr = json::object();
    if (params.input_audio_noise_reduction->type) nr["type"] = *params.input_audio_noise_reduction->type;
    body["input_audio_noise_reduction"] = std::move(nr);
  }
  if (params.input_audio_transcription) {
    json transcription = json::object();
    if (params.input_audio_transcription->language) transcription["language"] = *params.input_audio_transcription->language;
    if (params.input_audio_transcription->model) transcription["model"] = *params.input_audio_transcription->model;
    if (params.input_audio_transcription->prompt) transcription["prompt"] = *params.input_audio_transcription->prompt;
    body["input_audio_transcription"] = std::move(transcription);
  }
  if (params.modalities) body["modalities"] = *params.modalities;
  if (params.turn_detection) {
    json detection = json::object();
    if (params.turn_detection->create_response) detection["create_response"] = *params.turn_detection->create_response;
    if (params.turn_detection->eagerness) detection["eagerness"] = *params.turn_detection->eagerness;
    if (params.turn_detection->interrupt_response) detection["interrupt_response"] = *params.turn_detection->interrupt_response;
    if (params.turn_detection->prefix_padding_ms) detection["prefix_padding_ms"] = *params.turn_detection->prefix_padding_ms;
    if (params.turn_detection->silence_duration_ms) detection["silence_duration_ms"] = *params.turn_detection->silence_duration_ms;
    if (params.turn_detection->threshold) detection["threshold"] = *params.turn_detection->threshold;
    if (params.turn_detection->type) detection["type"] = *params.turn_detection->type;
    if (!detection.empty()) body["turn_detection"] = std::move(detection);
  }
  return body;
}

beta::RealtimeTranscriptionSession parse_transcription_session(const json& payload) {
  beta::RealtimeTranscriptionSession session;
  session.raw = payload;
  if (payload.contains("client_secret") && payload.at("client_secret").is_object()) {
    const auto& cs_json = payload.at("client_secret");
    beta::RealtimeTranscriptionSessionClientSecret secret;
    secret.expires_at = cs_json.value("expires_at", 0);
    if (cs_json.contains("value") && cs_json.at("value").is_string()) {
      secret.value = cs_json.at("value").get<std::string>();
    }
    session.client_secret = secret;
  }
  if (payload.contains("input_audio_format") && payload.at("input_audio_format").is_string()) {
    session.input_audio_format = payload.at("input_audio_format").get<std::string>();
  }
  if (payload.contains("input_audio_transcription") && payload.at("input_audio_transcription").is_object()) {
    const auto& transcription_json = payload.at("input_audio_transcription");
    beta::RealtimeTranscriptionSessionInputAudioTranscription transcription;
    if (transcription_json.contains("language") && transcription_json.at("language").is_string()) {
      transcription.language = transcription_json.at("language").get<std::string>();
    }
    if (transcription_json.contains("model") && transcription_json.at("model").is_string()) {
      transcription.model = transcription_json.at("model").get<std::string>();
    }
    if (transcription_json.contains("prompt") && transcription_json.at("prompt").is_string()) {
      transcription.prompt = transcription_json.at("prompt").get<std::string>();
    }
    session.input_audio_transcription = transcription;
  }
  if (payload.contains("modalities") && payload.at("modalities").is_array()) {
    session.modalities = payload.at("modalities").get<std::vector<std::string>>();
  }
  if (payload.contains("turn_detection") && payload.at("turn_detection").is_object()) {
    const auto& detection_json = payload.at("turn_detection");
    beta::RealtimeTranscriptionSessionTurnDetection detection;
    if (detection_json.contains("prefix_padding_ms") && detection_json.at("prefix_padding_ms").is_number_integer()) {
      detection.prefix_padding_ms = detection_json.at("prefix_padding_ms").get<int>();
    }
    if (detection_json.contains("silence_duration_ms") && detection_json.at("silence_duration_ms").is_number_integer()) {
      detection.silence_duration_ms = detection_json.at("silence_duration_ms").get<int>();
    }
    if (detection_json.contains("threshold") && detection_json.at("threshold").is_number()) {
      detection.threshold = detection_json.at("threshold").get<double>();
    }
    if (detection_json.contains("type") && detection_json.at("type").is_string()) {
      detection.type = detection_json.at("type").get<std::string>();
    }
    session.turn_detection = detection;
  }
  return session;
}

}  // namespace

beta::RealtimeSession beta::RealtimeSessionsResource::create(const RealtimeSessionCreateParams& params) const {
  return create(params, RequestOptions{});
}

beta::RealtimeSession beta::RealtimeSessionsResource::create(const RealtimeSessionCreateParams& params,
                                                             const RequestOptions& options) const {
  auto body = session_create_body(params).dump();
  RequestOptions request_options = options;
  request_options.headers["OpenAI-Beta"] = "assistants=v2";
  auto response = client_.perform_request("POST", "/realtime/sessions", body, request_options);
  try {
    return parse_session(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse realtime session response: ") + ex.what());
  }
}

beta::RealtimeSession beta::RealtimeSessionsResource::create() const {
  return create(RealtimeSessionCreateParams{});
}

beta::RealtimeTranscriptionSession beta::RealtimeTranscriptionSessionsResource::create(
    const RealtimeTranscriptionSessionCreateParams& params) const {
  return create(params, RequestOptions{});
}

beta::RealtimeTranscriptionSession beta::RealtimeTranscriptionSessionsResource::create(
    const RealtimeTranscriptionSessionCreateParams& params,
    const RequestOptions& options) const {
  auto body = transcription_session_create_body(params).dump();
  RequestOptions request_options = options;
  request_options.headers["OpenAI-Beta"] = "assistants=v2";
  auto response =
      client_.perform_request("POST", "/realtime/transcription_sessions", body, request_options);
  try {
    return parse_transcription_session(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse realtime transcription session response: ") + ex.what());
  }
}

beta::RealtimeTranscriptionSession beta::RealtimeTranscriptionSessionsResource::create() const {
  return create(RealtimeTranscriptionSessionCreateParams{});
}

namespace beta {

Thread BetaThreadsResource::create(const ThreadCreateRequest& request) const {
  return client_.threads().create(request);
}

Thread BetaThreadsResource::create(const ThreadCreateRequest& request, const RequestOptions& options) const {
  return client_.threads().create(request, options);
}

Thread BetaThreadsResource::retrieve(const std::string& thread_id) const {
  return client_.threads().retrieve(thread_id);
}

Thread BetaThreadsResource::retrieve(const std::string& thread_id, const RequestOptions& options) const {
  return client_.threads().retrieve(thread_id, options);
}

Thread BetaThreadsResource::update(const std::string& thread_id, const ThreadUpdateRequest& request) const {
  return client_.threads().update(thread_id, request);
}

Thread BetaThreadsResource::update(const std::string& thread_id,
                                   const ThreadUpdateRequest& request,
                                   const RequestOptions& options) const {
  return client_.threads().update(thread_id, request, options);
}

ThreadDeleteResponse BetaThreadsResource::remove(const std::string& thread_id) const {
  return client_.threads().remove(thread_id);
}

ThreadDeleteResponse BetaThreadsResource::remove(const std::string& thread_id, const RequestOptions& options) const {
  return client_.threads().remove(thread_id, options);
}

Run BetaThreadsResource::create_and_run(const ThreadCreateAndRunRequest& request) const {
  return client_.threads().create_and_run(request);
}

Run BetaThreadsResource::create_and_run(const ThreadCreateAndRunRequest& request,
                                        const RequestOptions& options) const {
  return client_.threads().create_and_run(request, options);
}

std::vector<AssistantStreamEvent> BetaThreadsResource::create_and_run_stream(
    const ThreadCreateAndRunRequest& request) const {
  return client_.threads().create_and_run_stream(request);
}

std::vector<AssistantStreamEvent> BetaThreadsResource::create_and_run_stream(
    const ThreadCreateAndRunRequest& request,
    const RequestOptions& options) const {
  return client_.threads().create_and_run_stream(request, options);
}

AssistantStreamSnapshot BetaThreadsResource::create_and_run_stream_snapshot(
    const ThreadCreateAndRunRequest& request) const {
  return client_.threads().create_and_run_stream_snapshot(request);
}

AssistantStreamSnapshot BetaThreadsResource::create_and_run_stream_snapshot(
    const ThreadCreateAndRunRequest& request,
    const RequestOptions& options) const {
  return client_.threads().create_and_run_stream_snapshot(request, options);
}

Run BetaThreadsResource::create_and_run_poll(const ThreadCreateAndRunRequest& request) const {
  return client_.threads().create_and_run_poll(request);
}

Run BetaThreadsResource::create_and_run_poll(const ThreadCreateAndRunRequest& request,
                                             const RequestOptions& options,
                                             std::chrono::milliseconds poll_interval) const {
  return client_.threads().create_and_run_poll(request, options, poll_interval);
}

ThreadMessagesResource& BetaThreadsResource::messages() {
  return client_.thread_messages();
}

const ThreadMessagesResource& BetaThreadsResource::messages() const {
  return client_.thread_messages();
}

RunsResource& BetaThreadsResource::runs() {
  return client_.runs();
}

const RunsResource& BetaThreadsResource::runs() const {
  return client_.runs();
}

RunStepsResource& BetaThreadsResource::run_steps() {
  return client_.run_steps();
}

const RunStepsResource& BetaThreadsResource::run_steps() const {
  return client_.run_steps();
}

}  // namespace beta

AssistantsResource& BetaResource::assistants() {
  return client_.assistants();
}

const AssistantsResource& BetaResource::assistants() const {
  return client_.assistants();
}

}  // namespace openai

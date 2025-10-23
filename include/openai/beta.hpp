#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/assistant_stream.hpp"
#include "openai/runs.hpp"
#include "openai/thread_types.hpp"

namespace openai {

class OpenAIClient;
struct RequestOptions;
struct Thread;
struct ThreadCreateRequest;
struct ThreadUpdateRequest;
struct ThreadDeleteResponse;
struct ThreadCreateAndRunRequest;
struct AssistantStreamSnapshot;

class AssistantsResource;
class ThreadsResource;
class ThreadMessagesResource;
class RunsResource;
class RunStepsResource;

namespace beta {

struct RealtimeSessionInputAudioNoiseReduction {
  std::optional<std::string> type;
};

struct RealtimeSessionInputAudioTranscription {
  std::optional<std::string> model;
  std::optional<std::string> language;
  std::optional<std::string> prompt;
};

struct RealtimeSessionTool {
  std::string type;
  nlohmann::json definition = nlohmann::json::object();
};

struct RealtimeSessionTracingConfiguration {
  std::optional<std::string> name;
  std::optional<std::string> group_id;
  std::map<std::string, std::string> metadata;
};

struct RealtimeSessionTurnDetection {
  std::string type;
  std::optional<double> threshold;
};

struct RealtimeSession {
  std::string id;
  std::optional<std::string> model;
  std::optional<std::string> client_secret;
  std::optional<std::string> voice;
  std::optional<std::vector<std::string>> modalities;
  std::optional<std::string> instructions;
  std::optional<int> max_response_output_tokens;
  std::optional<std::string> tool_choice;
  std::vector<RealtimeSessionTool> tools;
  std::optional<std::string> input_audio_format;
  std::optional<std::string> output_audio_format;
  std::optional<double> temperature;
  std::optional<double> speed;
  std::optional<RealtimeSessionInputAudioNoiseReduction> input_audio_noise_reduction;
  std::optional<RealtimeSessionInputAudioTranscription> input_audio_transcription;
  std::optional<RealtimeSessionTracingConfiguration> tracing;
  std::optional<RealtimeSessionTurnDetection> turn_detection;
  nlohmann::json raw = nlohmann::json::object();
};

struct RealtimeSessionCreateParams {
  std::optional<std::string> model;
  std::optional<std::string> voice;
  std::optional<std::vector<std::string>> modalities;
  std::optional<std::string> instructions;
  std::optional<int> max_response_output_tokens;
  std::optional<std::string> tool_choice;
  std::vector<RealtimeSessionTool> tools;
  std::optional<std::string> input_audio_format;
  std::optional<std::string> output_audio_format;
  std::optional<double> temperature;
  std::optional<double> speed;
  std::optional<RealtimeSessionInputAudioNoiseReduction> input_audio_noise_reduction;
  std::optional<RealtimeSessionInputAudioTranscription> input_audio_transcription;
  std::optional<RealtimeSessionTracingConfiguration> tracing;
  std::optional<RealtimeSessionTurnDetection> turn_detection;
};

struct RealtimeTranscriptionSessionClientSecret {
  int expires_at = 0;
  std::string value;
};

struct RealtimeTranscriptionSessionInputAudioTranscription {
  std::optional<std::string> language;
  std::optional<std::string> model;
  std::optional<std::string> prompt;
};

struct RealtimeTranscriptionSessionTurnDetection {
  std::optional<int> prefix_padding_ms;
  std::optional<int> silence_duration_ms;
  std::optional<double> threshold;
  std::optional<std::string> type;
};

struct RealtimeTranscriptionSession {
  std::optional<RealtimeTranscriptionSessionClientSecret> client_secret;
  std::optional<std::string> input_audio_format;
  std::optional<RealtimeTranscriptionSessionInputAudioTranscription> input_audio_transcription;
  std::optional<std::vector<std::string>> modalities;
  std::optional<RealtimeTranscriptionSessionTurnDetection> turn_detection;
  nlohmann::json raw = nlohmann::json::object();
};

struct RealtimeTranscriptionSessionCreateClientSecretExpiresAt {
  std::optional<std::string> anchor;
  std::optional<int> seconds;
};

struct RealtimeTranscriptionSessionCreateClientSecret {
  std::optional<RealtimeTranscriptionSessionCreateClientSecretExpiresAt> expires_at;
};

struct RealtimeTranscriptionSessionCreateInputAudioNoiseReduction {
  std::optional<std::string> type;
};

struct RealtimeTranscriptionSessionCreateInputAudioTranscription {
  std::optional<std::string> language;
  std::optional<std::string> model;
  std::optional<std::string> prompt;
};

struct RealtimeTranscriptionSessionCreateTurnDetection {
  std::optional<bool> create_response;
  std::optional<std::string> eagerness;
  std::optional<bool> interrupt_response;
  std::optional<int> prefix_padding_ms;
  std::optional<int> silence_duration_ms;
  std::optional<double> threshold;
  std::optional<std::string> type;
};

struct RealtimeTranscriptionSessionCreateParams {
  std::optional<RealtimeTranscriptionSessionCreateClientSecret> client_secret;
  std::optional<std::vector<std::string>> include;
  std::optional<std::string> input_audio_format;
  std::optional<RealtimeTranscriptionSessionCreateInputAudioNoiseReduction> input_audio_noise_reduction;
  std::optional<RealtimeTranscriptionSessionCreateInputAudioTranscription> input_audio_transcription;
  std::optional<std::vector<std::string>> modalities;
  std::optional<RealtimeTranscriptionSessionCreateTurnDetection> turn_detection;
};

class RealtimeSessionsResource {
public:
  explicit RealtimeSessionsResource(OpenAIClient& client) : client_(client) {}

  RealtimeSession create(const RealtimeSessionCreateParams& params) const;
  RealtimeSession create(const RealtimeSessionCreateParams& params, const RequestOptions& options) const;
  RealtimeSession create() const;

private:
  OpenAIClient& client_;
};

class RealtimeTranscriptionSessionsResource {
public:
  explicit RealtimeTranscriptionSessionsResource(OpenAIClient& client) : client_(client) {}

  RealtimeTranscriptionSession create(const RealtimeTranscriptionSessionCreateParams& params) const;
  RealtimeTranscriptionSession create(const RealtimeTranscriptionSessionCreateParams& params,
                                      const RequestOptions& options) const;
  RealtimeTranscriptionSession create() const;

private:
  OpenAIClient& client_;
};

class RealtimeResource {
public:
  explicit RealtimeResource(OpenAIClient& client) : sessions_(client), transcription_sessions_(client) {}

  RealtimeSessionsResource& sessions() { return sessions_; }
  const RealtimeSessionsResource& sessions() const { return sessions_; }

  RealtimeTranscriptionSessionsResource& transcription_sessions() { return transcription_sessions_; }
  const RealtimeTranscriptionSessionsResource& transcription_sessions() const { return transcription_sessions_; }

private:
  RealtimeSessionsResource sessions_;
  RealtimeTranscriptionSessionsResource transcription_sessions_;
};

class BetaThreadsResource {
public:
  explicit BetaThreadsResource(OpenAIClient& client) : client_(client) {}

  Thread create(const ThreadCreateRequest& request) const;
  Thread create(const ThreadCreateRequest& request, const RequestOptions& options) const;

  Thread retrieve(const std::string& thread_id) const;
  Thread retrieve(const std::string& thread_id, const RequestOptions& options) const;

  Thread update(const std::string& thread_id, const ThreadUpdateRequest& request) const;
  Thread update(const std::string& thread_id,
                const ThreadUpdateRequest& request,
                const RequestOptions& options) const;

  ThreadDeleteResponse remove(const std::string& thread_id) const;
  ThreadDeleteResponse remove(const std::string& thread_id, const RequestOptions& options) const;

  Run create_and_run(const ThreadCreateAndRunRequest& request) const;
  Run create_and_run(const ThreadCreateAndRunRequest& request, const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> create_and_run_stream(const ThreadCreateAndRunRequest& request) const;
  std::vector<AssistantStreamEvent> create_and_run_stream(const ThreadCreateAndRunRequest& request,
                                                          const RequestOptions& options) const;

  AssistantStreamSnapshot create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request) const;
  AssistantStreamSnapshot create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request,
                                                         const RequestOptions& options) const;

  Run create_and_run_poll(const ThreadCreateAndRunRequest& request) const;
  Run create_and_run_poll(const ThreadCreateAndRunRequest& request,
                          const RequestOptions& options,
                          std::chrono::milliseconds poll_interval) const;

  ThreadMessagesResource& messages();
  const ThreadMessagesResource& messages() const;

  RunsResource& runs();
  const RunsResource& runs() const;

  RunStepsResource& run_steps();
  const RunStepsResource& run_steps() const;

private:
  OpenAIClient& client_;
};

}  // namespace beta

class BetaResource {
public:
  explicit BetaResource(OpenAIClient& client) : client_(client), realtime_(client), threads_(client) {}

  AssistantsResource& assistants();
  const AssistantsResource& assistants() const;

  beta::BetaThreadsResource& threads() { return threads_; }
  const beta::BetaThreadsResource& threads() const { return threads_; }

  beta::RealtimeResource& realtime() { return realtime_; }
  const beta::RealtimeResource& realtime() const { return realtime_; }

private:
  OpenAIClient& client_;
  beta::RealtimeResource realtime_;
  beta::BetaThreadsResource threads_;
};

}  // namespace openai

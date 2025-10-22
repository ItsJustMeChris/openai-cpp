#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

class OpenAIClient;
struct RequestOptions;

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

class RealtimeSessionsResource {
public:
  explicit RealtimeSessionsResource(OpenAIClient& client) : client_(client) {}

  RealtimeSession create(const RealtimeSessionCreateParams& params) const;
  RealtimeSession create(const RealtimeSessionCreateParams& params, const RequestOptions& options) const;
  RealtimeSession create() const;

private:
  OpenAIClient& client_;
};

class RealtimeResource {
public:
  explicit RealtimeResource(OpenAIClient& client) : sessions_(client) {}

  RealtimeSessionsResource& sessions() { return sessions_; }
  const RealtimeSessionsResource& sessions() const { return sessions_; }

private:
  RealtimeSessionsResource sessions_;
};

}  // namespace beta

class BetaResource {
public:
  explicit BetaResource(OpenAIClient& client) : client_(client), realtime_(client) {}

  beta::RealtimeResource& realtime() { return realtime_; }
  const beta::RealtimeResource& realtime() const { return realtime_; }

private:
  OpenAIClient& client_;
  beta::RealtimeResource realtime_;
};

}  // namespace openai


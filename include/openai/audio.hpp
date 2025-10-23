#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/files.hpp"

namespace openai {

struct TranscriptionUsageInputTokenDetails {
  std::optional<int> audio_tokens;
  std::optional<int> text_tokens;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionUsage {
  enum class Type { Tokens, Duration, Unknown };
  Type type = Type::Unknown;
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  std::optional<TranscriptionUsageInputTokenDetails> input_token_details;
  double seconds = 0.0;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionLogprob {
  std::optional<std::string> token;
  std::optional<std::vector<int>> bytes;
  std::optional<double> logprob;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionSegment {
  int id = 0;
  double avg_logprob = 0.0;
  double compression_ratio = 0.0;
  double end = 0.0;
  double no_speech_prob = 0.0;
  int seek = 0;
  double start = 0.0;
  double temperature = 0.0;
  std::string text;
  std::vector<int> tokens;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionWord {
  double end = 0.0;
  double start = 0.0;
  std::string word;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionDiarizedSegment {
  std::string id;
  double end = 0.0;
  std::string speaker;
  double start = 0.0;
  std::string text;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionResponse {
  std::string text;
  std::optional<TranscriptionUsage> usage;
  std::optional<std::vector<TranscriptionLogprob>> logprobs;
  std::optional<std::vector<TranscriptionSegment>> segments;
  std::optional<std::vector<TranscriptionWord>> words;
  std::optional<std::vector<TranscriptionDiarizedSegment>> diarized_segments;
  std::optional<double> duration;
  std::optional<std::string> language;
  std::optional<std::string> task;
  bool is_diarized = false;
  bool is_verbose = false;
  bool is_plain_text = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionChunkingStrategy {
  enum class Type { Auto, ServerVad };
  Type type = Type::Auto;
  std::optional<int> prefix_padding_ms;
  std::optional<int> silence_duration_ms;
  std::optional<double> threshold;
};

struct TranscriptionRequest {
  FileUploadRequest file;
  std::string model;
  std::optional<TranscriptionChunkingStrategy> chunking_strategy;
  std::optional<std::vector<std::string>> include;
  std::optional<std::vector<std::string>> known_speaker_names;
  std::optional<std::vector<std::string>> known_speaker_references;
  std::optional<std::string> language;
  std::optional<std::string> prompt;
  std::optional<std::string> response_format;
  std::optional<bool> stream;
  std::optional<double> temperature;
  std::optional<std::vector<std::string>> timestamp_granularities;
};

struct TranslationResponse {
  std::string text;
  std::optional<double> duration;
  std::optional<std::string> language;
  std::optional<std::vector<TranscriptionSegment>> segments;
  bool is_verbose = false;
  bool is_plain_text = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranslationRequest {
  FileUploadRequest file;
  std::string model;
  std::optional<std::string> prompt;
  std::optional<std::string> response_format;
  std::optional<double> temperature;
};

struct SpeechResponse {
  std::vector<std::uint8_t> audio;
  std::map<std::string, std::string> headers;
};

struct SpeechRequest {
  std::string input;
  std::string model;
  std::string voice;
  std::optional<std::string> instructions;
  std::optional<std::string> response_format;
  std::optional<double> speed;
  std::optional<std::string> stream_format;
};

struct RequestOptions;
class OpenAIClient;

class AudioTranscriptionsResource {
public:
  explicit AudioTranscriptionsResource(OpenAIClient& client) : client_(client) {}

  TranscriptionResponse create(const TranscriptionRequest& request) const;
  TranscriptionResponse create(const TranscriptionRequest& request, const RequestOptions& options) const;

  TranslationResponse translate(const TranslationRequest& request) const;
  TranslationResponse translate(const TranslationRequest& request, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class AudioTranslationsResource {
public:
  explicit AudioTranslationsResource(OpenAIClient& client) : client_(client) {}

  TranslationResponse create(const TranslationRequest& request) const;
  TranslationResponse create(const TranslationRequest& request, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class AudioSpeechResource {
public:
  explicit AudioSpeechResource(OpenAIClient& client) : client_(client) {}

  SpeechResponse create(const SpeechRequest& request) const;
  SpeechResponse create(const SpeechRequest& request, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class AudioResource {
public:
  explicit AudioResource(OpenAIClient& client)
      : client_(client), transcriptions_(client), translations_(client), speech_(client) {}

  AudioTranscriptionsResource& transcriptions() { return transcriptions_; }
  const AudioTranscriptionsResource& transcriptions() const { return transcriptions_; }

  AudioTranslationsResource& translations() { return translations_; }
  const AudioTranslationsResource& translations() const { return translations_; }

  AudioSpeechResource& speech() { return speech_; }
  const AudioSpeechResource& speech() const { return speech_; }

private:
  OpenAIClient& client_;
  AudioTranscriptionsResource transcriptions_;
  AudioTranslationsResource translations_;
  AudioSpeechResource speech_;
};

}  // namespace openai

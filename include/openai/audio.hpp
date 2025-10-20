#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/files.hpp"

namespace openai {

struct TranscriptionTokens {
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  nlohmann::json extra = nlohmann::json::object();
};

struct TranscriptionResponse {
  std::string text;
  std::optional<TranscriptionTokens> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranscriptionRequest {
  FileUploadRequest file;
  std::string model;
  std::optional<std::string> response_format;
  std::optional<std::string> language;
  nlohmann::json extra = nlohmann::json::object();
};

struct TranslationResponse {
  std::string text;
  nlohmann::json raw = nlohmann::json::object();
};

struct TranslationRequest {
  FileUploadRequest file;
  std::string model;
  std::optional<std::string> prompt;
  std::optional<std::string> response_format;
  std::optional<double> temperature;
  nlohmann::json extra = nlohmann::json::object();
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
  nlohmann::json extra = nlohmann::json::object();
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

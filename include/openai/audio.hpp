#pragma once

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

struct RequestOptions;
class OpenAIClient;

class AudioTranscriptionsResource {
public:
  explicit AudioTranscriptionsResource(OpenAIClient& client) : client_(client) {}

  TranscriptionResponse create(const TranscriptionRequest& request) const;
  TranscriptionResponse create(const TranscriptionRequest& request, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class AudioResource {
public:
  explicit AudioResource(OpenAIClient& client) : client_(client), transcriptions_(client) {}

  AudioTranscriptionsResource& transcriptions() { return transcriptions_; }
  const AudioTranscriptionsResource& transcriptions() const { return transcriptions_; }

private:
  OpenAIClient& client_;
  AudioTranscriptionsResource transcriptions_;
};

}  // namespace openai


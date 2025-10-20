#include "openai/audio.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kAudioTranscriptions = "/audio/transcriptions";
constexpr const char* kAudioTranslations = "/audio/translations";
constexpr const char* kAudioSpeech = "/audio/speech";
constexpr const char* kBoundary = "----openai-cpp-audio-boundary";

std::string load_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void append_field(std::ostringstream& body, const std::string& name, const std::string& value) {
  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"" << name << "\"\r\n\r\n";
  body << value << "\r\n";
}

std::string build_transcription_multipart(const TranscriptionRequest& request) {
  std::ostringstream body;
  const auto file_content = load_file(request.file.file_path);
  const std::string filename = request.file.file_name.value_or("audio.wav");
  const std::string content_type = request.file.content_type.value_or("application/octet-stream");

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n";
  body << "Content-Type: " << content_type << "\r\n\r\n";
  body.write(file_content.data(), static_cast<std::streamsize>(file_content.size()));
  body << "\r\n";

  append_field(body, "model", request.model);
  if (request.response_format) append_field(body, "response_format", *request.response_format);
  if (request.language) append_field(body, "language", *request.language);
  if (request.prompt) append_field(body, "prompt", *request.prompt);
  if (request.temperature) append_field(body, "temperature", std::to_string(*request.temperature));
  if (request.timestamp_granularities) {
    for (const auto& granularity : *request.timestamp_granularities) {
      append_field(body, "timestamp_granularities[]", granularity);
    }
  }
  if (request.logprobs) append_field(body, "logprobs", *request.logprobs ? "true" : "false");
  if (request.max_alternatives) append_field(body, "max_alternatives", std::to_string(*request.max_alternatives));
  if (request.profanity_filter) append_field(body, "profanity_filter", *request.profanity_filter ? "true" : "false");
  if (request.remove_background) append_field(body, "remove_background", *request.remove_background ? "true" : "false");
  if (request.speaker_labels) append_field(body, "speaker_labels", *request.speaker_labels ? "true" : "false");

  body << "--" << kBoundary << "--\r\n";
  return body.str();
}

std::string build_translation_multipart(const TranslationRequest& request) {
  std::ostringstream body;
  const auto file_content = load_file(request.file.file_path);
  const std::string filename = request.file.file_name.value_or("audio.wav");
  const std::string content_type = request.file.content_type.value_or("application/octet-stream");

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n";
  body << "Content-Type: " << content_type << "\r\n\r\n";
  body.write(file_content.data(), static_cast<std::streamsize>(file_content.size()));
  body << "\r\n";

  append_field(body, "model", request.model);
  if (request.prompt) append_field(body, "prompt", *request.prompt);
  if (request.response_format) append_field(body, "response_format", *request.response_format);
  if (request.temperature) append_field(body, "temperature", std::to_string(*request.temperature));

  body << "--" << kBoundary << "--\r\n";
  return body.str();
}

TranscriptionResponse parse_transcription(const json& payload) {
  TranscriptionResponse response;
  response.raw = payload;
  response.text = payload.value("text", "");
  if (payload.contains("usage") && payload["usage"].is_object()) {
    TranscriptionTokens tokens;
    const auto& usage = payload.at("usage");
    tokens.input_tokens = usage.value("input_tokens", 0);
    tokens.output_tokens = usage.value("output_tokens", 0);
    tokens.total_tokens = usage.value("total_tokens", 0);
    tokens.extra = usage;
    response.usage = tokens;
  }
  return response;
}

}  // namespace

TranscriptionResponse AudioTranscriptionsResource::create(const TranscriptionRequest& request,
                                                          const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + std::string(kBoundary);
  auto body = build_transcription_multipart(request);
  auto response = client_.perform_request("POST", kAudioTranscriptions, body, request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_transcription(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse transcription response: ") + ex.what());
  }
}

TranscriptionResponse AudioTranscriptionsResource::create(const TranscriptionRequest& request) const {
  return create(request, RequestOptions{});
}

TranslationResponse AudioTranslationsResource::create(const TranslationRequest& request,
                                                       const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + std::string(kBoundary);
  auto body = build_translation_multipart(request);
  auto response = client_.perform_request("POST", kAudioTranslations, body, request_options);

  TranslationResponse translation;
  try {
    auto payload = json::parse(response.body);
    translation.raw = payload;
    translation.text = payload.value("text", response.body);
  } catch (const json::exception&) {
    translation.text = response.body;
  }
  return translation;
}

TranslationResponse AudioTranslationsResource::create(const TranslationRequest& request) const {
  return create(request, RequestOptions{});
}

SpeechResponse AudioSpeechResource::create(const SpeechRequest& request, const RequestOptions& options) const {
  json body;
  body["input"] = request.input;
  body["model"] = request.model;
  body["voice"] = request.voice;
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.response_format) body["response_format"] = *request.response_format;
  if (request.speed) body["speed"] = *request.speed;
  if (request.stream_format) body["stream_format"] = *request.stream_format;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "application/octet-stream";
  auto response = client_.perform_request("POST", kAudioSpeech, body.dump(), request_options);

  SpeechResponse speech;
  speech.headers = response.headers;
  speech.audio.assign(response.body.begin(), response.body.end());
  return speech;
}

SpeechResponse AudioSpeechResource::create(const SpeechRequest& request) const {
  return create(request, RequestOptions{});
}

}  // namespace openai


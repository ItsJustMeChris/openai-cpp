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
constexpr const char* kBoundary = "----openai-cpp-audio-boundary";

std::string load_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string build_multipart(const TranscriptionRequest& request) {
  std::ostringstream body;
  const auto file_content = load_file(request.file.file_path);
  const std::string filename = request.file.file_name.value_or("audio.wav");
  const std::string content_type = request.file.content_type.value_or("application/octet-stream");

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n";
  body << "Content-Type: " << content_type << "\r\n\r\n";
  body.write(file_content.data(), static_cast<std::streamsize>(file_content.size()));
  body << "\r\n";

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"model\"\r\n\r\n";
  body << request.model << "\r\n";

  if (request.response_format) {
    body << "--" << kBoundary << "\r\n";
    body << "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n";
    body << *request.response_format << "\r\n";
  }

  if (request.language) {
    body << "--" << kBoundary << "\r\n";
    body << "Content-Disposition: form-data; name=\"language\"\r\n\r\n";
    body << *request.language << "\r\n";
  }

  if (request.extra.is_object()) {
    for (auto it = request.extra.begin(); it != request.extra.end(); ++it) {
      body << "--" << kBoundary << "\r\n";
      body << "Content-Disposition: form-data; name=\"" << it.key() << "\"\r\n\r\n";
      body << it.value().dump() << "\r\n";
    }
  }

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
  auto body = build_multipart(request);
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

}  // namespace openai


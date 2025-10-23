#include "openai/audio.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <iomanip>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kAudioTranscriptions = "/audio/transcriptions";
constexpr const char* kAudioTranslations = "/audio/translations";
constexpr const char* kAudioSpeech = "/audio/speech";
constexpr const char* kBoundary = "----openai-cpp-audio-boundary";

void append_field(std::ostringstream& body, const std::string& name, const std::string& value) {
  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"" << name << "\"\r\n\r\n";
  body << value << "\r\n";
}

void append_form_value(std::ostringstream& body, const std::string& key, const json& value) {
  if (value.is_null()) {
    return;
  }

  if (value.is_object()) {
    for (const auto& [name, child] : value.items()) {
      append_form_value(body, key + "[" + name + "]", child);
    }
    return;
  }

  if (value.is_array()) {
    for (const auto& child : value) {
      append_form_value(body, key + "[]", child);
    }
    return;
  }

  if (value.is_string()) {
    append_field(body, key, value.get<std::string>());
    return;
  }

  if (value.is_boolean()) {
    append_field(body, key, value.get<bool>() ? "true" : "false");
    return;
  }

  if (value.is_number_integer()) {
    append_field(body, key, std::to_string(value.get<long long>()));
    return;
  }

  if (value.is_number_unsigned()) {
    append_field(body, key, std::to_string(value.get<unsigned long long>()));
    return;
  }

  if (value.is_number_float()) {
    std::ostringstream oss;
    oss << std::setprecision(15) << value.get<double>();
    append_field(body, key, oss.str());
    return;
  }

  append_field(body, key, value.dump());
}

std::optional<TranscriptionUsage> parse_usage(const json& usage_json) {
  if (!usage_json.is_object()) {
    return std::nullopt;
  }

  TranscriptionUsage usage;
  usage.raw = usage_json;

  const std::string type = usage_json.value("type", "");
  const bool has_token_fields = usage_json.contains("input_tokens") || usage_json.contains("output_tokens") ||
                                usage_json.contains("total_tokens") || type == "tokens";

  if (has_token_fields) {
    usage.type = TranscriptionUsage::Type::Tokens;
    usage.input_tokens = usage_json.value("input_tokens", 0);
    usage.output_tokens = usage_json.value("output_tokens", 0);
    usage.total_tokens = usage_json.value("total_tokens", usage.input_tokens + usage.output_tokens);

    if (usage_json.contains("input_token_details") && usage_json["input_token_details"].is_object()) {
      TranscriptionUsageInputTokenDetails details;
      const auto& details_json = usage_json["input_token_details"];
      if (details_json.contains("audio_tokens") && details_json["audio_tokens"].is_number()) {
        details.audio_tokens = details_json["audio_tokens"].get<int>();
      }
      if (details_json.contains("text_tokens") && details_json["text_tokens"].is_number()) {
        details.text_tokens = details_json["text_tokens"].get<int>();
      }
      details.raw = details_json;
      usage.input_token_details = details;
    }

    return usage;
  }

  if (type == "duration" || usage_json.contains("seconds")) {
    usage.type = TranscriptionUsage::Type::Duration;
    usage.seconds = usage_json.value("seconds", 0.0);
    return usage;
  }

  usage.type = TranscriptionUsage::Type::Unknown;
  return usage;
}

std::vector<TranscriptionLogprob> parse_logprobs(const json& array) {
  std::vector<TranscriptionLogprob> result;
  for (const auto& entry : array) {
    if (!entry.is_object()) {
      continue;
    }
    TranscriptionLogprob logprob;
    logprob.raw = entry;
    if (entry.contains("token") && entry["token"].is_string()) {
      logprob.token = entry["token"].get<std::string>();
    }
    if (entry.contains("bytes") && entry["bytes"].is_array()) {
      std::vector<int> bytes;
      for (const auto& value : entry["bytes"]) {
        if (value.is_number_integer()) {
          bytes.push_back(value.get<int>());
        } else if (value.is_number_unsigned()) {
          bytes.push_back(static_cast<int>(value.get<unsigned long long>()));
        }
      }
      logprob.bytes = std::move(bytes);
    }
    if (entry.contains("logprob") && entry["logprob"].is_number()) {
      logprob.logprob = entry["logprob"].get<double>();
    }
    result.push_back(std::move(logprob));
  }
  return result;
}

std::vector<TranscriptionSegment> parse_transcription_segments(const json& array) {
  std::vector<TranscriptionSegment> result;
  for (const auto& entry : array) {
    if (!entry.is_object()) {
      continue;
    }
    TranscriptionSegment segment;
    segment.raw = entry;
    segment.id = entry.value("id", 0);
    segment.avg_logprob = entry.value("avg_logprob", 0.0);
    segment.compression_ratio = entry.value("compression_ratio", 0.0);
    segment.end = entry.value("end", 0.0);
    segment.no_speech_prob = entry.value("no_speech_prob", 0.0);
    segment.seek = entry.value("seek", 0);
    segment.start = entry.value("start", 0.0);
    segment.temperature = entry.value("temperature", 0.0);
    segment.text = entry.value("text", "");
    if (entry.contains("tokens") && entry["tokens"].is_array()) {
      for (const auto& token_value : entry["tokens"]) {
        if (token_value.is_number_integer()) {
          segment.tokens.push_back(token_value.get<int>());
        } else if (token_value.is_number_unsigned()) {
          segment.tokens.push_back(static_cast<int>(token_value.get<unsigned long long>()));
        }
      }
    }
    result.push_back(std::move(segment));
  }
  return result;
}

std::vector<TranscriptionDiarizedSegment> parse_diarized_segments(const json& array) {
  std::vector<TranscriptionDiarizedSegment> result;
  for (const auto& entry : array) {
    if (!entry.is_object()) {
      continue;
    }
    TranscriptionDiarizedSegment segment;
    segment.raw = entry;
    if (entry.contains("id") && entry["id"].is_string()) {
      segment.id = entry["id"].get<std::string>();
    }
    segment.end = entry.value("end", 0.0);
    if (entry.contains("speaker") && entry["speaker"].is_string()) {
      segment.speaker = entry["speaker"].get<std::string>();
    }
    segment.start = entry.value("start", 0.0);
    if (entry.contains("text") && entry["text"].is_string()) {
      segment.text = entry["text"].get<std::string>();
    }
    result.push_back(std::move(segment));
  }
  return result;
}

std::vector<TranscriptionWord> parse_words(const json& array) {
  std::vector<TranscriptionWord> result;
  for (const auto& entry : array) {
    if (!entry.is_object()) {
      continue;
    }
    TranscriptionWord word;
    word.raw = entry;
    word.end = entry.value("end", 0.0);
    word.start = entry.value("start", 0.0);
    if (entry.contains("word") && entry["word"].is_string()) {
      word.word = entry["word"].get<std::string>();
    }
    result.push_back(std::move(word));
  }
  return result;
}

TranscriptionResponse parse_transcription_json(const json& payload) {
  TranscriptionResponse response;
  response.raw = payload;

  if (payload.is_string()) {
    response.text = payload.get<std::string>();
    response.is_plain_text = true;
    return response;
  }

  if (!payload.is_object()) {
    return response;
  }

  response.text = payload.value("text", "");

  if (payload.contains("usage")) {
    if (auto usage = parse_usage(payload["usage"])) {
      response.usage = *usage;
    }
  }

  if (payload.contains("logprobs") && payload["logprobs"].is_array()) {
    auto logprobs = parse_logprobs(payload["logprobs"]);
    if (!logprobs.empty()) {
      response.logprobs = std::move(logprobs);
    }
  }

  if (payload.contains("segments") && payload["segments"].is_array()) {
    const auto& segments_json = payload["segments"];
    bool diarized = false;
    if (!segments_json.empty() && segments_json.front().is_object()) {
      const auto& first = segments_json.front();
      diarized = first.contains("speaker") || first.value("type", "") == "transcript.text.segment";
    }
    if (diarized) {
      auto diarized_segments = parse_diarized_segments(segments_json);
      if (!diarized_segments.empty()) {
        response.diarized_segments = std::move(diarized_segments);
        response.is_diarized = true;
      }
    } else {
      auto segments = parse_transcription_segments(segments_json);
      if (!segments.empty()) {
        response.segments = std::move(segments);
        response.is_verbose = true;
      }
    }
  }

  if (payload.contains("words") && payload["words"].is_array()) {
    auto words = parse_words(payload["words"]);
    if (!words.empty()) {
      response.words = std::move(words);
      response.is_verbose = true;
    }
  }

  if (payload.contains("duration") && payload["duration"].is_number()) {
    response.duration = payload["duration"].get<double>();
    response.is_verbose = true;
  }

  if (payload.contains("language") && payload["language"].is_string()) {
    response.language = payload["language"].get<std::string>();
    response.is_verbose = true;
  }

  if (payload.contains("task") && payload["task"].is_string()) {
    response.task = payload["task"].get<std::string>();
    response.is_verbose = true;
  }

  if (response.is_diarized) {
    response.is_verbose = true;
  }

  return response;
}

TranscriptionResponse parse_transcription_body(const std::string& body) {
  try {
    auto payload = json::parse(body);
    return parse_transcription_json(payload);
  } catch (const json::exception&) {
    TranscriptionResponse fallback;
    fallback.text = body;
    fallback.is_plain_text = true;
    fallback.raw = body;
    return fallback;
  }
}

TranslationResponse parse_translation_body(const std::string& body) {
  TranslationResponse response;
  try {
    auto payload = json::parse(body);
    response.raw = payload;

    if (payload.is_string()) {
      response.text = payload.get<std::string>();
      response.is_plain_text = true;
      return response;
    }

    if (!payload.is_object()) {
      response.text = body;
      response.is_plain_text = true;
      return response;
    }

    response.text = payload.value("text", "");
    if (response.text.empty()) {
      response.text = body;
    }

    if (payload.contains("duration") && payload["duration"].is_number()) {
      response.duration = payload["duration"].get<double>();
      response.is_verbose = true;
    }

    if (payload.contains("language") && payload["language"].is_string()) {
      response.language = payload["language"].get<std::string>();
      response.is_verbose = true;
    }

    if (payload.contains("segments") && payload["segments"].is_array()) {
      auto segments = parse_transcription_segments(payload["segments"]);
      if (!segments.empty()) {
        response.segments = std::move(segments);
        response.is_verbose = true;
      }
    }
  } catch (const json::exception&) {
    response.text = body;
    response.is_plain_text = true;
    response.raw = body;
  }

  if (response.text.empty()) {
    response.text = body;
  }

  return response;
}

std::string build_transcription_multipart(const TranscriptionRequest& request) {
  std::ostringstream body;
  auto upload = request.file.materialize("audio.wav");

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << upload.filename << "\"\r\n";
  body << "Content-Type: " << *upload.content_type << "\r\n\r\n";
  body.write(reinterpret_cast<const char*>(upload.data.data()), static_cast<std::streamsize>(upload.data.size()));
  body << "\r\n";

  json fields = json::object();
  fields["model"] = request.model;

  if (request.chunking_strategy) {
    if (request.chunking_strategy->type == TranscriptionChunkingStrategy::Type::Auto) {
      fields["chunking_strategy"] = "auto";
    } else {
      json chunk = json::object();
      chunk["type"] = "server_vad";
      if (request.chunking_strategy->prefix_padding_ms) {
        chunk["prefix_padding_ms"] = *request.chunking_strategy->prefix_padding_ms;
      }
      if (request.chunking_strategy->silence_duration_ms) {
        chunk["silence_duration_ms"] = *request.chunking_strategy->silence_duration_ms;
      }
      if (request.chunking_strategy->threshold) {
        chunk["threshold"] = *request.chunking_strategy->threshold;
      }
      fields["chunking_strategy"] = chunk;
    }
  }

  if (request.include && !request.include->empty()) {
    fields["include"] = *request.include;
  }

  if (request.known_speaker_names && !request.known_speaker_names->empty()) {
    fields["known_speaker_names"] = *request.known_speaker_names;
  }

  if (request.known_speaker_references && !request.known_speaker_references->empty()) {
    fields["known_speaker_references"] = *request.known_speaker_references;
  }

  if (request.language) {
    fields["language"] = *request.language;
  }

  if (request.prompt) {
    fields["prompt"] = *request.prompt;
  }

  if (request.response_format) {
    fields["response_format"] = *request.response_format;
  }

  if (request.stream.has_value()) {
    fields["stream"] = *request.stream;
  }

  if (request.temperature) {
    fields["temperature"] = *request.temperature;
  }

  if (request.timestamp_granularities && !request.timestamp_granularities->empty()) {
    fields["timestamp_granularities"] = *request.timestamp_granularities;
  }

  for (const auto& [key, value] : fields.items()) {
    append_form_value(body, key, value);
  }

  body << "--" << kBoundary << "--\r\n";
  return body.str();
}

std::string build_translation_multipart(const TranslationRequest& request) {
  std::ostringstream body;
  auto upload = request.file.materialize("audio.wav");

  body << "--" << kBoundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << upload.filename << "\"\r\n";
  body << "Content-Type: " << *upload.content_type << "\r\n\r\n";
  body.write(reinterpret_cast<const char*>(upload.data.data()), static_cast<std::streamsize>(upload.data.size()));
  body << "\r\n";

  append_field(body, "model", request.model);
  if (request.prompt) append_field(body, "prompt", *request.prompt);
  if (request.response_format) append_field(body, "response_format", *request.response_format);
  if (request.temperature) append_field(body, "temperature", std::to_string(*request.temperature));

  body << "--" << kBoundary << "--\r\n";
  return body.str();
}

}  // namespace

TranscriptionResponse AudioTranscriptionsResource::create(const TranscriptionRequest& request,
                                                          const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + std::string(kBoundary);
  auto body = build_transcription_multipart(request);
  auto response = client_.perform_request("POST", kAudioTranscriptions, body, request_options);
  return parse_transcription_body(response.body);
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
  return parse_translation_body(response.body);
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


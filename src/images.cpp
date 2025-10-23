#include "openai/images.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

const char* kImagesGenerate = "/images/generations";
const char* kImagesVariation = "/images/variations";
const char* kImagesEdit = "/images/edits";

ImageUsage parse_image_usage_json(const json& usage_json) {
  ImageUsage usage;
  usage.raw = usage_json;
  usage.input_tokens = usage_json.value("input_tokens", 0);
  usage.output_tokens = usage_json.value("output_tokens", 0);
  usage.total_tokens = usage_json.value("total_tokens", 0);

  if (usage_json.contains("input_tokens_details") && usage_json.at("input_tokens_details").is_object()) {
    ImageUsageInputTokensDetails details;
    const auto& details_json = usage_json.at("input_tokens_details");
    details.raw = details_json;
    details.image_tokens = details_json.value("image_tokens", 0);
    details.text_tokens = details_json.value("text_tokens", 0);
    usage.input_tokens_details = std::move(details);
  }

  return usage;
}

ImagesResponse parse_images_response(const json& payload) {
  ImagesResponse response;
  response.raw = payload;
  response.created = payload.value("created", 0);

  if (payload.contains("background") && payload.at("background").is_string()) {
    response.background = payload.at("background").get<std::string>();
  }
  if (payload.contains("output_format") && payload.at("output_format").is_string()) {
    response.output_format = payload.at("output_format").get<std::string>();
  }
  if (payload.contains("quality") && payload.at("quality").is_string()) {
    response.quality = payload.at("quality").get<std::string>();
  }
  if (payload.contains("size") && payload.at("size").is_string()) {
    response.size = payload.at("size").get<std::string>();
  }
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      ImageData data;
      data.raw = item;
      if (item.contains("b64_json") && item.at("b64_json").is_string()) {
        data.b64_json = item.at("b64_json").get<std::string>();
      }
      if (item.contains("url") && item.at("url").is_string()) {
        data.url = item.at("url").get<std::string>();
      }
      if (item.contains("revised_prompt") && item.at("revised_prompt").is_string()) {
        data.revised_prompt = item.at("revised_prompt").get<std::string>();
      }
      response.data.push_back(std::move(data));
    }
  }
  if (payload.contains("usage") && payload.at("usage").is_object()) {
    response.usage = parse_image_usage_json(payload.at("usage"));
  }
  return response;
}

std::string materialize_file_as_string(const FileUploadRequest& request,
                                       const std::string& default_filename) {
  auto upload = request.materialize(default_filename);
  return std::string(upload.data.begin(), upload.data.end());
}

std::string build_multipart_body(const std::vector<std::pair<std::string, std::string>>& parts,
                                 const std::string& boundary) {
  std::string body;
  for (const auto& part : parts) {
    body += "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"" + part.first + "\"\r\n\r\n";
    body += part.second + "\r\n";
  }
  body += "--" + boundary + "--\r\n";
  return body;
}

std::string double_to_string(double value) {
  std::ostringstream oss;
  oss << std::setprecision(15) << value;
  return oss.str();
}

json build_generate_body(const ImageGenerateRequest& request, bool force_stream) {
  json body;
  body["prompt"] = request.prompt;
  if (request.model) body["model"] = *request.model;
  if (request.n) body["n"] = *request.n;
  if (request.size) body["size"] = *request.size;
  if (request.response_format) body["response_format"] = *request.response_format;
  if (request.quality) body["quality"] = *request.quality;
  if (request.style) body["style"] = *request.style;
  if (request.moderation) body["moderation"] = *request.moderation;
  if (request.output_compression) body["output_compression"] = *request.output_compression;
  if (request.output_format) body["output_format"] = *request.output_format;
  if (request.partial_images) body["partial_images"] = *request.partial_images;
  if (request.background) body["background"] = *request.background;
  if (request.user) body["user"] = *request.user;

  if (force_stream) {
    body["stream"] = true;
  } else if (request.stream.has_value()) {
    body["stream"] = *request.stream;
  }

  return body;
}

std::string build_variation_multipart(const ImageVariationRequest& request, const std::string& boundary) {
  std::vector<std::pair<std::string, std::string>> parts;
  parts.emplace_back("image", materialize_file_as_string(request.image, "image.bin"));
  if (request.model) parts.emplace_back("model", *request.model);
  if (request.prompt) parts.emplace_back("prompt", *request.prompt);
  if (request.background) parts.emplace_back("background", *request.background);
  if (request.n) parts.emplace_back("n", std::to_string(*request.n));
  if (request.size) parts.emplace_back("size", *request.size);
  if (request.response_format) parts.emplace_back("response_format", *request.response_format);
  if (request.quality) parts.emplace_back("quality", *request.quality);
  if (request.style) parts.emplace_back("style", *request.style);
  if (request.user) parts.emplace_back("user", *request.user);
  return build_multipart_body(parts, boundary);
}

std::vector<std::pair<std::string, std::string>> build_edit_parts(const ImageEditRequest& request,
                                                                  bool force_stream) {
  std::vector<std::pair<std::string, std::string>> parts;
  if (request.model) parts.emplace_back("model", *request.model);
  parts.emplace_back("prompt", request.prompt.value_or(""));
  if (request.n) parts.emplace_back("n", std::to_string(*request.n));
  if (request.size) parts.emplace_back("size", *request.size);
  if (request.response_format) parts.emplace_back("response_format", *request.response_format);
  parts.emplace_back("image", materialize_file_as_string(request.image, "image.bin"));
  if (request.mask) {
    parts.emplace_back("mask", materialize_file_as_string(*request.mask, "mask.bin"));
  }
  if (request.quality) parts.emplace_back("quality", *request.quality);
  if (request.style) parts.emplace_back("style", *request.style);
  if (request.input_fidelity) parts.emplace_back("input_fidelity", *request.input_fidelity);
  if (request.output_compression) {
    parts.emplace_back("output_compression", double_to_string(*request.output_compression));
  }
  if (request.output_format) parts.emplace_back("output_format", *request.output_format);
  if (request.partial_images) parts.emplace_back("partial_images", std::to_string(*request.partial_images));
  if (request.background) parts.emplace_back("background", *request.background);
  if (request.user) parts.emplace_back("user", *request.user);

  if (force_stream) {
    parts.emplace_back("stream", "true");
  } else if (request.stream.has_value()) {
    parts.emplace_back("stream", *request.stream ? "true" : "false");
  }

  return parts;
}

std::string build_edit_multipart(const ImageEditRequest& request,
                                 const std::string& boundary,
                                 bool force_stream) {
  return build_multipart_body(build_edit_parts(request, force_stream), boundary);
}

ImageStreamPartialEvent parse_stream_partial_event(const json& payload) {
  ImageStreamPartialEvent partial;
  partial.raw = payload;
  if (payload.contains("b64_json") && payload.at("b64_json").is_string()) {
    partial.b64_json = payload.at("b64_json").get<std::string>();
  }
  if (payload.contains("background") && payload.at("background").is_string()) {
    partial.background = payload.at("background").get<std::string>();
  }
  partial.created_at = payload.value("created_at", 0);
  if (payload.contains("output_format") && payload.at("output_format").is_string()) {
    partial.output_format = payload.at("output_format").get<std::string>();
  }
  partial.partial_image_index = payload.value("partial_image_index", 0);
  if (payload.contains("quality") && payload.at("quality").is_string()) {
    partial.quality = payload.at("quality").get<std::string>();
  }
  if (payload.contains("size") && payload.at("size").is_string()) {
    partial.size = payload.at("size").get<std::string>();
  }
  return partial;
}

ImageStreamCompletedEvent parse_stream_completed_event(const json& payload) {
  ImageStreamCompletedEvent completed;
  completed.raw = payload;
  if (payload.contains("b64_json") && payload.at("b64_json").is_string()) {
    completed.b64_json = payload.at("b64_json").get<std::string>();
  }
  if (payload.contains("background") && payload.at("background").is_string()) {
    completed.background = payload.at("background").get<std::string>();
  }
  completed.created_at = payload.value("created_at", 0);
  if (payload.contains("output_format") && payload.at("output_format").is_string()) {
    completed.output_format = payload.at("output_format").get<std::string>();
  }
  if (payload.contains("quality") && payload.at("quality").is_string()) {
    completed.quality = payload.at("quality").get<std::string>();
  }
  if (payload.contains("size") && payload.at("size").is_string()) {
    completed.size = payload.at("size").get<std::string>();
  }
  if (payload.contains("usage") && payload.at("usage").is_object()) {
    completed.usage = parse_image_usage_json(payload.at("usage"));
  }
  return completed;
}

}  // namespace

std::optional<ImageStreamEvent> parse_image_stream_event(const ServerSentEvent& event) {
  if (event.data == "[DONE]") {
    return std::nullopt;
  }

  try {
    auto payload = json::parse(event.data);
    if (!payload.is_object()) {
      return std::nullopt;
    }

    ImageStreamEvent result;
    result.raw = payload;
    result.event_name = event.event;
    result.type_name = payload.value("type", std::string{});

    const std::string& type = result.type_name;
    if (type == "image_generation.partial_image") {
      result.type = ImageStreamEvent::Type::ImageGenerationPartialImage;
      result.generation_partial = parse_stream_partial_event(payload);
      return result;
    }
    if (type == "image_generation.completed") {
      result.type = ImageStreamEvent::Type::ImageGenerationCompleted;
      result.generation_completed = parse_stream_completed_event(payload);
      return result;
    }
    if (type == "image_edit.partial_image") {
      result.type = ImageStreamEvent::Type::ImageEditPartialImage;
      result.edit_partial = parse_stream_partial_event(payload);
      return result;
    }
    if (type == "image_edit.completed") {
      result.type = ImageStreamEvent::Type::ImageEditCompleted;
      result.edit_completed = parse_stream_completed_event(payload);
      return result;
    }

    result.type = ImageStreamEvent::Type::Unknown;
    return result;
  } catch (const json::exception&) {
    return std::nullopt;
  }
}

ImagesResponse ImagesResource::generate(const ImageGenerateRequest& request,
                                        const RequestOptions& options) const {
  if (request.stream && *request.stream) {
    throw OpenAIError("Use generate_stream() to receive streaming image generation events.");
  }

  auto body = build_generate_body(request, false);
  auto response = client_.perform_request("POST", kImagesGenerate, body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_images_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse image response: ") + ex.what());
  }
}

ImagesResponse ImagesResource::generate(const ImageGenerateRequest& request) const {
  return generate(request, RequestOptions{});
}

std::vector<ServerSentEvent> ImagesResource::generate_stream(const ImageGenerateRequest& request,
                                                             const RequestOptions& options) const {
  SSEEventStream stream;

  auto body = build_generate_body(request, true);

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kImagesGenerate, body.dump(), request_options);

  stream.finalize();
  return stream.events();
}

std::vector<ServerSentEvent> ImagesResource::generate_stream(const ImageGenerateRequest& request) const {
  return generate_stream(request, RequestOptions{});
}

void ImagesResource::generate_stream(const ImageGenerateRequest& request,
                                     const std::function<bool(const ImageStreamEvent&)>& on_event,
                                     const RequestOptions& options) const {
  SSEEventStream stream([&](const ServerSentEvent& sse_event) {
    if (!on_event) {
      return true;
    }
    if (auto parsed = parse_image_stream_event(sse_event)) {
      return on_event(*parsed);
    }
    return true;
  });

  auto body = build_generate_body(request, true);

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kImagesGenerate, body.dump(), request_options);

  stream.finalize();
}

void ImagesResource::generate_stream(const ImageGenerateRequest& request,
                                     const std::function<bool(const ImageStreamEvent&)>& on_event) const {
  generate_stream(request, on_event, RequestOptions{});
}

ImagesResponse ImagesResource::create_variation(const ImageVariationRequest& request,
                                                const RequestOptions& options) const {
  const std::string boundary = "----openai-cpp-image-boundary";
  auto body = build_variation_multipart(request, boundary);

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + boundary;

  auto response = client_.perform_request("POST", kImagesVariation, body, request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_images_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse image variation response: ") + ex.what());
  }
}

ImagesResponse ImagesResource::create_variation(const ImageVariationRequest& request) const {
  return create_variation(request, RequestOptions{});
}

ImagesResponse ImagesResource::edit(const ImageEditRequest& request, const RequestOptions& options) const {
  if (request.stream && *request.stream) {
    throw OpenAIError("Use edit_stream() to receive streaming image edit events.");
  }

  const std::string boundary = "----openai-cpp-image-boundary";
  auto body = build_edit_multipart(request, boundary, false);

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + boundary;

  auto response = client_.perform_request("POST", kImagesEdit, body, request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_images_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse image edit response: ") + ex.what());
  }
}

ImagesResponse ImagesResource::edit(const ImageEditRequest& request) const {
  return edit(request, RequestOptions{});
}

std::vector<ServerSentEvent> ImagesResource::edit_stream(const ImageEditRequest& request,
                                                         const RequestOptions& options) const {
  SSEEventStream stream;

  const std::string boundary = "----openai-cpp-image-boundary";
  auto body = build_edit_multipart(request, boundary, true);

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + boundary;
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kImagesEdit, body, request_options);

  stream.finalize();
  return stream.events();
}

std::vector<ServerSentEvent> ImagesResource::edit_stream(const ImageEditRequest& request) const {
  return edit_stream(request, RequestOptions{});
}

void ImagesResource::edit_stream(const ImageEditRequest& request,
                                 const std::function<bool(const ImageStreamEvent&)>& on_event,
                                 const RequestOptions& options) const {
  SSEEventStream stream([&](const ServerSentEvent& sse_event) {
    if (!on_event) {
      return true;
    }
    if (auto parsed = parse_image_stream_event(sse_event)) {
      return on_event(*parsed);
    }
    return true;
  });

  const std::string boundary = "----openai-cpp-image-boundary";
  auto body = build_edit_multipart(request, boundary, true);

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.headers["Content-Type"] = "multipart/form-data; boundary=" + boundary;
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kImagesEdit, body, request_options);

  stream.finalize();
}

void ImagesResource::edit_stream(const ImageEditRequest& request,
                                 const std::function<bool(const ImageStreamEvent&)>& on_event) const {
  edit_stream(request, on_event, RequestOptions{});
}

}  // namespace openai

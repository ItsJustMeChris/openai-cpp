#include "openai/images.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

const char* kImagesGenerate = "/images/generations";
const char* kImagesVariation = "/images/variations";
const char* kImagesEdit = "/images/edits";

ImagesResponse parse_images_response(const json& payload) {
  ImagesResponse response;
  response.raw = payload;
  response.created = payload.value("created", 0);
  if (payload.contains("background") && payload["background"].is_string()) {
    response.background = payload["background"].get<std::string>();
  }
  if (payload.contains("output_format") && payload["output_format"].is_string()) {
    response.output_format = payload["output_format"].get<std::string>();
  }
  if (payload.contains("quality") && payload["quality"].is_string()) {
    response.quality = payload["quality"].get<std::string>();
  }
  if (payload.contains("size") && payload["size"].is_string()) {
    response.size = payload["size"].get<std::string>();
  }
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      ImageData data;
      data.raw = item;
      if (item.contains("b64_json") && item["b64_json"].is_string()) {
        data.b64_json = item["b64_json"].get<std::string>();
      }
      if (item.contains("url") && item["url"].is_string()) {
        data.url = item["url"].get<std::string>();
      }
      if (item.contains("revised_prompt") && item["revised_prompt"].is_string()) {
        data.revised_prompt = item["revised_prompt"].get<std::string>();
      }
      response.data.push_back(std::move(data));
    }
  }
  if (payload.contains("usage") && payload["usage"].is_object()) {
    ImageUsage usage;
    const auto& usage_json = payload["usage"];
    usage.raw = usage_json;
    usage.input_tokens = usage_json.value("input_tokens", 0);
    usage.output_tokens = usage_json.value("output_tokens", 0);
    usage.total_tokens = usage_json.value("total_tokens", 0);
    if (usage_json.contains("input_tokens_details") && usage_json["input_tokens_details"].is_object()) {
      ImageUsageInputTokensDetails details;
      const auto& details_json = usage_json["input_tokens_details"];
      details.raw = details_json;
      details.image_tokens = details_json.value("image_tokens", 0);
      details.text_tokens = details_json.value("text_tokens", 0);
      usage.input_tokens_details = std::move(details);
    }
    response.usage = std::move(usage);
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

std::string build_image_multipart(const ImageVariationRequest& request, const std::string& boundary) {
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

}  // namespace

ImagesResponse ImagesResource::generate(const ImageGenerateRequest& request, const RequestOptions& options) const {
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
  if (request.stream) {
    if (*request.stream) {
      throw OpenAIError("Image streaming is not yet supported. Remove stream=true for now.");
    }
    body["stream"] = *request.stream;
  }

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

ImagesResponse ImagesResource::create_variation(const ImageVariationRequest& request,
                                                const RequestOptions& options) const {
  const std::string boundary = "----openai-cpp-image-boundary";
  auto body = build_image_multipart(request, boundary);

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
  const std::string boundary = "----openai-cpp-image-boundary";
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
  if (request.output_compression) parts.emplace_back("output_compression", std::to_string(*request.output_compression));
  if (request.output_format) parts.emplace_back("output_format", *request.output_format);
  if (request.partial_images) parts.emplace_back("partial_images", std::to_string(*request.partial_images));
  if (request.background) parts.emplace_back("background", *request.background);
  if (request.user) parts.emplace_back("user", *request.user);
  if (request.stream) {
    if (*request.stream) {
      throw OpenAIError("Image streaming is not yet supported. Remove stream=true for now.");
    }
    parts.emplace_back("stream", *request.stream ? "true" : "false");
  }
  auto body = build_multipart_body(parts, boundary);

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

}  // namespace openai

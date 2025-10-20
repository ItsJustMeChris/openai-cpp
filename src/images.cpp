#include "openai/images.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <fstream>

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
  return response;
}

std::string load_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
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
  parts.emplace_back("purpose", request.image.purpose);
  parts.emplace_back("prompt", request.prompt.value_or(""));
  if (request.n) parts.emplace_back("n", std::to_string(*request.n));
  if (request.size) parts.emplace_back("size", *request.size);
  if (request.response_format) parts.emplace_back("response_format", *request.response_format);
  parts.emplace_back("image", load_file(request.image.file_path));
  return build_multipart_body(parts, boundary);
}

}  // namespace

ImagesResponse ImagesResource::generate(const ImageGenerateRequest& request, const RequestOptions& options) const {
  json body = request.extra.is_null() ? json::object() : request.extra;
  if (!body.is_object()) {
    throw OpenAIError("ImageGenerateRequest.extra must be an object");
  }
  body["prompt"] = request.prompt;
  if (request.model) body["model"] = *request.model;
  if (request.n) body["n"] = *request.n;
  if (request.size) body["size"] = *request.size;
  if (request.response_format) body["response_format"] = *request.response_format;

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
  parts.emplace_back("prompt", request.prompt.value_or(""));
  if (request.n) parts.emplace_back("n", std::to_string(*request.n));
  if (request.size) parts.emplace_back("size", *request.size);
  if (request.response_format) parts.emplace_back("response_format", *request.response_format);
  parts.emplace_back("image", load_file(request.image.file_path));
  if (request.mask) {
    parts.emplace_back("mask", load_file(request.mask->file_path));
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


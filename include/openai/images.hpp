#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/files.hpp"

namespace openai {

struct ImageData {
  std::optional<std::string> b64_json;
  std::optional<std::string> url;
  std::optional<std::string> revised_prompt;
  nlohmann::json raw = nlohmann::json::object();
};

struct ImageUsageInputTokensDetails {
  int image_tokens = 0;
  int text_tokens = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ImageUsage {
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  std::optional<ImageUsageInputTokensDetails> input_tokens_details;
  nlohmann::json raw = nlohmann::json::object();
};

struct ImagesResponse {
  int created = 0;
  std::optional<std::string> background;
  std::vector<ImageData> data;
  std::optional<std::string> output_format;
  std::optional<std::string> quality;
  std::optional<std::string> size;
  std::optional<ImageUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct ImageGenerateRequest {
  std::string prompt;
  std::optional<std::string> model;
  std::optional<int> n;
  std::optional<std::string> size;
  std::optional<std::string> response_format;
  std::optional<std::string> quality;
  std::optional<std::string> style;
  std::optional<std::string> moderation;
  std::optional<double> output_compression;
  std::optional<std::string> output_format;
  std::optional<int> partial_images;
  std::optional<bool> stream;
  std::optional<std::string> background;
  std::optional<std::string> user;
};

struct ImageVariationRequest {
  FileUploadRequest image;
  std::optional<std::string> model;
  std::optional<std::string> prompt;
  std::optional<int> n;
  std::optional<std::string> size;
  std::optional<std::string> response_format;
  std::optional<std::string> quality;
  std::optional<std::string> style;
  std::optional<std::string> background;
  std::optional<std::string> user;
};

struct ImageEditRequest {
  FileUploadRequest image;
  std::optional<FileUploadRequest> mask;
  std::optional<std::string> model;
  std::optional<std::string> prompt;
  std::optional<int> n;
  std::optional<std::string> size;
  std::optional<std::string> response_format;
  std::optional<std::string> quality;
  std::optional<std::string> style;
  std::optional<std::string> input_fidelity;
  std::optional<double> output_compression;
  std::optional<std::string> output_format;
  std::optional<int> partial_images;
  std::optional<bool> stream;
  std::optional<std::string> background;
  std::optional<std::string> user;
};

struct RequestOptions;
class OpenAIClient;

class ImagesResource {
public:
  explicit ImagesResource(OpenAIClient& client) : client_(client) {}

  ImagesResponse generate(const ImageGenerateRequest& request) const;
  ImagesResponse generate(const ImageGenerateRequest& request, const RequestOptions& options) const;

  ImagesResponse create_variation(const ImageVariationRequest& request) const;
  ImagesResponse create_variation(const ImageVariationRequest& request, const RequestOptions& options) const;

  ImagesResponse edit(const ImageEditRequest& request) const;
  ImagesResponse edit(const ImageEditRequest& request, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

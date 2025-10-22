#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/files.hpp"

namespace openai {

struct RequestOptions;
class OpenAIClient;

struct UploadCreateExpiresAfter {
  std::string anchor;
  int seconds = 0;
};

struct UploadCreateParams {
  std::size_t bytes = 0;
  std::string filename;
  std::string mime_type;
  std::string purpose;
  std::optional<UploadCreateExpiresAfter> expires_after;
};

struct UploadCompleteParams {
  std::vector<std::string> part_ids;
  std::optional<std::string> md5;
};

struct Upload {
  std::string id;
  std::size_t bytes = 0;
  int created_at = 0;
  int expires_at = 0;
  std::string filename;
  std::string object;
  std::string purpose;
  std::string status;
  std::optional<FileObject> file;
  nlohmann::json raw = nlohmann::json::object();
};

struct UploadPart {
  std::string id;
  int created_at = 0;
  std::string object;
  std::string upload_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct UploadPartCreateParams {
  std::vector<std::uint8_t> data;
  std::optional<std::string> filename;
  std::optional<std::string> content_type;
};

class UploadPartsResource {
public:
  explicit UploadPartsResource(OpenAIClient& client) : client_(client) {}

  UploadPart create(const std::string& upload_id, const UploadPartCreateParams& params) const;
  UploadPart create(const std::string& upload_id,
                    const UploadPartCreateParams& params,
                    const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class UploadsResource {
public:
  explicit UploadsResource(OpenAIClient& client) : client_(client), parts_(client) {}

  Upload create(const UploadCreateParams& params) const;
  Upload create(const UploadCreateParams& params, const RequestOptions& options) const;

  Upload cancel(const std::string& upload_id) const;
  Upload cancel(const std::string& upload_id, const RequestOptions& options) const;

  Upload complete(const std::string& upload_id, const UploadCompleteParams& params) const;
  Upload complete(const std::string& upload_id,
                  const UploadCompleteParams& params,
                  const RequestOptions& options) const;

  UploadPartsResource& parts() { return parts_; }
  const UploadPartsResource& parts() const { return parts_; }

private:
  OpenAIClient& client_;
  UploadPartsResource parts_;
};

}  // namespace openai


#include "openai/uploads.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/utils/multipart.hpp"

namespace openai {
namespace {

using json = nlohmann::json;

FileObject parse_file_object(const json& payload) {
  FileObject file;
  file.raw = payload;
  file.id = payload.value("id", "");
  file.bytes = payload.value("bytes", 0);
  file.created_at = payload.value("created_at", 0);
  file.filename = payload.value("filename", "");
  file.object = payload.value("object", "");
  file.purpose = payload.value("purpose", "");
  file.status = payload.value("status", "");
  if (payload.contains("expires_at") && !payload.at("expires_at").is_null()) {
    file.expires_at = payload.at("expires_at").get<int>();
  }
  if (payload.contains("status_details") && !payload.at("status_details").is_null()) {
    file.status_details = payload.at("status_details").get<std::string>();
  }
  return file;
}

Upload parse_upload(const json& payload) {
  Upload upload;
  upload.raw = payload;
  upload.id = payload.value("id", "");
  upload.bytes = payload.value("bytes", static_cast<std::size_t>(0));
  upload.created_at = payload.value("created_at", 0);
  upload.expires_at = payload.value("expires_at", 0);
  upload.filename = payload.value("filename", "");
  upload.object = payload.value("object", "");
  upload.purpose = payload.value("purpose", "");
  upload.status = payload.value("status", "");
  if (payload.contains("file") && !payload.at("file").is_null()) {
    upload.file = parse_file_object(payload.at("file"));
  }
  return upload;
}

UploadPart parse_upload_part(const json& payload) {
  UploadPart part;
  part.raw = payload;
  part.id = payload.value("id", "");
  part.created_at = payload.value("created_at", 0);
  part.object = payload.value("object", "");
  part.upload_id = payload.value("upload_id", "");
  return part;
}

json build_upload_create_body(const UploadCreateParams& params) {
  json body = json::object();
  body["bytes"] = params.bytes;
  body["filename"] = params.filename;
  body["mime_type"] = params.mime_type;
  body["purpose"] = params.purpose;
  if (params.expires_after) {
    json expires;
    expires["anchor"] = params.expires_after->anchor;
    expires["seconds"] = params.expires_after->seconds;
    body["expires_after"] = std::move(expires);
  }
  return body;
}

json build_upload_complete_body(const UploadCompleteParams& params) {
  json body = json::object();
  body["part_ids"] = params.part_ids;
  if (params.md5) {
    body["md5"] = *params.md5;
  }
  return body;
}

}  // namespace

UploadPart UploadPartsResource::create(const std::string& upload_id,
                                       const UploadPartCreateParams& params,
                                       const RequestOptions& options) const {
  utils::MultipartFormData form;
  const std::string filename = params.filename.value_or("chunk.bin");
  const std::string content_type = params.content_type.value_or("application/octet-stream");
  form.append_file("data", filename, content_type, params.data);
  auto encoded = form.build();

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = encoded.content_type;

  auto response =
      client_.perform_request("POST", "/uploads/" + upload_id + "/parts", encoded.body, request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_upload_part(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse upload part response: ") + ex.what());
  }
}

UploadPart UploadPartsResource::create(const std::string& upload_id,
                                       const UploadPartCreateParams& params) const {
  return create(upload_id, params, RequestOptions{});
}

Upload UploadsResource::create(const UploadCreateParams& params, const RequestOptions& options) const {
  auto body = build_upload_create_body(params).dump();
  auto response = client_.perform_request("POST", "/uploads", body, options);
  try {
    auto payload = json::parse(response.body);
    return parse_upload(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse upload create response: ") + ex.what());
  }
}

Upload UploadsResource::create(const UploadCreateParams& params) const {
  return create(params, RequestOptions{});
}

Upload UploadsResource::cancel(const std::string& upload_id, const RequestOptions& options) const {
  auto response = client_.perform_request("POST", "/uploads/" + upload_id + "/cancel", "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_upload(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse upload cancel response: ") + ex.what());
  }
}

Upload UploadsResource::cancel(const std::string& upload_id) const {
  return cancel(upload_id, RequestOptions{});
}

Upload UploadsResource::complete(const std::string& upload_id,
                                 const UploadCompleteParams& params,
                                 const RequestOptions& options) const {
  auto body = build_upload_complete_body(params).dump();
  auto response = client_.perform_request("POST", "/uploads/" + upload_id + "/complete", body, options);
  try {
    auto payload = json::parse(response.body);
    return parse_upload(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse upload complete response: ") + ex.what());
  }
}

Upload UploadsResource::complete(const std::string& upload_id, const UploadCompleteParams& params) const {
  return complete(upload_id, params, RequestOptions{});
}

}  // namespace openai

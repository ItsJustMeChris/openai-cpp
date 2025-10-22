#include "openai/files.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"
#include "openai/utils/multipart.hpp"

#include <nlohmann/json.hpp>

#include <memory>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kFilesPath = "/files";

}  // namespace

utils::UploadFile FileUploadRequest::materialize(const std::string& default_filename) const {
  utils::UploadFile upload;

  if (file_data.has_value()) {
    upload = *file_data;
  } else if (file_path.has_value()) {
    upload = utils::to_file(*file_path);
  } else {
    throw OpenAIError("FileUploadRequest must provide either file_path or file_data");
  }

  if (file_name) {
    upload.filename = *file_name;
  }

  if (upload.filename.empty()) {
    upload.filename = default_filename;
  }

  if (content_type) {
    upload.content_type = content_type;
  }

  if (!upload.content_type.has_value()) {
    upload.content_type = std::string("application/octet-stream");
  }

  return upload;
}

namespace {

FileObject parse_file(const json& payload) {
  FileObject file;
  file.raw = payload;
  file.id = payload.value("id", "");
  file.bytes = payload.value("bytes", 0);
  file.created_at = payload.value("created_at", 0);
  file.filename = payload.value("filename", "");
  file.object = payload.value("object", "");
  file.purpose = payload.value("purpose", "");
  if (payload.contains("status") && !payload.at("status").is_null()) {
    file.status = payload.at("status").get<std::string>();
  }
  if (payload.contains("expires_at") && !payload.at("expires_at").is_null()) {
    file.expires_at = payload.at("expires_at").get<int>();
  }
  if (payload.contains("status_details") && !payload.at("status_details").is_null()) {
    file.status_details = payload.at("status_details").get<std::string>();
  }
  return file;
}

FileDeleted parse_file_deleted(const json& payload) {
  FileDeleted result;
  result.raw = payload;
  result.id = payload.value("id", "");
  result.deleted = payload.value("deleted", false);
  result.object = payload.value("object", "");
  return result;
}

FileList parse_file_list(const json& payload) {
  FileList list;
  list.raw = payload;
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_file(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  return list;
}

}  // namespace

CursorPage<FileObject> FilesResource::list_page(const RequestOptions& options) const {
  auto fetch_impl = std::make_shared<std::function<CursorPage<FileObject>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<FileObject> {
    RequestOptions next_options = to_request_options(request_options);
    auto response = client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    try {
      auto payload = json::parse(response.body);
      auto list = parse_file_list(payload);
      std::optional<std::string> cursor = list.next_cursor;
      if (!cursor && !list.data.empty()) {
        cursor = list.data.back().id;
      }

      return CursorPage<FileObject>(
          std::move(list.data),
          list.has_more,
          std::move(cursor),
          request_options,
          *fetch_impl,
          "after",
          std::move(list.raw));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse file list: ") + ex.what());
    }
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kFilesPath;
  initial.headers = materialize_headers(options);
  initial.query = materialize_query(options);

  return (*fetch_impl)(initial);
}

CursorPage<FileObject> FilesResource::list_page() const {
  return list_page(RequestOptions{});
}

FileList FilesResource::list(const RequestOptions& options) const {
  auto page = list_page(options);
  FileList list;
  list.data = page.data();
  list.has_more = page.has_next_page();
  list.next_cursor = page.next_cursor();
  list.raw = page.raw();
  return list;
}

FileList FilesResource::list() const {
  return list(RequestOptions{});
}

FileObject FilesResource::retrieve(const std::string& file_id, const RequestOptions& options) const {
  auto path = std::string(kFilesPath) + "/" + file_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_file(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse file: ") + ex.what());
  }
}

FileObject FilesResource::retrieve(const std::string& file_id) const {
  return retrieve(file_id, RequestOptions{});
}

FileDeleted FilesResource::remove(const std::string& file_id, const RequestOptions& options) const {
  auto path = std::string(kFilesPath) + "/" + file_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_file_deleted(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse file delete response: ") + ex.what());
  }
}

FileDeleted FilesResource::remove(const std::string& file_id) const {
  return remove(file_id, RequestOptions{});
}

FileObject FilesResource::create(const FileUploadRequest& request, const RequestOptions& options) const {
  utils::UploadFile upload = request.materialize("upload.bin");
  const std::string content_type = *upload.content_type;

  utils::MultipartFormData form;
  form.append_text("purpose", request.purpose);
  form.append_file("file", upload.filename, content_type, upload.data);
  auto encoded = form.build();

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = encoded.content_type;

  auto response = client_.perform_request("POST", kFilesPath, encoded.body, request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_file(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse file upload response: ") + ex.what());
  }
}

FileObject FilesResource::create(const FileUploadRequest& request) const {
  return create(request, RequestOptions{});
}

FileContent FilesResource::content(const std::string& file_id, const RequestOptions& options) const {
  auto path = std::string(kFilesPath) + "/" + file_id + "/content";
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "application/octet-stream";
  auto response = client_.perform_request("GET", path, "", request_options);
  FileContent content;
  content.headers = response.headers;
  content.data.assign(response.body.begin(), response.body.end());
  return content;
}

FileContent FilesResource::content(const std::string& file_id) const {
  return content(file_id, RequestOptions{});
}

}  // namespace openai

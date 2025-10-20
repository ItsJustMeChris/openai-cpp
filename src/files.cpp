#include "openai/files.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kFilesPath = "/files";
constexpr const char* kMultipartBoundary = "----openai-cpp-boundary";

std::string filename_from_path(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

std::vector<std::uint8_t> read_file_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string build_multipart_body(const FileUploadRequest& request, const std::vector<std::uint8_t>& file_data) {
  std::ostringstream body;
  const std::string boundary = kMultipartBoundary;
  const std::string filename = request.file_name.value_or(filename_from_path(request.file_path));
  const std::string content_type = request.content_type.value_or("application/octet-stream");

  body << "--" << boundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"purpose\"\r\n\r\n";
  body << request.purpose << "\r\n";

  body << "--" << boundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n";
  body << "Content-Type: " << content_type << "\r\n\r\n";
  body.write(reinterpret_cast<const char*>(file_data.data()), static_cast<std::streamsize>(file_data.size()));
  body << "\r\n--" << boundary << "--\r\n";

  return body.str();
}

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

FileList FilesResource::list(const RequestOptions& options) const {
  auto response = client_.perform_request("GET", kFilesPath, "", options);
  try {
    auto payload = json::parse(response.body);
    return parse_file_list(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse file list: ") + ex.what());
  }
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
  auto file_data = read_file_binary(request.file_path);
  auto body = build_multipart_body(request, file_data);

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = std::string("multipart/form-data; boundary=") + kMultipartBoundary;

  auto response = client_.perform_request("POST", kFilesPath, body, request_options);
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

#include "openai/files.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kFilesPath = "/files";

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

}  // namespace openai


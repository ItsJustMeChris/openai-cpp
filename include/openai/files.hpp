#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/utils/to_file.hpp"

namespace openai {

struct FileObject {
  std::string id;
  std::size_t bytes = 0;
  int created_at = 0;
  std::string filename;
  std::string object;
  std::string purpose;
  std::string status;
  std::optional<int> expires_at;
  std::optional<std::string> status_details;
  nlohmann::json raw = nlohmann::json::object();
};

struct FileDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct FileList {
  std::vector<FileObject> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct FileListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> order;
  std::optional<std::string> purpose;
};

struct FileContent {
  std::vector<std::uint8_t> data;
  std::map<std::string, std::string> headers;
};

struct FileUploadExpiresAfter {
  std::string anchor;
  int seconds = 0;
};

struct FileUploadRequest {
  std::string purpose;
  std::optional<std::string> file_path;
  std::optional<utils::UploadFile> file_data;
  std::optional<std::string> file_name;
  std::optional<std::string> content_type;
  std::optional<FileUploadExpiresAfter> expires_after;

  utils::UploadFile materialize(const std::string& default_filename = "file") const;
};

struct RequestOptions;
template <typename Item>
class CursorPage;

class OpenAIClient;

class FilesResource {
public:
  explicit FilesResource(OpenAIClient& client) : client_(client) {}

  FileList list() const;
  FileList list(const RequestOptions& options) const;
  FileList list(const FileListParams& params) const;
  FileList list(const FileListParams& params, const RequestOptions& options) const;
  CursorPage<FileObject> list_page() const;
  CursorPage<FileObject> list_page(const RequestOptions& options) const;
  CursorPage<FileObject> list_page(const FileListParams& params) const;
  CursorPage<FileObject> list_page(const FileListParams& params, const RequestOptions& options) const;

  FileObject retrieve(const std::string& file_id) const;
  FileObject retrieve(const std::string& file_id, const RequestOptions& options) const;

  FileDeleted remove(const std::string& file_id) const;
  FileDeleted remove(const std::string& file_id, const RequestOptions& options) const;

  FileObject create(const FileUploadRequest& request) const;
  FileObject create(const FileUploadRequest& request, const RequestOptions& options) const;

  FileContent content(const std::string& file_id) const;
  FileContent content(const std::string& file_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

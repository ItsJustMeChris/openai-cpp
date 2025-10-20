#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct FileObject {
  std::string id;
  std::size_t bytes = 0;
  int created_at = 0;
  std::string filename;
  std::string object;
  std::string purpose;
  std::optional<std::string> status;
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

struct RequestOptions;

class OpenAIClient;

class FilesResource {
public:
  explicit FilesResource(OpenAIClient& client) : client_(client) {}

  FileList list() const;
  FileList list(const RequestOptions& options) const;

  FileObject retrieve(const std::string& file_id) const;
  FileObject retrieve(const std::string& file_id, const RequestOptions& options) const;

  FileDeleted remove(const std::string& file_id) const;
  FileDeleted remove(const std::string& file_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai


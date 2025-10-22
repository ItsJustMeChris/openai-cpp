#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

struct ContainerExpiresAfter {
  std::optional<std::string> anchor;
  std::optional<int> minutes;
};

struct Container {
  std::string id;
  int created_at = 0;
  std::string name;
  std::string object;
  std::string status;
  std::optional<ContainerExpiresAfter> expires_after;
  nlohmann::json raw = nlohmann::json::object();
};

struct ContainerList {
  std::vector<Container> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ContainerCreateRequest {
  struct ExpiresAfter {
    std::string anchor;
    int minutes = 0;
  };

  std::string name;
  std::optional<ExpiresAfter> expires_after;
  std::vector<std::string> file_ids;
};

struct ContainerListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
};

struct ContainerFile {
  std::string id;
  std::size_t bytes = 0;
  std::string container_id;
  int created_at = 0;
  std::string object;
  std::string path;
  std::string source;
  nlohmann::json raw = nlohmann::json::object();
};

struct ContainerFileList {
  std::vector<ContainerFile> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ContainerFileCreateRequest {
  std::optional<std::string> file_id;
  std::optional<std::string> file_path;
  std::optional<std::vector<std::uint8_t>> file_data;
  std::optional<std::string> file_name;
  std::optional<std::string> content_type;
};

struct ContainerFileListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
};

struct ContainerFileContent {
  std::vector<std::uint8_t> data;
  std::map<std::string, std::string> headers;
};

class ContainerFilesContentResource {
public:
  explicit ContainerFilesContentResource(OpenAIClient& client) : client_(client) {}

  ContainerFileContent retrieve(const std::string& container_id,
                                const std::string& file_id) const;
  ContainerFileContent retrieve(const std::string& container_id,
                                const std::string& file_id,
                                const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class ContainerFilesResource {
public:
  explicit ContainerFilesResource(OpenAIClient& client)
      : client_(client), content_(client) {}

  ContainerFile create(const std::string& container_id,
                       const ContainerFileCreateRequest& request) const;
  ContainerFile create(const std::string& container_id,
                       const ContainerFileCreateRequest& request,
                       const RequestOptions& options) const;

  ContainerFile retrieve(const std::string& container_id,
                         const std::string& file_id) const;
  ContainerFile retrieve(const std::string& container_id,
                         const std::string& file_id,
                         const RequestOptions& options) const;

  ContainerFileList list(const std::string& container_id) const;
  ContainerFileList list(const std::string& container_id,
                         const ContainerFileListParams& params) const;
  ContainerFileList list(const std::string& container_id,
                         const ContainerFileListParams& params,
                         const RequestOptions& options) const;

  CursorPage<ContainerFile> list_page(const std::string& container_id) const;
  CursorPage<ContainerFile> list_page(const std::string& container_id,
                                      const ContainerFileListParams& params) const;
  CursorPage<ContainerFile> list_page(const std::string& container_id,
                                      const ContainerFileListParams& params,
                                      const RequestOptions& options) const;

  void remove(const std::string& container_id,
              const std::string& file_id) const;
  void remove(const std::string& container_id,
              const std::string& file_id,
              const RequestOptions& options) const;

  ContainerFilesContentResource& content() { return content_; }
  const ContainerFilesContentResource& content() const { return content_; }

private:
  OpenAIClient& client_;
  ContainerFilesContentResource content_;
};

class ContainersResource {
public:
  explicit ContainersResource(OpenAIClient& client)
      : client_(client), files_(client) {}

  Container create(const ContainerCreateRequest& request) const;
  Container create(const ContainerCreateRequest& request,
                   const RequestOptions& options) const;

  Container retrieve(const std::string& container_id) const;
  Container retrieve(const std::string& container_id,
                     const RequestOptions& options) const;

  ContainerList list() const;
  ContainerList list(const ContainerListParams& params) const;
  ContainerList list(const ContainerListParams& params,
                     const RequestOptions& options) const;

  CursorPage<Container> list_page() const;
  CursorPage<Container> list_page(const ContainerListParams& params) const;
  CursorPage<Container> list_page(const ContainerListParams& params,
                                  const RequestOptions& options) const;

  void remove(const std::string& container_id) const;
  void remove(const std::string& container_id, const RequestOptions& options) const;

  ContainerFilesResource& files() { return files_; }
  const ContainerFilesResource& files() const { return files_; }

private:
  OpenAIClient& client_;
  ContainerFilesResource files_;
};

}  // namespace openai

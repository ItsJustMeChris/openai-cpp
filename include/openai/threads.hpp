#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/assistants.hpp"

namespace openai {

struct ThreadToolResources {
  std::vector<std::string> code_interpreter_file_ids;
  std::vector<std::string> file_search_vector_store_ids;
};

struct ThreadMessageAttachmentTool {
  enum class Type { CodeInterpreter, FileSearch };
  Type type = Type::CodeInterpreter;
};

struct ThreadMessageAttachment {
  std::string file_id;
  std::vector<ThreadMessageAttachmentTool> tools;
};

struct ThreadMessageContentPart {
  enum class Type { Text, ImageFile, ImageURL, Raw };

  struct ImageFileData {
    std::string file_id;
    std::optional<std::string> detail;
  };

  struct ImageURLData {
    std::string url;
    std::optional<std::string> detail;
  };

  Type type = Type::Text;
  std::string text;
  std::optional<ImageFileData> image_file;
  std::optional<ImageURLData> image_url;
  nlohmann::json raw = nlohmann::json::object();
};

struct ThreadMessageCreate {
  std::string role;
  std::variant<std::monostate, std::string, std::vector<ThreadMessageContentPart>> content;
  std::vector<ThreadMessageAttachment> attachments;
  std::map<std::string, std::string> metadata;
};

struct ThreadCreateRequest {
  std::vector<ThreadMessageCreate> messages;
  std::map<std::string, std::string> metadata;
  std::optional<ThreadToolResources> tool_resources;
};

struct ThreadUpdateRequest {
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<ThreadToolResources> tool_resources;
};

struct Thread {
  std::string id;
  int created_at = 0;
  std::map<std::string, std::string> metadata;
  std::string object;
  std::optional<ThreadToolResources> tool_resources;
  nlohmann::json raw = nlohmann::json::object();
};

struct ThreadDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct RequestOptions;
class OpenAIClient;

class ThreadsResource {
public:
  explicit ThreadsResource(OpenAIClient& client) : client_(client) {}

  Thread create(const ThreadCreateRequest& request) const;
  Thread create(const ThreadCreateRequest& request, const RequestOptions& options) const;

  Thread retrieve(const std::string& thread_id) const;
  Thread retrieve(const std::string& thread_id, const RequestOptions& options) const;

  Thread update(const std::string& thread_id, const ThreadUpdateRequest& request) const;
  Thread update(const std::string& thread_id, const ThreadUpdateRequest& request, const RequestOptions& options) const;

  ThreadDeleteResponse remove(const std::string& thread_id) const;
  ThreadDeleteResponse remove(const std::string& thread_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

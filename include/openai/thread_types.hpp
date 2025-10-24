#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct ThreadToolResources {
  struct CodeInterpreter {
    std::vector<std::string> file_ids;
  };

  struct FileSearchVectorStoreChunkingStrategy {
    std::string type;
    std::optional<int> chunk_overlap_tokens;
    std::optional<int> max_chunk_size_tokens;
  };

  struct FileSearchVectorStore {
    std::optional<FileSearchVectorStoreChunkingStrategy> chunking_strategy;
    std::vector<std::string> file_ids;
    std::optional<std::map<std::string, std::string>> metadata;
  };

  struct FileSearch {
    std::vector<std::string> vector_store_ids;
    std::vector<FileSearchVectorStore> vector_stores;
  };

  std::optional<CodeInterpreter> code_interpreter;
  std::optional<FileSearch> file_search;
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

}  // namespace openai

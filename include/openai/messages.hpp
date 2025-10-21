#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/threads.hpp"

namespace openai {

struct MessageTextAnnotation {
  enum class Type { FileCitation, FilePath, Raw };
  Type type = Type::Raw;
  std::string text;
  std::optional<std::string> file_id;
  std::optional<std::string> quote;
  nlohmann::json raw = nlohmann::json::object();
  int start_index = 0;
  int end_index = 0;
};

struct MessageTextContent {
  std::string value;
  std::vector<MessageTextAnnotation> annotations;
};

struct MessageContentPart {
  enum class Type { Text, ImageFile, ImageURL, Refusal, Raw };

  struct ImageFileData {
    std::string file_id;
    std::optional<std::string> detail;
  };

  struct ImageURLData {
    std::string url;
    std::optional<std::string> detail;
  };

  Type type = Type::Text;
  MessageTextContent text;
  std::optional<ImageFileData> image_file;
  std::optional<ImageURLData> image_url;
  std::string refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct MessageAttachment {
  std::string file_id;
  std::vector<ThreadMessageAttachmentTool> tools;
};

struct ThreadMessage {
  std::string id;
  std::optional<std::string> assistant_id;
  std::vector<MessageAttachment> attachments;
  std::optional<int> completed_at;
  std::vector<MessageContentPart> content;
  int created_at = 0;
  std::optional<int> incomplete_at;
  std::optional<std::string> incomplete_reason;
  std::map<std::string, std::string> metadata;
  std::string object;
  std::string role;
  std::optional<std::string> run_id;
  std::string status;
  std::string thread_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct MessageList {
  std::vector<ThreadMessage> data;
  bool has_more = false;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct MessageCreateRequest {
  std::variant<std::string, std::vector<ThreadMessageContentPart>> content;
  std::string role;
  std::vector<MessageAttachment> attachments;
  std::map<std::string, std::string> metadata;
};

struct MessageRetrieveParams {
  std::string thread_id;
};

struct MessageUpdateRequest {
  std::string thread_id;
  std::optional<std::map<std::string, std::string>> metadata;
};

struct MessageListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> run_id;
};

struct MessageDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct MessageContentDeltaPart {
  enum class Type { Text, ImageFile, ImageURL, Refusal, Raw };

  int index = 0;
  Type type = Type::Text;
  std::optional<MessageTextContent> text;
  std::optional<MessageContentPart::ImageFileData> image_file;
  std::optional<MessageContentPart::ImageURLData> image_url;
  std::optional<std::string> refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct ThreadMessageDelta {
  std::optional<std::string> role;
  std::vector<MessageContentDeltaPart> content;
  nlohmann::json raw = nlohmann::json::object();
};

struct ThreadMessageDeltaEvent {
  std::string id;
  ThreadMessageDelta delta;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct RequestOptions;
class OpenAIClient;

class ThreadMessagesResource {
public:
  explicit ThreadMessagesResource(OpenAIClient& client) : client_(client) {}

  ThreadMessage create(const std::string& thread_id, const MessageCreateRequest& request) const;
  ThreadMessage create(const std::string& thread_id, const MessageCreateRequest& request,
                       const RequestOptions& options) const;

  ThreadMessage retrieve(const std::string& thread_id, const std::string& message_id) const;
  ThreadMessage retrieve(const std::string& thread_id, const std::string& message_id, const RequestOptions& options) const;

  ThreadMessage update(const std::string& thread_id, const std::string& message_id, const MessageUpdateRequest& request) const;
  ThreadMessage update(const std::string& thread_id, const std::string& message_id,
                       const MessageUpdateRequest& request, const RequestOptions& options) const;

  MessageDeleteResponse remove(const std::string& thread_id, const std::string& message_id) const;
  MessageDeleteResponse remove(const std::string& thread_id, const std::string& message_id, const RequestOptions& options) const;

  MessageList list(const std::string& thread_id) const;
  MessageList list(const std::string& thread_id, const MessageListParams& params) const;
  MessageList list(const std::string& thread_id, const RequestOptions& options) const;
  MessageList list(const std::string& thread_id, const MessageListParams& params, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

ThreadMessage parse_thread_message_json(const nlohmann::json& payload);
ThreadMessageDeltaEvent parse_thread_message_delta_json(const nlohmann::json& payload);

}  // namespace openai

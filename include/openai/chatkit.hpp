#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

class OpenAIClient;
struct RequestOptions;

namespace beta {

using ChatKitStateVariableValue = std::variant<std::string, bool, double>;

struct ChatKitWorkflowTracing {
  bool enabled = true;
};

struct ChatKitWorkflow {
  std::string id;
  std::optional<std::map<std::string, ChatKitStateVariableValue>> state_variables;
  ChatKitWorkflowTracing tracing;
  std::optional<std::string> version;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitSessionWorkflowParam {
  struct Tracing {
    std::optional<bool> enabled;
  };

  std::string id;
  std::optional<std::map<std::string, ChatKitStateVariableValue>> state_variables;
  std::optional<Tracing> tracing;
  std::optional<std::string> version;
};

struct ChatKitSessionChatKitConfigurationParam {
  struct AutomaticThreadTitling {
    std::optional<bool> enabled;
  };

  struct FileUpload {
    std::optional<bool> enabled;
    std::optional<int> max_file_size;
    std::optional<int> max_files;
  };

  struct History {
    std::optional<bool> enabled;
    std::optional<int> recent_threads;
  };

  std::optional<AutomaticThreadTitling> automatic_thread_titling;
  std::optional<FileUpload> file_upload;
  std::optional<History> history;
};

struct ChatKitSessionExpiresAfterParam {
  std::string anchor = "created_at";
  int seconds = 0;
};

struct ChatKitSessionRateLimitsParam {
  std::optional<int> max_requests_per_1_minute;
};

struct ChatKitSessionCreateParams {
  std::string user;
  ChatKitSessionWorkflowParam workflow;
  std::optional<ChatKitSessionChatKitConfigurationParam> chatkit_configuration;
  std::optional<ChatKitSessionExpiresAfterParam> expires_after;
  std::optional<ChatKitSessionRateLimitsParam> rate_limits;
};

struct ChatKitSessionAutomaticThreadTitling {
  bool enabled = true;
};

struct ChatKitSessionFileUpload {
  bool enabled = false;
  std::optional<int> max_file_size;
  std::optional<int> max_files;
};

struct ChatKitSessionHistory {
  bool enabled = true;
  std::optional<int> recent_threads;
};

struct ChatKitSessionChatKitConfiguration {
  ChatKitSessionAutomaticThreadTitling automatic_thread_titling;
  ChatKitSessionFileUpload file_upload;
  ChatKitSessionHistory history;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitSessionRateLimits {
  std::optional<int> max_requests_per_1_minute;
  nlohmann::json raw = nlohmann::json::object();
};

enum class ChatKitSessionStatus {
  Active,
  Expired,
  Cancelled,
  Unknown
};

struct ChatKitSession {
  std::string id;
  std::string object;
  int expires_at = 0;
  std::optional<std::string> client_secret;
  std::optional<int> max_requests_per_1_minute;
  ChatKitSessionStatus status = ChatKitSessionStatus::Unknown;
  std::string user;
  ChatKitWorkflow workflow;
  ChatKitSessionChatKitConfiguration chatkit_configuration;
  ChatKitSessionRateLimits rate_limits;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitAttachment {
  std::string id;
  std::string mime_type;
  std::string name;
  std::optional<std::string> preview_url;
  std::string type;
};

struct ChatKitResponseOutputTextAnnotationFileSource {
  std::string filename;
  std::string type;
};

struct ChatKitResponseOutputTextAnnotationFile {
  ChatKitResponseOutputTextAnnotationFileSource source;
  std::string type;
};

struct ChatKitResponseOutputTextAnnotationURLSource {
  std::string type;
  std::string url;
};

struct ChatKitResponseOutputTextAnnotationURL {
  ChatKitResponseOutputTextAnnotationURLSource source;
  std::string type;
};

using ChatKitResponseOutputTextAnnotation =
    std::variant<ChatKitResponseOutputTextAnnotationFile, ChatKitResponseOutputTextAnnotationURL>;

struct ChatKitResponseOutputText {
  std::vector<ChatKitResponseOutputTextAnnotation> annotations;
  std::string text;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

enum class ChatKitThreadStatusType {
  Active,
  Locked,
  Closed,
  Unknown
};

struct ChatKitThreadStatus {
  ChatKitThreadStatusType type = ChatKitThreadStatusType::Unknown;
  std::optional<std::string> reason;
};

struct ChatKitThread {
  std::string id;
  int created_at = 0;
  std::string object;
  ChatKitThreadStatus status;
  std::optional<std::string> title;
  std::string user;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadUserMessageContent {
  enum class Type {
    InputText,
    QuotedText,
    Unknown
  };

  Type type = Type::Unknown;
  std::string text;
};

struct ChatKitThreadUserMessageItem {
  struct InferenceOptions {
    struct ToolChoice {
      std::string id;
    };

    std::optional<std::string> model;
    std::optional<ToolChoice> tool_choice;
  };

  std::string id;
  std::vector<ChatKitAttachment> attachments;
  std::vector<ChatKitThreadUserMessageContent> content;
  int created_at = 0;
  std::optional<InferenceOptions> inference_options;
  std::string object;
  std::string thread_id;
  std::string type;
};

struct ChatKitThreadAssistantMessageItem {
  std::string id;
  std::vector<ChatKitResponseOutputText> content;
  int created_at = 0;
  std::string object;
  std::string thread_id;
  std::string type;
};

struct ChatKitWidgetItem {
  std::string id;
  int created_at = 0;
  std::string object;
  std::string thread_id;
  std::string type;
  std::string widget;
};

struct ChatKitThreadClientToolCall {
  std::string id;
  std::string arguments;
  std::string call_id;
  int created_at = 0;
  std::string name;
  std::string object;
  std::optional<std::string> output;
  std::string status;
  std::string thread_id;
  std::string type;
};

struct ChatKitThreadTask {
  std::string id;
  int created_at = 0;
  std::optional<std::string> heading;
  std::string object;
  std::optional<std::string> summary;
  std::string task_type;
  std::string thread_id;
  std::string type;
};

struct ChatKitThreadTaskGroup {
  struct Task {
    std::optional<std::string> heading;
    std::optional<std::string> summary;
    std::string type;
  };

  std::string id;
  int created_at = 0;
  std::string object;
  std::vector<Task> tasks;
  std::string thread_id;
  std::string type;
};

struct ChatKitThreadItem {
  enum class Kind {
    AssistantMessage,
    UserMessage,
    Widget,
    ClientToolCall,
    Task,
    TaskGroup,
    Unknown
  };

  Kind kind = Kind::Unknown;
  std::optional<ChatKitThreadAssistantMessageItem> assistant_message;
  std::optional<ChatKitThreadUserMessageItem> user_message;
  std::optional<ChatKitWidgetItem> widget;
  std::optional<ChatKitThreadClientToolCall> client_tool_call;
  std::optional<ChatKitThreadTask> task;
  std::optional<ChatKitThreadTaskGroup> task_group;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadList {
  std::vector<ChatKitThread> data;
  std::optional<std::string> first_id;
  bool has_more = false;
  std::optional<std::string> last_id;
  std::optional<std::string> next_cursor;
  std::optional<std::string> object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadItemList {
  std::vector<ChatKitThreadItem> data;
  std::optional<std::string> first_id;
  bool has_more = false;
  std::optional<std::string> last_id;
  std::optional<std::string> next_cursor;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> order;
  std::optional<std::string> user;
};

struct ChatKitThreadListItemsParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> order;
};

struct ChatKitThreadDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

class ChatKitSessionsResource {
public:
  explicit ChatKitSessionsResource(OpenAIClient& client) : client_(client) {}

  ChatKitSession create(const ChatKitSessionCreateParams& params) const;
  ChatKitSession create(const ChatKitSessionCreateParams& params, const RequestOptions& options) const;

  ChatKitSession cancel(const std::string& session_id) const;
  ChatKitSession cancel(const std::string& session_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class ChatKitThreadsResource {
public:
  explicit ChatKitThreadsResource(OpenAIClient& client) : client_(client) {}

  ChatKitThread retrieve(const std::string& thread_id) const;
  ChatKitThread retrieve(const std::string& thread_id, const RequestOptions& options) const;

  ChatKitThreadList list() const;
  ChatKitThreadList list(const ChatKitThreadListParams& params) const;
  ChatKitThreadList list(const ChatKitThreadListParams& params, const RequestOptions& options) const;
  ChatKitThreadList list(const RequestOptions& options) const;

  ChatKitThreadDeleteResponse remove(const std::string& thread_id) const;
  ChatKitThreadDeleteResponse remove(const std::string& thread_id, const RequestOptions& options) const;

  ChatKitThreadItemList list_items(const std::string& thread_id) const;
  ChatKitThreadItemList list_items(const std::string& thread_id, const ChatKitThreadListItemsParams& params) const;
  ChatKitThreadItemList list_items(const std::string& thread_id,
                                   const ChatKitThreadListItemsParams& params,
                                   const RequestOptions& options) const;
  ChatKitThreadItemList list_items(const std::string& thread_id, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

class ChatKitResource {
public:
  explicit ChatKitResource(OpenAIClient& client) : sessions_(client), threads_(client) {}

  ChatKitSessionsResource& sessions() { return sessions_; }
  const ChatKitSessionsResource& sessions() const { return sessions_; }

  ChatKitThreadsResource& threads() { return threads_; }
  const ChatKitThreadsResource& threads() const { return threads_; }

private:
  ChatKitSessionsResource sessions_;
  ChatKitThreadsResource threads_;
};

}  // namespace beta

}  // namespace openai

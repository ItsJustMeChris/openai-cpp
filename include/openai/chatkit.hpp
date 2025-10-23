#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

class OpenAIClient;
struct RequestOptions;

namespace beta {

struct ChatKitSessionWorkflowParam {
  struct Tracing {
    std::optional<bool> enabled;
  };

  std::string id;
  std::optional<nlohmann::json> state_variables;
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

struct ChatKitSessionChatKitConfiguration {
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitSessionRateLimits {
  std::optional<int> max_requests_per_1_minute;
};

struct ChatKitSession {
  std::string id;
  std::string object;
  int expires_at = 0;
  std::optional<std::string> client_secret;
  std::optional<int> max_requests_per_1_minute;
  std::string status;
  std::string user;
  nlohmann::json workflow = nlohmann::json::object();
  nlohmann::json configuration = nlohmann::json::object();
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadWorkflow {
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThread {
  std::string id;
  std::string object;
  int created_at = 0;
  std::string status;
  std::string user;
  ChatKitThreadWorkflow workflow;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadList {
  std::vector<ChatKitThread> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadListParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> before;
};

struct ChatKitThreadItem {
  std::string id;
  std::string object;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadItemList {
  std::vector<ChatKitThreadItem> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct ChatKitThreadListItemsParams {
  std::optional<int> limit;
  std::optional<std::string> after;
  std::optional<std::string> before;
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

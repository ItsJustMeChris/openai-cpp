#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/responses.hpp"

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

struct ComputerScreenshotContent {
  std::optional<std::string> file_id;
  std::optional<std::string> image_url;
  std::string type;
};

struct ConversationMessageContent {
  enum class Kind {
    InputText,
    OutputText,
    Text,
    SummaryText,
    ReasoningText,
    OutputRefusal,
    InputImage,
    ComputerScreenshot,
    InputFile,
    Unknown
  };

  Kind kind = Kind::Unknown;
  std::optional<std::string> text;
  std::optional<ComputerScreenshotContent> computer_screenshot;
  std::optional<std::string> image_url;
  std::optional<std::string> file_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationMessage {
  std::string id;
  std::vector<ConversationMessageContent> content;
  std::string role;
  std::string status;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct Conversation {
  std::string id;
  int created_at = 0;
  nlohmann::json metadata = nlohmann::json::object();
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationCreateParams {
  std::optional<std::vector<ResponseInputItem>> items;
  std::optional<std::map<std::string, std::string>> metadata;
};

struct ConversationUpdateParams {
  std::optional<std::map<std::string, std::string>> metadata;
};

struct ConversationLocalShellCallAction {
  std::vector<std::string> command;
  std::map<std::string, std::string> env;
  std::optional<int> timeout_ms;
  std::optional<std::string> user;
  std::optional<std::string> working_directory;
  std::string type;
};

struct ConversationLocalShellCall {
  std::string id;
  ConversationLocalShellCallAction action;
  std::string call_id;
  std::string status;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationLocalShellCallOutput {
  std::string id;
  std::string output;
  std::optional<std::string> status;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationMcpListTool {
  std::string name;
  nlohmann::json input_schema = {};
  std::optional<nlohmann::json> annotations;
  std::optional<std::string> description;
};

struct ConversationMcpListTools {
  std::string id;
  std::string server_label;
  std::vector<ConversationMcpListTool> tools;
  std::string type;
  std::optional<std::string> error;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationMcpApprovalRequest {
  std::string id;
  std::string arguments;
  std::string name;
  std::string server_label;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationMcpApprovalResponse {
  std::string id;
  std::string approval_request_id;
  bool approve = false;
  std::string type;
  std::optional<std::string> reason;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationMcpCall {
  std::string id;
  std::string arguments;
  std::string name;
  std::string server_label;
  std::optional<std::string> approval_request_id;
  std::optional<std::string> error;
  std::optional<std::string> output;
  std::optional<std::string> status;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationImageGenerationCall {
  std::string id;
  std::optional<std::string> result;
  std::string status;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationItem {
  enum class Kind {
    Message,
    FunctionToolCall,
    FunctionToolCallOutput,
    FileSearchToolCall,
    FunctionWebSearch,
    ImageGenerationCall,
    ComputerToolCall,
    ComputerToolCallOutput,
    Reasoning,
    CodeInterpreterToolCall,
    LocalShellCall,
    LocalShellCallOutput,
    McpListTools,
    McpApprovalRequest,
    McpApprovalResponse,
    McpCall,
    CustomToolCall,
    Unknown
  };

  Kind kind = Kind::Unknown;
  std::string type;
  std::optional<ConversationMessage> message;
  std::optional<ResponseFunctionToolCall> function_tool_call;
  std::optional<ResponseFunctionToolCallOutput> function_tool_call_output;
  std::optional<ResponseFileSearchToolCall> file_search_tool_call;
  std::optional<ResponseFunctionWebSearch> function_web_search;
  std::optional<ConversationImageGenerationCall> image_generation_call;
  std::optional<ResponseComputerToolCall> computer_tool_call;
  std::optional<ResponseComputerToolCallOutput> computer_tool_call_output;
  std::optional<ResponseReasoningItemDetails> reasoning;
  std::optional<ResponseCodeInterpreterToolCall> code_interpreter_tool_call;
  std::optional<ConversationLocalShellCall> local_shell_call;
  std::optional<ConversationLocalShellCallOutput> local_shell_output;
  std::optional<ConversationMcpListTools> mcp_list_tools;
  std::optional<ConversationMcpApprovalRequest> mcp_approval_request;
  std::optional<ConversationMcpApprovalResponse> mcp_approval_response;
  std::optional<ConversationMcpCall> mcp_call;
  std::optional<ResponseCustomToolCall> custom_tool_call;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationItemList {
  std::vector<ConversationItem> data;
  std::optional<std::string> first_id;
  bool has_more = false;
  std::optional<std::string> last_id;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationItemsPage {
  std::vector<ConversationItem> data;
  std::optional<std::string> first_id;
  bool has_more = false;
  std::optional<std::string> last_id;
  std::optional<std::string> next_cursor;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct ConversationListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
};

struct ItemCreateParams {
  std::vector<ResponseInputItem> items;
  std::optional<std::vector<std::string>> include;
};

struct ItemRetrieveParams {
  std::string conversation_id;
  std::optional<std::vector<std::string>> include;
};

struct ItemListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::vector<std::string>> include;
};

struct ItemDeleteParams {
  std::string conversation_id;
};

class ConversationItemsResource;

class ConversationsResource {
public:
  explicit ConversationsResource(OpenAIClient& client);

  Conversation create(const ConversationCreateParams& params) const;
  Conversation create(const ConversationCreateParams& params, const RequestOptions& options) const;
  Conversation create() const;

  Conversation retrieve(const std::string& conversation_id) const;
  Conversation retrieve(const std::string& conversation_id, const RequestOptions& options) const;

  Conversation update(const std::string& conversation_id, const ConversationUpdateParams& params) const;
  Conversation update(const std::string& conversation_id,
                      const ConversationUpdateParams& params,
                      const RequestOptions& options) const;

  ConversationDeleted remove(const std::string& conversation_id) const;
  ConversationDeleted remove(const std::string& conversation_id, const RequestOptions& options) const;

  ConversationItemsResource& items();
  const ConversationItemsResource& items() const;

private:
  OpenAIClient& client_;
  std::unique_ptr<ConversationItemsResource> items_;
};

class ConversationItemsResource {
public:
  explicit ConversationItemsResource(OpenAIClient& client) : client_(client) {}

  ConversationItemList create(const std::string& conversation_id, const ItemCreateParams& params) const;
  ConversationItemList create(const std::string& conversation_id,
                              const ItemCreateParams& params,
                              const RequestOptions& options) const;

  ConversationItem retrieve(const std::string& conversation_id,
                            const std::string& item_id,
                            const ItemRetrieveParams& params) const;
  ConversationItem retrieve(const std::string& conversation_id,
                            const std::string& item_id,
                            const ItemRetrieveParams& params,
                            const RequestOptions& options) const;

  ConversationItemsPage list(const std::string& conversation_id, const ItemListParams& params) const;
  ConversationItemsPage list(const std::string& conversation_id,
                             const ItemListParams& params,
                             const RequestOptions& options) const;
  ConversationItemsPage list(const std::string& conversation_id) const;

  CursorPage<ConversationItem> list_page(const std::string& conversation_id, const ItemListParams& params) const;
  CursorPage<ConversationItem> list_page(const std::string& conversation_id,
                                         const ItemListParams& params,
                                         const RequestOptions& options) const;

  Conversation remove(const std::string& conversation_id, const std::string& item_id) const;
  Conversation remove(const std::string& conversation_id,
                      const std::string& item_id,
                      const ItemDeleteParams& params,
                      const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

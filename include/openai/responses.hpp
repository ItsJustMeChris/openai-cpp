#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/streaming.hpp"

namespace openai {

struct ResponseInputContent;

struct ResponseUsageInputTokensDetails {
  std::optional<int> cached_tokens;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseUsageOutputTokensDetails {
  std::optional<int> reasoning_tokens;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseUsage {
  int input_tokens = 0;
  int output_tokens = 0;
  int total_tokens = 0;
  std::optional<ResponseUsageInputTokensDetails> input_tokens_details;
  std::optional<ResponseUsageOutputTokensDetails> output_tokens_details;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseError {
  std::optional<std::string> code;
  std::string message;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseIncompleteDetails {
  std::optional<std::string> reason;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseConversationRef {
  std::string id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionToolDefinition {
  std::string name;
  std::optional<std::string> description;
  std::optional<nlohmann::json> parameters;
  std::optional<bool> strict;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCustomToolFormat {
  enum class Type {
    Text,
    Grammar,
    Unknown
  };

  Type type = Type::Unknown;
  std::optional<std::string> definition;
  std::optional<std::string> syntax;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCustomToolDefinition {
  std::string name;
  std::optional<std::string> description;
  std::optional<ResponseCustomToolFormat> format;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCustomToolCall {
  std::string call_id;
  std::string input;
  std::string name;
  std::optional<std::string> id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFileSearchRankingOptions {
  std::optional<std::string> ranker;
  std::optional<double> score_threshold;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFileSearchToolCallResult {
  std::optional<nlohmann::json> attributes;
  std::optional<std::string> file_id;
  std::optional<std::string> filename;
  std::optional<double> score;
  std::optional<std::string> text;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFileSearchToolCall {
  std::string id;
  std::vector<std::string> queries;
  std::string status;
  std::vector<ResponseFileSearchToolCallResult> results;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseSearchFilter {
  struct Comparison {
    enum class Operator {
      Eq,
      Ne,
      Gt,
      Gte,
      Lt,
      Lte,
      In,
      Nin,
      Unknown
    };

    std::string key;
    Operator op = Operator::Unknown;
    nlohmann::json value;
  };

  struct Compound {
    enum class Logical {
      And,
      Or,
      Unknown
    };

    Logical logical = Logical::Unknown;
    std::vector<ResponseSearchFilter> filters;
  };

  enum class Type {
    Comparison,
    Compound,
    Unknown
  };

  Type type = Type::Unknown;
  std::optional<Comparison> comparison;
  std::optional<Compound> compound;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFileSearchToolDefinition {
  std::vector<std::string> vector_store_ids;
  std::optional<ResponseSearchFilter> filters;
  std::optional<int> max_num_results;
  std::optional<ResponseFileSearchRankingOptions> ranking_options;
  nlohmann::json raw = nlohmann::json::object();
};

enum class ResponseComputerEnvironment {
  Windows,
  Mac,
  Linux,
  Ubuntu,
  Browser,
  Unknown
};

struct ResponseComputerToolDefinition {
  int display_height = 0;
  int display_width = 0;
  ResponseComputerEnvironment environment = ResponseComputerEnvironment::Unknown;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseWebSearchToolFilters {
  std::vector<std::string> allowed_domains;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseWebSearchToolUserLocation {
  std::optional<std::string> city;
  std::optional<std::string> country;
  std::optional<std::string> region;
  std::optional<std::string> timezone;
  std::optional<std::string> type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseWebSearchToolDefinition {
  std::string type = "web_search";
  std::optional<ResponseWebSearchToolFilters> filters;
  std::optional<std::string> search_context_size;
  std::optional<ResponseWebSearchToolUserLocation> user_location;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseWebSearchPreviewUserLocation {
  std::optional<std::string> city;
  std::optional<std::string> country;
  std::optional<std::string> region;
  std::optional<std::string> timezone;
  std::optional<std::string> type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseWebSearchPreviewToolDefinition {
  std::string type = "web_search_preview";
  std::optional<std::string> search_context_size;
  std::optional<ResponseWebSearchPreviewUserLocation> user_location;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCodeInterpreterAutoContainer {
  std::vector<std::string> file_ids;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCodeInterpreterToolDefinition {
  std::variant<std::monostate, std::string, ResponseCodeInterpreterAutoContainer> container;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseImageGenerationToolMask {
  std::optional<std::string> image_url;
  std::optional<std::string> file_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseImageGenerationToolDefinition {
  std::optional<std::string> background;
  std::optional<std::string> input_fidelity;
  std::optional<ResponseImageGenerationToolMask> input_image_mask;
  std::optional<std::string> model;
  std::optional<std::string> moderation;
  std::optional<int> output_compression;
  std::optional<std::string> output_format;
  std::optional<std::string> visual_quality;
  std::optional<int> width;
  std::optional<int> height;
  std::optional<std::string> aspect_ratio;
  std::optional<int> seed;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseLocalShellToolDefinition {
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpToolFilter {
  std::optional<bool> read_only;
  std::vector<std::string> tool_names;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpToolApprovalFilter {
  std::optional<ResponseMcpToolFilter> always;
  std::optional<ResponseMcpToolFilter> never;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpToolDefinition {
  std::string server_label;
  std::optional<std::vector<std::string>> allowed_tool_names;
  std::optional<ResponseMcpToolFilter> allowed_tool_filter;
  std::optional<std::string> authorization;
  std::optional<std::string> connector_id;
  std::map<std::string, std::string> headers;
  std::variant<std::monostate, std::string, ResponseMcpToolApprovalFilter> require_approval;
  std::optional<std::string> server_description;
  std::optional<std::string> server_url;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionToolCall {
  std::string id;
  std::string call_id;
  std::string name;
  std::string arguments;
  std::optional<std::string> status;
  std::optional<nlohmann::json> parsed_arguments;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionToolCallOutput {
  std::string id;
  std::string call_id;
  std::optional<std::string> output_text;
  std::optional<nlohmann::json> output_content;
  std::optional<std::string> status;
  std::optional<nlohmann::json> parsed_output_json;
  std::vector<ResponseInputContent> structured_output;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionWebSearch {
  struct Action {
    enum class Type {
      Search,
      OpenPage,
      Find,
      Unknown
    };

    struct Source {
      std::string url;
      nlohmann::json raw = nlohmann::json::object();
    };

    Type type = Type::Unknown;
    std::optional<std::string> query;
    std::optional<std::string> url;
    std::optional<std::string> pattern;
    std::vector<Source> sources;
    nlohmann::json raw = nlohmann::json::object();
  };

  std::string id;
  std::string status;
  std::vector<Action> actions;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseReasoningSummary {
  std::string text;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseReasoningContent {
  std::string text;
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseReasoningItemDetails {
  std::string id;
  std::vector<ResponseReasoningSummary> summary;
  std::vector<ResponseReasoningContent> content;
  std::optional<std::string> encrypted_content;
  std::optional<std::string> status;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCodeInterpreterLogOutput {
  std::string logs;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCodeInterpreterImageOutput {
  std::string url;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCodeInterpreterToolCall {
  std::string id;
  std::optional<std::string> code;
  std::string container_id;
  std::vector<ResponseCodeInterpreterLogOutput> log_outputs;
  std::vector<ResponseCodeInterpreterImageOutput> image_outputs;
  std::optional<std::string> status;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseImageGenerationCall {
  std::string id;
  std::optional<std::string> result;
  std::string status;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseComputerToolCall {
  struct Action {
    enum class Type {
      Click,
      DoubleClick,
      Drag,
      Keypress,
      Move,
      Screenshot,
      Scroll,
      Type,
      Wait,
      Unknown
    };

    struct DragPathPoint {
      int x = 0;
      int y = 0;
    };

    Type type = Type::Unknown;
    std::optional<std::string> button;
    std::optional<int> x;
    std::optional<int> y;
    std::vector<DragPathPoint> path;
    std::vector<std::string> keys;
    std::optional<int> scroll_x;
    std::optional<int> scroll_y;
    std::optional<std::string> text;
    nlohmann::json raw = nlohmann::json::object();
  };

  struct PendingSafetyCheck {
    std::string id;
    std::string code;
    std::string message;
    nlohmann::json raw = nlohmann::json::object();
  };

  std::string id;
  std::string call_id;
  std::string status;
  Action action;
  std::vector<PendingSafetyCheck> pending_safety_checks;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseComputerToolCallOutputScreenshot {
  std::optional<std::string> file_id;
  std::optional<std::string> image_url;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseComputerToolCallOutput {
  std::string id;
  std::string call_id;
  ResponseComputerToolCallOutputScreenshot screenshot;
  std::vector<ResponseComputerToolCall::PendingSafetyCheck> acknowledged_safety_checks;
  std::optional<std::string> status;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseLocalShellCall {
  struct Action {
    enum class Type {
      Exec,
      Unknown
    };

    Type type = Type::Unknown;
    std::vector<std::string> command;
    std::map<std::string, std::string> env;
    std::optional<int> timeout_ms;
    std::optional<std::string> user;
    std::optional<std::string> working_directory;
    nlohmann::json raw = nlohmann::json::object();
  };

  std::string id;
  std::string call_id;
  std::optional<std::string> status;
  Action action;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseLocalShellOutput {
  std::string id;
  std::string output;
  std::optional<std::string> status;
  std::optional<nlohmann::json> parsed_output;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpCall {
  enum class Status {
    InProgress,
    Completed,
    Incomplete,
    Calling,
    Failed,
    Unknown
  };

  std::string id;
  std::string arguments;
  std::string name;
  std::string server_label;
  Status status = Status::Unknown;
  std::optional<std::string> approval_request_id;
  std::optional<std::string> error;
  std::optional<std::string> output;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpListToolsItem {
  std::string name;
  nlohmann::json input_schema = nlohmann::json::object();
  std::optional<std::string> description;
  std::optional<std::vector<std::string>> tags;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpListTools {
  std::string id;
  std::string server_label;
  std::vector<ResponseMcpListToolsItem> tools;
  std::optional<std::string> error;
  std::optional<std::string> next_page_token;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpApprovalRequest {
  enum class Decision {
    Pending,
    Approved,
    Rejected,
    Unknown
  };

  std::string id;
  std::string arguments;
  std::optional<std::string> name;
  std::optional<std::string> server_label;
  std::optional<Decision> suggested_decision;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseMcpApprovalResponse {
  enum class Decision {
    Approved,
    Rejected,
    Unknown
  };

  std::string id;
  Decision decision = Decision::Unknown;
  std::optional<std::string> reason;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolDefinition {
  std::string type;
  std::optional<ResponseFunctionToolDefinition> function;
  std::optional<ResponseFileSearchToolDefinition> file_search;
  std::optional<ResponseComputerToolDefinition> computer;
  std::optional<ResponseWebSearchToolDefinition> web_search;
  std::optional<ResponseWebSearchPreviewToolDefinition> web_search_preview;
  std::optional<ResponseMcpToolDefinition> mcp;
  std::optional<ResponseCodeInterpreterToolDefinition> code_interpreter;
  std::optional<ResponseImageGenerationToolDefinition> image_generation;
  std::optional<ResponseLocalShellToolDefinition> local_shell;
  std::optional<ResponseCustomToolDefinition> custom;
  nlohmann::json raw = nlohmann::json::object();
};

enum class ResponseToolChoiceSimpleOption {
  None,
  Auto,
  Required
};

enum class ResponseToolChoiceAllowedMode {
  Auto,
  Required
};

struct ResponseToolChoiceAllowed {
  ResponseToolChoiceAllowedMode mode = ResponseToolChoiceAllowedMode::Auto;
  std::vector<ResponseToolDefinition> tools;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolChoiceFunction {
  std::string name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolChoiceCustom {
  std::string name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolChoiceMcp {
  std::string server_label;
  std::optional<std::string> name;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolChoiceTypes {
  std::string type;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseToolChoice {
  enum class Kind {
    Simple,
    Allowed,
    Function,
    Mcp,
    Types,
    Custom,
    Unknown
  };

  Kind kind = Kind::Simple;
  ResponseToolChoiceSimpleOption simple = ResponseToolChoiceSimpleOption::Auto;
  std::optional<ResponseToolChoiceAllowed> allowed;
  std::optional<ResponseToolChoiceFunction> function;
  std::optional<ResponseToolChoiceMcp> mcp;
  std::optional<ResponseToolChoiceTypes> types;
  std::optional<ResponseToolChoiceCustom> custom;
  nlohmann::json raw = nlohmann::json();
};

struct ResponseOutputTextAnnotation {
  enum class Type {
    FileCitation,
    UrlCitation,
    ContainerFileCitation,
    FilePath,
    Unknown
  };

  Type type = Type::Unknown;
  std::optional<std::string> file_id;
  std::optional<std::string> filename;
  std::optional<int> index;
  std::optional<int> start_index;
  std::optional<int> end_index;
  std::optional<std::string> title;
  std::optional<std::string> url;
  std::optional<std::string> container_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputTextLogprobTop {
  std::string token;
  std::vector<std::uint8_t> bytes;
  double logprob = 0.0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputTextLogprob {
  std::string token;
  std::vector<std::uint8_t> bytes;
  double logprob = 0.0;
  std::vector<ResponseOutputTextLogprobTop> top_logprobs;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputTextSegment {
  std::string text;
  std::vector<ResponseOutputTextAnnotation> annotations;
  std::vector<ResponseOutputTextLogprob> logprobs;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputRefusalSegment {
  std::string refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputContent {
  enum class Type {
    Text,
    Refusal,
    Raw
  };

  Type type = Type::Raw;
  std::optional<ResponseOutputTextSegment> text;
  std::optional<ResponseOutputRefusalSegment> refusal;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputMessage {
  std::string id;
  std::string role = "assistant";
  std::optional<std::string> status;
  std::vector<ResponseOutputContent> content;
  std::vector<ResponseOutputTextSegment> text_segments;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputItem {
  enum class Type {
    Message,
    FileSearchToolCall,
    FunctionToolCall,
    FunctionToolCallOutput,
    ComputerToolCall,
    ComputerToolCallOutput,
    Reasoning,
    ImageGenerationCall,
    CodeInterpreterToolCall,
    LocalShellCall,
    LocalShellOutput,
    McpCall,
    McpListTools,
    McpApprovalRequest,
    McpApprovalResponse,
    FunctionWebSearch,
    CustomToolCall,
    Raw
  };

  Type type = Type::Raw;
  std::string item_type;
  std::optional<ResponseOutputMessage> message;
  std::optional<ResponseFileSearchToolCall> file_search_call;
  std::optional<ResponseFunctionToolCall> function_call;
  std::optional<ResponseFunctionToolCallOutput> function_call_output;
  std::optional<ResponseFunctionWebSearch> web_search_call;
  std::optional<ResponseComputerToolCall> computer_call;
  std::optional<ResponseComputerToolCallOutput> computer_call_output;
  std::optional<ResponseCodeInterpreterToolCall> code_interpreter_call;
  std::optional<ResponseImageGenerationCall> image_generation_call;
  std::optional<ResponseReasoningItemDetails> reasoning;
  std::optional<ResponseCustomToolCall> custom_tool_call;
  std::optional<ResponseLocalShellCall> local_shell_call;
  std::optional<ResponseLocalShellOutput> local_shell_output;
  std::optional<ResponseMcpCall> mcp_call;
  std::optional<ResponseMcpListTools> mcp_list_tools;
  std::optional<ResponseMcpApprovalRequest> mcp_approval_request;
  std::optional<ResponseMcpApprovalResponse> mcp_approval_response;
  nlohmann::json raw_details = nlohmann::json::object();
  nlohmann::json raw = nlohmann::json::object();
};

struct Response {
  std::string id;
  std::string object;
  int created = 0;
  std::string model;
  std::optional<std::string> status;
  std::optional<ResponseError> error;
  std::optional<ResponseIncompleteDetails> incomplete_details;
  std::optional<ResponseConversationRef> conversation;
  std::map<std::string, std::string> metadata;
  std::optional<bool> background;
  std::optional<int> max_output_tokens;
  std::optional<std::string> previous_response_id;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<bool> parallel_tool_calls;
  std::vector<ResponseToolDefinition> tools;
  std::optional<ResponseToolChoice> tool_choice;
  std::vector<ResponseOutputItem> output;
  std::vector<ResponseOutputMessage> messages;
  std::string output_text;
  std::optional<ResponseUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputContent {
  enum class Type { Text, Image, File, Audio, Raw };

  Type type = Type::Text;
  std::optional<std::string> id;
  std::string text;
  std::string image_url;
  std::string image_detail;
  std::string file_id;
  std::string file_url;
  std::string filename;
  std::string audio_data;
  std::string audio_format;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputMessage {
  std::string role;
  std::vector<ResponseInputContent> content;
  std::map<std::string, std::string> metadata;
  std::optional<std::string> id;
  std::optional<std::string> status;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputTextItem {
  std::string text;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputImageItem {
  std::optional<std::string> image_url;
  std::optional<std::string> file_id;
  std::optional<std::string> detail;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputFileItem {
  std::optional<std::string> file_data;
  std::optional<std::string> file_id;
  std::optional<std::string> file_url;
  std::optional<std::string> filename;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputAudioItem {
  std::string data;
  std::string format;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputItem {
  enum class Type {
    Message,
    InputText,
    InputImage,
    InputFile,
    InputAudio,
    Raw
  };

  Type type = Type::Message;
  ResponseInputMessage message;
  std::optional<ResponseInputTextItem> input_text;
  std::optional<ResponseInputImageItem> input_image;
  std::optional<ResponseInputFileItem> input_file;
  std::optional<ResponseInputAudioItem> input_audio;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponsePrompt {
  std::string id;
  std::map<std::string, std::string> variables;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseReasoningConfig {
  std::optional<std::string> effort;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseStreamOptions {
  std::optional<bool> include_usage;
  nlohmann::json extra = nlohmann::json::object();
};

struct ResponseRequest {
  std::string model;
  std::vector<ResponseInputItem> input;
  std::map<std::string, std::string> metadata;
  std::optional<bool> background;
  std::optional<std::string> conversation_id;
  std::vector<std::string> include;
  std::optional<std::string> instructions;
  std::optional<int> max_output_tokens;
  std::optional<bool> parallel_tool_calls;
  std::optional<std::string> previous_response_id;
  std::optional<ResponsePrompt> prompt;
  std::optional<std::string> prompt_cache_key;
  std::optional<ResponseReasoningConfig> reasoning;
  std::optional<std::string> safety_identifier;
  std::optional<std::string> service_tier;
  std::optional<bool> store;
  std::optional<bool> stream;
  std::optional<ResponseStreamOptions> stream_options;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::vector<ResponseToolDefinition> tools;
  std::optional<ResponseToolChoice> tool_choice;
};

struct ResponseRetrieveOptions {
  bool stream = false;
};

struct ResponseList {
  std::vector<Response> data;
  bool has_more = false;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseItem {
  std::string type;
  std::optional<ResponseOutputItem> output_item;
  std::optional<ResponseInputTextItem> input_text;
  std::optional<ResponseInputImageItem> input_image;
  std::optional<ResponseInputFileItem> input_file;
  std::optional<ResponseInputAudioItem> input_audio;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseItemList {
  std::vector<ResponseItem> data;
  bool has_more = false;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseInputItemListParams {
  std::optional<std::vector<std::string>> include;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<int> limit;
};

struct ResponseTextDeltaEvent {
  int content_index = 0;
  std::string delta;
  std::string item_id;
  std::vector<ResponseOutputTextLogprob> logprobs;
  int output_index = 0;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseTextDoneEvent {
  int content_index = 0;
  std::string item_id;
  std::string text;
  int output_index = 0;
  int sequence_number = 0;
  std::vector<ResponseOutputTextLogprob> logprobs;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionCallArgumentsDeltaEvent {
  std::string delta;
  std::string item_id;
  int output_index = 0;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseFunctionCallArgumentsDoneEvent {
  std::string arguments;
  std::string item_id;
  std::string name;
  int output_index = 0;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseReasoningTextDeltaEvent {
  int content_index = 0;
  std::string item_id;
  int output_index = 0;
  int sequence_number = 0;
  std::string delta;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseReasoningTextDoneEvent {
  int content_index = 0;
  std::string item_id;
  int output_index = 0;
  int sequence_number = 0;
  std::string text;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseContentPartAddedEvent {
  int content_index = 0;
  int output_index = 0;
  std::string item_id;
  std::optional<ResponseOutputContent> content_part;
  std::optional<ResponseReasoningContent> reasoning_part;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseContentPartDoneEvent {
  int content_index = 0;
  int output_index = 0;
  std::string item_id;
  std::optional<ResponseOutputContent> content_part;
  std::optional<ResponseReasoningContent> reasoning_part;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputItemAddedEvent {
  ResponseOutputItem item;
  std::string item_id;
  int output_index = 0;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseOutputItemDoneEvent {
  ResponseOutputItem item;
  std::string item_id;
  int output_index = 0;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCreatedEvent {
  Response response;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseCompletedEvent {
  Response response;
  int sequence_number = 0;
  nlohmann::json raw = nlohmann::json::object();
};

struct ResponseStreamEvent {
  enum class Type {
    OutputTextDelta,
    OutputTextDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    Created,
    Completed,
    OutputItemAdded,
    OutputItemDone,
    ContentPartAdded,
    ContentPartDone,
    ReasoningTextDelta,
    ReasoningTextDone,
    Unknown
  };

  Type type = Type::Unknown;
  int sequence_number = 0;
  std::string type_name;
  std::optional<ResponseTextDeltaEvent> text_delta;
  std::optional<ResponseTextDoneEvent> text_done;
  std::optional<ResponseFunctionCallArgumentsDeltaEvent> function_arguments_delta;
  std::optional<ResponseFunctionCallArgumentsDoneEvent> function_arguments_done;
  std::optional<ResponseReasoningTextDeltaEvent> reasoning_text_delta;
  std::optional<ResponseReasoningTextDoneEvent> reasoning_text_done;
  std::optional<ResponseOutputItemAddedEvent> output_item_added;
  std::optional<ResponseOutputItemDoneEvent> output_item_done;
  std::optional<ResponseContentPartAddedEvent> content_part_added;
  std::optional<ResponseContentPartDoneEvent> content_part_done;
  std::optional<ResponseCreatedEvent> created;
  std::optional<ResponseCompletedEvent> completed;
  std::optional<std::string> event_name;
  nlohmann::json raw = nlohmann::json::object();
};

std::optional<ResponseStreamEvent> parse_response_stream_event(const struct ServerSentEvent& event);

class OpenAIClient;
template <typename Item>
class CursorPage;
struct ResponseStreamSnapshot;

class ResponsesResource {
public:
  class InputItemsResource {
  public:
    explicit InputItemsResource(OpenAIClient& client) : client_(client) {}

    ResponseItemList list(const std::string& response_id) const;
    ResponseItemList list(const std::string& response_id,
                          const ResponseInputItemListParams& params) const;
    ResponseItemList list(const std::string& response_id,
                          const ResponseInputItemListParams& params,
                          const struct RequestOptions& options) const;

  private:
    OpenAIClient& client_;
  };

  explicit ResponsesResource(OpenAIClient& client) : client_(client), input_items_(client) {}

  Response create(const ResponseRequest& request) const;
  Response create(const ResponseRequest& request, const struct RequestOptions& options) const;

  Response retrieve(const std::string& response_id) const;
  Response retrieve(const std::string& response_id,
                    const ResponseRetrieveOptions& retrieve_options,
                    const struct RequestOptions& options) const;

  void remove(const std::string& response_id) const;
  void remove(const std::string& response_id, const struct RequestOptions& options) const;

  Response cancel(const std::string& response_id) const;
  Response cancel(const std::string& response_id, const struct RequestOptions& options) const;

  ResponseList list() const;
  ResponseList list(const struct RequestOptions& options) const;
  CursorPage<Response> list_page() const;
  CursorPage<Response> list_page(const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> create_stream(const ResponseRequest& request) const;
  std::vector<ServerSentEvent> create_stream(const ResponseRequest& request,
                                             const struct RequestOptions& options) const;
  void create_stream(const ResponseRequest& request,
                     const std::function<bool(const ResponseStreamEvent&)>& on_event) const;
  void create_stream(const ResponseRequest& request,
                     const std::function<bool(const ResponseStreamEvent&)>& on_event,
                     const struct RequestOptions& options) const;

  ResponseStreamSnapshot create_stream_snapshot(const ResponseRequest& request) const;
  ResponseStreamSnapshot create_stream_snapshot(const ResponseRequest& request,
                                                const struct RequestOptions& options) const;

  std::vector<ServerSentEvent> retrieve_stream(const std::string& response_id) const;
  std::vector<ServerSentEvent> retrieve_stream(const std::string& response_id,
                                               const ResponseRetrieveOptions& retrieve_options,
                                               const struct RequestOptions& options) const;

  ResponseStreamSnapshot retrieve_stream_snapshot(const std::string& response_id) const;
  ResponseStreamSnapshot retrieve_stream_snapshot(const std::string& response_id,
                                                  const ResponseRetrieveOptions& retrieve_options,
                                                  const struct RequestOptions& options) const;

  InputItemsResource& input_items() { return input_items_; }
  const InputItemsResource& input_items() const { return input_items_; }

private:
  OpenAIClient& client_;
  InputItemsResource input_items_;
};

}  // namespace openai

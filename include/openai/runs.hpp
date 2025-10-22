#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <functional>

#include <nlohmann/json.hpp>

#include "openai/assistants.hpp"
#include "openai/thread_types.hpp"
#include "openai/messages.hpp"
#include "openai/run_steps.hpp"

namespace openai {

struct Run;
struct RunRequiredAction;
struct RunSubmitToolOutput;
struct RequestOptions;

using ToolOutputGenerator = std::function<std::vector<RunSubmitToolOutput>(const Run& run, const RunRequiredAction& action)>;

struct RunTruncationStrategy {
  enum class Type { Auto, LastMessages };
  Type type = Type::Auto;
  std::optional<int> last_messages;
};

struct RunUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  nlohmann::json extra = nlohmann::json::object();
};

struct RunLastError {
  std::string code;
  std::string message;
};

struct RunRequiredAction {
  struct ToolCall {
    std::string id;
    struct FunctionCall {
      std::string name;
      std::string arguments;
    } function;
  };

  std::vector<ToolCall> tool_calls;
};

struct RunIncompleteDetails {
  std::string reason;
};

struct Run {
  std::string id;
  std::string assistant_id;
  std::optional<int> cancelled_at;
  std::optional<int> completed_at;
  int created_at = 0;
  std::optional<int> expires_at;
  std::optional<int> failed_at;
  std::optional<RunIncompleteDetails> incomplete_details;
  std::string instructions;
  std::optional<RunLastError> last_error;
  std::optional<int> max_completion_tokens;
  std::optional<int> max_prompt_tokens;
  std::map<std::string, std::string> metadata;
  std::string model;
  std::string object;
  bool parallel_tool_calls = false;
  std::optional<RunRequiredAction> required_action;
  std::optional<AssistantResponseFormat> response_format;
  std::optional<int> started_at;
  std::string status;
  std::string thread_id;
  std::optional<AssistantToolChoice> tool_choice;
  std::vector<AssistantTool> tools;
  std::optional<RunTruncationStrategy> truncation_strategy;
  std::optional<RunUsage> usage;
  std::optional<double> temperature;
  std::optional<double> top_p;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunList {
  std::vector<Run> data;
  bool has_more = false;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunAdditionalMessage {
  std::string role;
  std::variant<std::string, std::vector<ThreadMessageContentPart>> content;
  std::vector<MessageAttachment> attachments;
  std::map<std::string, std::string> metadata;
};

struct RunCreateRequest {
  std::string assistant_id;
  std::optional<std::vector<std::string>> include;
  std::optional<std::string> additional_instructions;
  std::vector<RunAdditionalMessage> additional_messages;
  std::optional<std::string> instructions;
  std::optional<int> max_completion_tokens;
  std::optional<int> max_prompt_tokens;
  std::map<std::string, std::string> metadata;
  std::optional<std::string> model;
  std::optional<bool> parallel_tool_calls;
  std::optional<std::string> reasoning_effort;
  std::optional<AssistantResponseFormat> response_format;
  std::optional<bool> stream;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<AssistantToolChoice> tool_choice;
  std::vector<AssistantTool> tools;
  std::optional<RunTruncationStrategy> truncation_strategy;
};

struct RunRetrieveParams {
  std::string thread_id;
};

struct RunUpdateRequest {
  std::string thread_id;
  std::optional<std::map<std::string, std::string>> metadata;
};

struct RunListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::string> status;
};

struct RunCancelParams {
  std::string thread_id;
};

struct RunSubmitToolOutput {
  std::string tool_call_id;
  std::string output;
};

struct RunSubmitToolOutputsRequest {
  std::string thread_id;
  std::vector<RunSubmitToolOutput> outputs;
  std::optional<bool> stream;
};

struct ThreadCreateAndRunRequest {
  std::optional<ThreadCreateRequest> thread;
  RunCreateRequest run;
};

struct AssistantThreadEvent {
  std::string name;
  Thread thread;
};

struct AssistantRunEvent {
  std::string name;
  Run run;
};

struct AssistantRunStepEvent {
  std::string name;
  RunStep run_step;
};

struct AssistantRunStepDeltaEvent {
  std::string name;
  RunStepDeltaEvent delta;
};

struct AssistantMessageEvent {
  std::string name;
  ThreadMessage message;
};

struct AssistantMessageDeltaEvent {
  std::string name;
  ThreadMessageDeltaEvent delta;
};

struct AssistantErrorEvent {
  std::string name;
  std::string error;
};

using AssistantStreamEvent = std::variant<AssistantThreadEvent,
                                          AssistantRunEvent,
                                          AssistantRunStepEvent,
                                          AssistantRunStepDeltaEvent,
                                          AssistantMessageEvent,
                                          AssistantMessageDeltaEvent,
                                          AssistantErrorEvent>;

struct AssistantStreamSnapshot;

class OpenAIClient;

class RunsResource {
public:
  explicit RunsResource(OpenAIClient& client) : client_(client) {}

  Run create(const std::string& thread_id, const RunCreateRequest& request) const;
  Run create(const std::string& thread_id, const RunCreateRequest& request, const RequestOptions& options) const;

  Run retrieve(const std::string& thread_id, const std::string& run_id) const;
  Run retrieve(const std::string& thread_id, const std::string& run_id, const RequestOptions& options) const;

  Run update(const std::string& thread_id, const std::string& run_id, const RunUpdateRequest& request) const;
  Run update(const std::string& thread_id, const std::string& run_id, const RunUpdateRequest& request,
             const RequestOptions& options) const;

  RunList list(const std::string& thread_id) const;
  RunList list(const std::string& thread_id, const RunListParams& params) const;
  RunList list(const std::string& thread_id, const RequestOptions& options) const;
  RunList list(const std::string& thread_id, const RunListParams& params, const RequestOptions& options) const;

  Run cancel(const std::string& thread_id, const std::string& run_id) const;
  Run cancel(const std::string& thread_id, const std::string& run_id, const RequestOptions& options) const;

  Run submit_tool_outputs(const std::string& thread_id, const std::string& run_id,
                          const RunSubmitToolOutputsRequest& request) const;
  Run submit_tool_outputs(const std::string& thread_id, const std::string& run_id,
                          const RunSubmitToolOutputsRequest& request, const RequestOptions& options) const;

  Run submit_tool_outputs(const std::string& run_id, const RunSubmitToolOutputsRequest& request) const;
  Run submit_tool_outputs(const std::string& run_id,
                          const RunSubmitToolOutputsRequest& request,
                          const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> create_stream(const std::string& thread_id, const RunCreateRequest& request) const;
  std::vector<AssistantStreamEvent> create_stream(const std::string& thread_id,
                                                  const RunCreateRequest& request,
                                                  const RequestOptions& options) const;
  void create_stream(const std::string& thread_id,
                     const RunCreateRequest& request,
                     const std::function<bool(const AssistantStreamEvent&)>& on_event) const;
  void create_stream(const std::string& thread_id,
                     const RunCreateRequest& request,
                     const std::function<bool(const AssistantStreamEvent&)>& on_event,
                     const RequestOptions& options) const;

  AssistantStreamSnapshot create_stream_snapshot(const std::string& thread_id, const RunCreateRequest& request) const;
  AssistantStreamSnapshot create_stream_snapshot(const std::string& thread_id,
                                                 const RunCreateRequest& request,
                                                 const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> stream(const std::string& thread_id, const RunCreateRequest& request) const;
  std::vector<AssistantStreamEvent> stream(const std::string& thread_id,
                                           const RunCreateRequest& request,
                                           const RequestOptions& options) const;
  void stream(const std::string& thread_id,
              const RunCreateRequest& request,
              const std::function<bool(const AssistantStreamEvent&)>& on_event) const;
  void stream(const std::string& thread_id,
              const RunCreateRequest& request,
              const std::function<bool(const AssistantStreamEvent&)>& on_event,
              const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> submit_tool_outputs_stream(const std::string& thread_id,
                                                               const std::string& run_id,
                                                               const RunSubmitToolOutputsRequest& request) const;
  std::vector<AssistantStreamEvent> submit_tool_outputs_stream(const std::string& thread_id,
                                                               const std::string& run_id,
                                                               const RunSubmitToolOutputsRequest& request,
                                                               const RequestOptions& options) const;
  void submit_tool_outputs_stream(const std::string& thread_id,
                                  const std::string& run_id,
                                  const RunSubmitToolOutputsRequest& request,
                                  const std::function<bool(const AssistantStreamEvent&)>& on_event) const;
  void submit_tool_outputs_stream(const std::string& thread_id,
                                  const std::string& run_id,
                                  const RunSubmitToolOutputsRequest& request,
                                  const std::function<bool(const AssistantStreamEvent&)>& on_event,
                                  const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> submit_tool_outputs_stream(const std::string& run_id,
                                                               const RunSubmitToolOutputsRequest& request) const;
  std::vector<AssistantStreamEvent> submit_tool_outputs_stream(const std::string& run_id,
                                                               const RunSubmitToolOutputsRequest& request,
                                                               const RequestOptions& options) const;
  void submit_tool_outputs_stream(const std::string& run_id,
                                  const RunSubmitToolOutputsRequest& request,
                                  const std::function<bool(const AssistantStreamEvent&)>& on_event) const;
  void submit_tool_outputs_stream(const std::string& run_id,
                                  const RunSubmitToolOutputsRequest& request,
                                  const std::function<bool(const AssistantStreamEvent&)>& on_event,
                                  const RequestOptions& options) const;

  AssistantStreamSnapshot submit_tool_outputs_stream_snapshot(const std::string& thread_id,
                                                              const std::string& run_id,
                                                              const RunSubmitToolOutputsRequest& request) const;
  AssistantStreamSnapshot submit_tool_outputs_stream_snapshot(const std::string& thread_id,
                                                              const std::string& run_id,
                                                              const RunSubmitToolOutputsRequest& request,
                                                              const RequestOptions& options) const;

  AssistantStreamSnapshot submit_tool_outputs_stream_snapshot(const std::string& run_id,
                                                              const RunSubmitToolOutputsRequest& request) const;
  AssistantStreamSnapshot submit_tool_outputs_stream_snapshot(const std::string& run_id,
                                                              const RunSubmitToolOutputsRequest& request,
                                                              const RequestOptions& options) const;

  Run poll(const std::string& run_id, const RunRetrieveParams& params) const;
  Run poll(const std::string& run_id,
           const RunRetrieveParams& params,
           const RequestOptions& options,
           std::chrono::milliseconds poll_interval) const;

  Run create_and_run_poll(const std::string& thread_id, const RunCreateRequest& request) const;
  Run create_and_run_poll(const std::string& thread_id,
                          const RunCreateRequest& request,
                          const RequestOptions& options,
                          std::chrono::milliseconds poll_interval) const;

  Run submit_tool_outputs_and_poll(const std::string& thread_id,
                                   const std::string& run_id,
                                   const RunSubmitToolOutputsRequest& request) const;
  Run submit_tool_outputs_and_poll(const std::string& thread_id,
                                   const std::string& run_id,
                                   const RunSubmitToolOutputsRequest& request,
                                   const RequestOptions& options,
                                   std::chrono::milliseconds poll_interval) const;

  Run submit_tool_outputs_and_poll(const std::string& run_id,
                                   const RunSubmitToolOutputsRequest& request) const;
  Run submit_tool_outputs_and_poll(const std::string& run_id,
                                   const RunSubmitToolOutputsRequest& request,
                                   const RequestOptions& options,
                                   std::chrono::milliseconds poll_interval) const;

  Run resolve_required_action(const Run& run,
                              const ToolOutputGenerator& generator) const;
  Run resolve_required_action(const Run& run,
                              const ToolOutputGenerator& generator,
                              const RequestOptions& options,
                              std::chrono::milliseconds poll_interval) const;

  Run create_and_run_auto(const std::string& thread_id,
                          const RunCreateRequest& request,
                          const ToolOutputGenerator& generator) const;
  Run create_and_run_auto(const std::string& thread_id,
                          const RunCreateRequest& request,
                          const ToolOutputGenerator& generator,
                          const RequestOptions& options,
                          std::chrono::milliseconds poll_interval) const;

private:
  OpenAIClient& client_;
};

Run parse_run_json(const nlohmann::json& payload);
RunList parse_run_list_json(const nlohmann::json& payload);
nlohmann::json build_run_create_body(const RunCreateRequest& request);

}  // namespace openai

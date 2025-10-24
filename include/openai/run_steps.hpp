#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/assistants.hpp"
#include "openai/thread_types.hpp"
#include "openai/messages.hpp"

namespace openai {

struct CodeInterpreterLogOutput {
  int index = 0;
  std::string logs;
};

struct CodeInterpreterImageOutput {
  struct ImageData {
    std::optional<std::string> file_id;
  };

  int index = 0;
  std::string file_id;
  std::optional<ImageData> image;
};

struct CodeInterpreterOutput {
  enum class Type {
    Logs,
    Image
  };

  Type type = Type::Logs;
  std::optional<std::string> logs;
  std::optional<CodeInterpreterImageOutput::ImageData> image;
};

struct RunStepLastError {
  std::string code;
  std::string message;
};

struct CodeInterpreterToolCallDetails {
  std::string id;
  std::string input;
  std::vector<CodeInterpreterLogOutput> log_outputs;
  std::vector<CodeInterpreterImageOutput> image_outputs;
  std::vector<CodeInterpreterOutput> outputs;
};

struct FileSearchRankingOptions {
  std::string ranker;
  double score_threshold = 0.0;
};

struct FileSearchResultContent {
  std::string type;
  std::optional<std::string> text;
};

struct FileSearchResult {
  std::string file_id;
  std::string file_name;
  double score = 0.0;
  std::vector<FileSearchResultContent> content;
};

struct FileSearchToolCallDetails {
  std::string id;
  std::optional<FileSearchRankingOptions> ranking_options;
  std::vector<FileSearchResult> results;
};

struct FunctionToolCallDetails {
  std::string id;
  std::string name;
  std::string arguments;
  std::optional<std::string> output;
};

struct ToolCallDetails {
  enum class Type { CodeInterpreter, FileSearch, Function };
  Type type = Type::Function;
  std::optional<CodeInterpreterToolCallDetails> code_interpreter;
  std::optional<FileSearchToolCallDetails> file_search;
  std::optional<FunctionToolCallDetails> function;
};

struct ToolCallDelta {
  ToolCallDetails::Type type = ToolCallDetails::Type::Function;
  int index = 0;
  std::optional<std::string> id;
  std::optional<CodeInterpreterToolCallDetails> code_interpreter;
  std::optional<FileSearchToolCallDetails> file_search;
  std::optional<FunctionToolCallDetails> function;
};

struct MessageCreationDetails {
  std::string message_id;
};

struct RunStepDetails {
  enum class Type { MessageCreation, ToolCalls };
  Type type = Type::ToolCalls;
  std::optional<MessageCreationDetails> message_creation;
  std::vector<ToolCallDetails> tool_calls;
};

struct MessageCreationDeltaDetails {
  std::optional<std::string> message_id;
};

struct RunStepDeltaDetails {
  enum class Type { MessageCreation, ToolCalls };
  Type type = Type::ToolCalls;
  std::optional<MessageCreationDeltaDetails> message_creation;
  std::vector<ToolCallDelta> tool_calls;
};

struct RunStepDelta {
  std::optional<RunStepDeltaDetails> details;
  std::optional<RunStepDeltaDetails> step_details;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunStepDeltaEvent {
  std::string id;
  RunStepDelta delta;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunStepUsage {
  int completion_tokens = 0;
  int prompt_tokens = 0;
  int total_tokens = 0;
};

struct RunStep {
  std::string id;
  std::string assistant_id;
  std::optional<int> cancelled_at;
  std::optional<int> completed_at;
  int created_at = 0;
  std::optional<int> expired_at;
  std::optional<int> failed_at;
  std::optional<RunStepLastError> last_error;
  std::map<std::string, std::string> metadata;
  std::string object;
  std::string run_id;
  std::string status;
  RunStepDetails details;
  RunStepDetails step_details;
  std::string thread_id;
  std::optional<RunStepUsage> usage;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunStepList {
  std::vector<RunStep> data;
  bool has_more = false;
  std::optional<std::string> first_id;
  std::optional<std::string> last_id;
  nlohmann::json raw = nlohmann::json::object();
};

struct RunStepRetrieveParams {
  std::string thread_id;
  std::string run_id;
  std::optional<std::vector<std::string>> include;
};

struct RunStepListParams {
  std::string thread_id;
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
  std::optional<std::vector<std::string>> include;
};

struct RequestOptions;
class OpenAIClient;

class RunStepsResource {
public:
  explicit RunStepsResource(OpenAIClient& client) : client_(client) {}

  RunStep retrieve(const std::string& run_id, const std::string& step_id, const RunStepRetrieveParams& params) const;
  RunStep retrieve(const std::string& run_id, const std::string& step_id, const RunStepRetrieveParams& params,
                   const RequestOptions& options) const;

  RunStepList list(const std::string& run_id, const RunStepListParams& params) const;
  RunStepList list(const std::string& run_id, const RunStepListParams& params, const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

RunStep parse_run_step_json(const nlohmann::json& payload);
RunStepList parse_run_step_list_json(const nlohmann::json& payload);
RunStepDeltaEvent parse_run_step_delta_json(const nlohmann::json& payload);

}  // namespace openai

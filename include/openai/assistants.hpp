#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/streaming.hpp"

namespace openai {

struct AssistantTool {
  enum class Type {
    CodeInterpreter,
    FileSearch,
    Function
  };

  struct FileSearchOverrides {
    std::optional<int> max_num_results;
    std::optional<std::string> ranker;
    std::optional<double> score_threshold;
  };

  struct FunctionDefinition {
    std::string name;
    std::optional<std::string> description;
    nlohmann::json parameters = nlohmann::json::object();
  };

  Type type = Type::CodeInterpreter;
  std::optional<FileSearchOverrides> file_search;
  std::optional<FunctionDefinition> function;
};

struct AssistantToolResources {
  std::vector<std::string> code_interpreter_file_ids;
  std::vector<std::string> file_search_vector_store_ids;
};

struct AssistantResponseFormat {
  std::string type;
  nlohmann::json json_schema = nlohmann::json::object();
};

struct Assistant {
  std::string id;
  int created_at = 0;
  std::optional<std::string> description;
  std::optional<std::string> instructions;
  std::map<std::string, std::string> metadata;
  std::string model;
  std::optional<std::string> name;
  std::string object;
  std::vector<AssistantTool> tools;
  std::optional<AssistantResponseFormat> response_format;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<AssistantToolResources> tool_resources;
  nlohmann::json raw = nlohmann::json::object();
};

struct AssistantDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct AssistantCreateRequest {
  std::string model;
  std::optional<std::string> description;
  std::optional<std::string> instructions;
  std::optional<std::string> name;
  std::map<std::string, std::string> metadata;
  std::vector<AssistantTool> tools;
  std::optional<AssistantToolResources> tool_resources;
  std::optional<AssistantResponseFormat> response_format;
  std::optional<double> temperature;
  std::optional<double> top_p;
};

struct AssistantUpdateRequest {
  std::optional<std::string> model;
  std::optional<std::string> description;
  std::optional<std::string> instructions;
  std::optional<std::string> name;
  std::optional<std::map<std::string, std::string>> metadata;
  std::optional<std::vector<AssistantTool>> tools;
  std::optional<AssistantToolResources> tool_resources;
  std::optional<AssistantResponseFormat> response_format;
  std::optional<double> temperature;
  std::optional<double> top_p;
};

struct AssistantList {
  std::vector<Assistant> data;
  bool has_more = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct AssistantListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
  std::optional<std::string> before;
};

struct RequestOptions;
class OpenAIClient;

class AssistantsResource {
public:
  explicit AssistantsResource(OpenAIClient& client) : client_(client) {}

  Assistant create(const AssistantCreateRequest& request) const;
  Assistant create(const AssistantCreateRequest& request, const RequestOptions& options) const;

  Assistant retrieve(const std::string& assistant_id) const;
  Assistant retrieve(const std::string& assistant_id, const RequestOptions& options) const;

  Assistant update(const std::string& assistant_id, const AssistantUpdateRequest& request) const;
  Assistant update(const std::string& assistant_id, const AssistantUpdateRequest& request,
                   const RequestOptions& options) const;

  AssistantDeleteResponse remove(const std::string& assistant_id) const;
  AssistantDeleteResponse remove(const std::string& assistant_id, const RequestOptions& options) const;

  AssistantList list() const;
  AssistantList list(const AssistantListParams& params, const RequestOptions& options) const;
  AssistantList list(const AssistantListParams& params) const;
  AssistantList list(const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai


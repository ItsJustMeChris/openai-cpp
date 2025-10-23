#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

class OpenAIClient;

namespace graders {

struct LabelModelGraderMessageContent {
  std::string type;
  nlohmann::json data = nlohmann::json::object();
};

struct LabelModelGraderInput {
  std::string role;
  nlohmann::json content = nlohmann::json::array();
  std::optional<std::string> type;
};

struct LabelModelGrader {
  std::vector<LabelModelGraderInput> input;
  std::vector<std::string> labels;
  std::string model;
  std::string name;
  std::vector<std::string> passing_labels;
  std::string type;
};

struct StringCheckGrader {
  std::string input;
  std::string name;
  std::string operation;
  std::string reference;
  std::string type;
};

struct TextSimilarityGrader {
  std::string evaluation_metric;
  std::string input;
  std::string name;
  std::string reference;
  std::string type;
};

struct PythonGrader {
  std::string name;
  std::string source;
  std::string type;
  std::optional<std::string> image_tag;
};

struct ScoreModelGraderInput {
  std::string role;
  nlohmann::json content = nlohmann::json::array();
  std::optional<std::string> type;
};

struct ScoreModelGraderSamplingParams {
  std::optional<int> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
};

struct ScoreModelGrader {
  std::vector<ScoreModelGraderInput> input;
  std::string model;
  std::string name;
  std::string type;
  std::optional<std::vector<double>> range;
  std::optional<ScoreModelGraderSamplingParams> sampling_params;
};

struct MultiGrader {
  std::string calculate_output;
  std::variant<StringCheckGrader, TextSimilarityGrader, PythonGrader, ScoreModelGrader, LabelModelGrader> graders;
  std::string name;
  std::string type;
};

class GraderModelsResource {
public:
  GraderModelsResource() = default;
};

}  // namespace graders

class GradersResource {
public:
  explicit GradersResource(OpenAIClient& client) : client_(client) {}

  graders::GraderModelsResource& grader_models() { return grader_models_; }
  const graders::GraderModelsResource& grader_models() const { return grader_models_; }

private:
  OpenAIClient& client_;
  graders::GraderModelsResource grader_models_;
};

}  // namespace openai


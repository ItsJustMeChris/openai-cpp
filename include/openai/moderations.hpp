#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {

struct ModerationCategories {
  bool harassment = false;
  bool harassment_threatening = false;
  bool hate = false;
  bool hate_threatening = false;
  std::optional<bool> illicit;
  std::optional<bool> illicit_violent;
  bool self_harm = false;
  bool self_harm_instructions = false;
  bool self_harm_intent = false;
  bool sexual = false;
  bool sexual_minors = false;
  bool violence = false;
  bool violence_graphic = false;
};

struct ModerationCategoryAppliedInputTypes {
  std::vector<std::string> harassment;
  std::vector<std::string> harassment_threatening;
  std::vector<std::string> hate;
  std::vector<std::string> hate_threatening;
  std::vector<std::string> illicit;
  std::vector<std::string> illicit_violent;
  std::vector<std::string> self_harm;
  std::vector<std::string> self_harm_instructions;
  std::vector<std::string> self_harm_intent;
  std::vector<std::string> sexual;
  std::vector<std::string> sexual_minors;
  std::vector<std::string> violence;
  std::vector<std::string> violence_graphic;
};

struct ModerationCategoryScores {
  double harassment = 0.0;
  double harassment_threatening = 0.0;
  double hate = 0.0;
  double hate_threatening = 0.0;
  double illicit = 0.0;
  double illicit_violent = 0.0;
  double self_harm = 0.0;
  double self_harm_instructions = 0.0;
  double self_harm_intent = 0.0;
  double sexual = 0.0;
  double sexual_minors = 0.0;
  double violence = 0.0;
  double violence_graphic = 0.0;
};

struct Moderation {
  ModerationCategories categories;
  ModerationCategoryAppliedInputTypes category_applied_input_types;
  ModerationCategoryScores category_scores;
  bool flagged = false;
  nlohmann::json raw = nlohmann::json::object();
};

struct ModerationImageURL {
  std::string url;
  std::optional<std::string> detail;
};

struct ModerationImageInput {
  ModerationImageURL image_url;
  std::string type = "image_url";
};

struct ModerationTextInput {
  std::string text;
  std::string type = "text";
};

using ModerationMultiModalInput = std::variant<ModerationImageInput, ModerationTextInput>;

struct ModerationRequest {
  using Input = std::variant<std::string,
                             std::vector<std::string>,
                             std::vector<ModerationMultiModalInput>>;

  Input input;
  std::optional<std::string> model;
};

struct ModerationCreateResponse {
  std::string id;
  std::string model;
  std::vector<Moderation> results;
  nlohmann::json raw = nlohmann::json::object();
};

}  // namespace openai

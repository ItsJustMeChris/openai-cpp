#include "openai/moderations.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>
#include <type_traits>

namespace openai {
namespace {

using json = nlohmann::json;

json moderation_multi_modal_to_json(const ModerationMultiModalInput& input) {
  return std::visit(
      [](const auto& item) -> json {
        json obj = json::object();
        obj["type"] = item.type;
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, ModerationImageInput>) {
          json image = json::object({{"url", item.image_url.url}});
          if (item.image_url.detail) {
            image["detail"] = *item.image_url.detail;
          }
          obj["image_url"] = std::move(image);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(item)>, ModerationTextInput>) {
          obj["text"] = item.text;
        }
        return obj;
      },
      input);
}

json moderation_request_input_to_json(const ModerationRequest::Input& input) {
  return std::visit(
      [](const auto& value) -> json {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return json(value);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          return json(value);
        } else {
          json arr = json::array();
          for (const auto& item : value) {
            arr.push_back(moderation_multi_modal_to_json(item));
          }
          return arr;
        }
      },
      input);
}

std::vector<std::string> parse_string_array(const json& payload, const char* key) {
  if (!payload.contains(key) || payload.at(key).is_null()) {
    return {};
  }
  std::vector<std::string> result;
  for (const auto& item : payload.at(key)) {
    if (item.is_string()) {
      result.push_back(item.get<std::string>());
    }
  }
  return result;
}

ModerationCategories parse_categories(const json& payload) {
  ModerationCategories categories;
  categories.harassment = payload.value("harassment", false);
  categories.harassment_threatening = payload.value("harassment/threatening", false);
  categories.hate = payload.value("hate", false);
  categories.hate_threatening = payload.value("hate/threatening", false);
  if (payload.contains("illicit") && !payload.at("illicit").is_null()) {
    categories.illicit = payload.at("illicit").get<bool>();
  }
  if (payload.contains("illicit/violent") && !payload.at("illicit/violent").is_null()) {
    categories.illicit_violent = payload.at("illicit/violent").get<bool>();
  }
  categories.self_harm = payload.value("self-harm", false);
  categories.self_harm_instructions = payload.value("self-harm/instructions", false);
  categories.self_harm_intent = payload.value("self-harm/intent", false);
  categories.sexual = payload.value("sexual", false);
  categories.sexual_minors = payload.value("sexual/minors", false);
  categories.violence = payload.value("violence", false);
  categories.violence_graphic = payload.value("violence/graphic", false);
  return categories;
}

ModerationCategoryAppliedInputTypes parse_applied_input_types(const json& payload) {
  ModerationCategoryAppliedInputTypes applied;
  applied.harassment = parse_string_array(payload, "harassment");
  applied.harassment_threatening = parse_string_array(payload, "harassment/threatening");
  applied.hate = parse_string_array(payload, "hate");
  applied.hate_threatening = parse_string_array(payload, "hate/threatening");
  applied.illicit = parse_string_array(payload, "illicit");
  applied.illicit_violent = parse_string_array(payload, "illicit/violent");
  applied.self_harm = parse_string_array(payload, "self-harm");
  applied.self_harm_instructions = parse_string_array(payload, "self-harm/instructions");
  applied.self_harm_intent = parse_string_array(payload, "self-harm/intent");
  applied.sexual = parse_string_array(payload, "sexual");
  applied.sexual_minors = parse_string_array(payload, "sexual/minors");
  applied.violence = parse_string_array(payload, "violence");
  applied.violence_graphic = parse_string_array(payload, "violence/graphic");
  return applied;
}

ModerationCategoryScores parse_category_scores(const json& payload) {
  ModerationCategoryScores scores;
  scores.harassment = payload.value("harassment", 0.0);
  scores.harassment_threatening = payload.value("harassment/threatening", 0.0);
  scores.hate = payload.value("hate", 0.0);
  scores.hate_threatening = payload.value("hate/threatening", 0.0);
  scores.illicit = payload.value("illicit", 0.0);
  scores.illicit_violent = payload.value("illicit/violent", 0.0);
  scores.self_harm = payload.value("self-harm", 0.0);
  scores.self_harm_instructions = payload.value("self-harm/instructions", 0.0);
  scores.self_harm_intent = payload.value("self-harm/intent", 0.0);
  scores.sexual = payload.value("sexual", 0.0);
  scores.sexual_minors = payload.value("sexual/minors", 0.0);
  scores.violence = payload.value("violence", 0.0);
  scores.violence_graphic = payload.value("violence/graphic", 0.0);
  return scores;
}

Moderation parse_moderation(const json& payload) {
  Moderation moderation;
  moderation.raw = payload;
  if (payload.contains("categories")) {
    moderation.categories = parse_categories(payload.at("categories"));
  }
  if (payload.contains("category_applied_input_types")) {
    moderation.category_applied_input_types = parse_applied_input_types(payload.at("category_applied_input_types"));
  }
  if (payload.contains("category_scores")) {
    moderation.category_scores = parse_category_scores(payload.at("category_scores"));
  }
  moderation.flagged = payload.value("flagged", false);
  return moderation;
}

ModerationCreateResponse parse_moderation_response(const json& payload) {
  ModerationCreateResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.model = payload.value("model", "");
  if (payload.contains("results")) {
    for (const auto& result_json : payload.at("results")) {
      response.results.push_back(parse_moderation(result_json));
    }
  }
  return response;
}

}  // namespace

ModerationCreateResponse ModerationsResource::create(const ModerationRequest& request,
                                                      const RequestOptions& options) const {
  json body;
  body["input"] = moderation_request_input_to_json(request.input);
  if (request.model) {
    body["model"] = *request.model;
  }

  auto response = client_.perform_request("POST", "/moderations", body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_moderation_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse moderation response: ") + ex.what());
  }
}

ModerationCreateResponse ModerationsResource::create(const ModerationRequest& request) const {
  return create(request, RequestOptions{});
}

}  // namespace openai

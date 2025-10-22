#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/client.hpp"
#include "openai/moderations.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(ModerationsResourceTest, HandlesMultiInputArray) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "modr_1",
    "model": "omni-moderation-latest",
    "results": [
      {"flagged": false}
    ]
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  ModerationRequest request;
  request.input = std::vector<std::string>{"text one", "text two"};
  request.model = "omni-moderation-latest";

  auto result = client.moderations().create(request);
  EXPECT_EQ(result.id, "modr_1");
  ASSERT_FALSE(result.results.empty());
  EXPECT_FALSE(result.results.front().flagged);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  auto payload = nlohmann::json::parse(last_request.body);
  ASSERT_TRUE(payload.at("input").is_array());
  EXPECT_EQ(payload.at("input").size(), 2u);
  EXPECT_EQ(payload.at("model"), "omni-moderation-latest");
}


TEST(ModerationsResourceTest, HandlesMultiModalInput) {
  using namespace openai;

  auto http = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "id": "modr_multi",
    "model": "omni-moderation-latest",
    "results": [
      {
        "flagged": false,
        "category_applied_input_types": {
          "self-harm": ["text", "image"],
          "violence": ["text"],
          "sexual": null
        }
      }
    ]
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http));

  ModerationTextInput text;
  text.text = "Check this";
  ModerationImageInput image;
  image.image_url.url = "https://example.com/image.png";
  image.image_url.detail = "low";

  ModerationRequest request;
  request.input = std::vector<ModerationMultiModalInput>{text, image};

  auto result = client.moderations().create(request);
  ASSERT_FALSE(result.results.empty());
  const auto& moderation = result.results.front();
  EXPECT_FALSE(moderation.flagged);
  EXPECT_EQ(moderation.category_applied_input_types.self_harm.size(), 2u);
  EXPECT_EQ(moderation.category_applied_input_types.self_harm[1], "image");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  auto payload = nlohmann::json::parse(last_request.body);
  ASSERT_TRUE(payload.at("input").is_array());
  ASSERT_EQ(payload.at("input").size(), 2u);
  EXPECT_EQ(payload.at("input")[0].at("type"), "text");
  EXPECT_EQ(payload.at("input")[1].at("type"), "image_url");
  EXPECT_EQ(payload.at("input")[1].at("image_url").at("detail"), "low");
}

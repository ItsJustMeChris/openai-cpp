#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/images.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <optional>

namespace oait = openai::testing;

TEST(ImagesResourceTest, GenerateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{
      200,
      {},
      R"({
        "created":1,
        "background":"transparent",
        "output_format":"png",
        "quality":"high",
        "size":"1024x1024",
        "usage":{
          "input_tokens":11,
          "output_tokens":7,
          "total_tokens":18,
          "input_tokens_details":{
            "image_tokens":5,
            "text_tokens":6
          }
        },
        "data":[{"url":"https://example.com"}]
      })"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "A cute otter";

  auto response = client.images().generate(request);
  EXPECT_EQ(response.created, 1);
  ASSERT_TRUE(response.background.has_value());
  EXPECT_EQ(*response.background, "transparent");
  ASSERT_TRUE(response.output_format.has_value());
  EXPECT_EQ(*response.output_format, "png");
  ASSERT_TRUE(response.quality.has_value());
  EXPECT_EQ(*response.quality, "high");
  ASSERT_TRUE(response.size.has_value());
  EXPECT_EQ(*response.size, "1024x1024");
  ASSERT_TRUE(response.usage.has_value());
  EXPECT_EQ(response.usage->input_tokens, 11);
  EXPECT_EQ(response.usage->output_tokens, 7);
  EXPECT_EQ(response.usage->total_tokens, 18);
  ASSERT_TRUE(response.usage->input_tokens_details.has_value());
  EXPECT_EQ(response.usage->input_tokens_details->image_tokens, 5);
  EXPECT_EQ(response.usage->input_tokens_details->text_tokens, 6);
  ASSERT_EQ(response.data.size(), 1u);
  ASSERT_TRUE(response.data[0].url.has_value());
  EXPECT_EQ(*response.data[0].url, "https://example.com");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
}

TEST(ImagesResourceTest, GenerateIncludesAdvancedFieldsInBody) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"created":1,"data":[]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "A scenic mountain";
  request.model = "gpt-image-1";
  request.n = 2;
  request.size = "auto";
  request.response_format = "b64_json";
  request.quality = "medium";
  request.style = "natural";
  request.moderation = "low";
  request.output_compression = 82.5;
  request.output_format = "png";
  request.partial_images = 1;
  request.background = "auto";
  request.user = "user-123";
  request.stream = false;

  EXPECT_NO_THROW(client.images().generate(request));
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  auto body = nlohmann::json::parse(http_request.body);
  EXPECT_EQ(body.at("prompt").get<std::string>(), "A scenic mountain");
  EXPECT_EQ(body.at("model").get<std::string>(), "gpt-image-1");
  EXPECT_EQ(body.at("n").get<int>(), 2);
  EXPECT_EQ(body.at("size").get<std::string>(), "auto");
  EXPECT_EQ(body.at("response_format").get<std::string>(), "b64_json");
  EXPECT_EQ(body.at("quality").get<std::string>(), "medium");
  EXPECT_EQ(body.at("style").get<std::string>(), "natural");
  EXPECT_EQ(body.at("moderation").get<std::string>(), "low");
  EXPECT_DOUBLE_EQ(body.at("output_compression").get<double>(), 82.5);
  EXPECT_EQ(body.at("output_format").get<std::string>(), "png");
  EXPECT_EQ(body.at("partial_images").get<int>(), 1);
  EXPECT_EQ(body.at("background").get<std::string>(), "auto");
  EXPECT_EQ(body.at("user").get<std::string>(), "user-123");
  EXPECT_FALSE(body.at("stream").get<bool>());
}

TEST(ImagesResourceTest, GenerateRejectsStreamingTrue) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "A streaming image";
  request.stream = true;

  EXPECT_THROW(client.images().generate(request), OpenAIError);
  ASSERT_FALSE(mock_ptr->last_request().has_value());
}

TEST(ImagesResourceTest, GenerateStreamCollectsEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body =
      "data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"AAA\",\"background\":\"auto\","
      "\"created_at\":1,\"output_format\":\"png\",\"partial_image_index\":0,\"quality\":\"auto\",\"size\":\"auto\"}\n\n"
      "data: {\"type\":\"image_generation.completed\",\"b64_json\":\"BBB\",\"background\":\"auto\",\"created_at\":2,"
      "\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"auto\",\"usage\":{\"input_tokens\":1,"
      "\"output_tokens\":2,\"total_tokens\":3,\"input_tokens_details\":{\"image_tokens\":1,\"text_tokens\":0}}}\n\n"
      "data: [DONE]\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "Stream please";

  auto events = client.images().generate_stream(request);
  ASSERT_EQ(events.size(), 3u);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  ASSERT_TRUE(http_request.headers.count("Accept"));
  EXPECT_EQ(http_request.headers.at("Accept"), "text/event-stream");
}

TEST(ImagesResourceTest, GenerateStreamParsesTypedEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body =
      "data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"AAA\",\"background\":\"auto\","
      "\"created_at\":1,\"output_format\":\"png\",\"partial_image_index\":0,\"quality\":\"auto\",\"size\":\"auto\"}\n\n"
      "data: {\"type\":\"image_generation.completed\",\"b64_json\":\"BBB\",\"background\":\"auto\",\"created_at\":2,"
      "\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"auto\",\"usage\":{\"input_tokens\":1,"
      "\"output_tokens\":2,\"total_tokens\":3,\"input_tokens_details\":{\"image_tokens\":1,\"text_tokens\":0}}}\n\n"
      "data: [DONE]\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "Stream please";

  int partial_count = 0;
  int completed_count = 0;
  std::optional<ImageUsage> usage_from_event;

  client.images().generate_stream(request, [&](const ImageStreamEvent& event) {
    if (event.type == ImageStreamEvent::Type::ImageGenerationPartialImage) {
      ++partial_count;
      EXPECT_TRUE(event.generation_partial.has_value());
      if (event.generation_partial) {
        EXPECT_EQ(event.generation_partial->partial_image_index, 0);
        EXPECT_TRUE(event.generation_partial->b64_json.has_value());
      }
    } else if (event.type == ImageStreamEvent::Type::ImageGenerationCompleted) {
      ++completed_count;
      EXPECT_TRUE(event.generation_completed.has_value());
      if (event.generation_completed) {
        usage_from_event = event.generation_completed->usage;
      }
    }
    return true;
  });

  EXPECT_EQ(partial_count, 1);
  EXPECT_EQ(completed_count, 1);
  ASSERT_TRUE(usage_from_event.has_value());
  EXPECT_EQ(usage_from_event->input_tokens, 1);
  EXPECT_EQ(usage_from_event->output_tokens, 2);
  EXPECT_EQ(usage_from_event->total_tokens, 3);
  ASSERT_TRUE(usage_from_event->input_tokens_details.has_value());
  EXPECT_EQ(usage_from_event->input_tokens_details->image_tokens, 1);
}

TEST(ImagesResourceTest, CreateVariationSendsMultipart) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"created":1,"data":[]})"});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-image.png";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "binary";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageVariationRequest request;
  request.image.purpose = "assistants";
  request.image.file_path = tmp.string();

  auto response = client.images().create_variation(request);
  EXPECT_EQ(response.created, 1);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& req = *mock_ptr->last_request();
  ASSERT_TRUE(req.headers.count("Content-Type"));
  EXPECT_NE(req.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(req.body.find("binary"), std::string::npos);

  std::filesystem::remove(tmp);
}

TEST(ImagesResourceTest, EditRejectsStreamingTrue) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-edit.png";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "binary";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageEditRequest request;
  request.image.purpose = "assistants";
  request.image.file_path = tmp.string();
  request.prompt = "Edit this image";
  request.stream = true;

  EXPECT_THROW(client.images().edit(request), OpenAIError);
  EXPECT_EQ(mock_ptr->call_count(), 0u);

  std::filesystem::remove(tmp);
}

TEST(ImagesResourceTest, EditStreamCollectsEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body =
      "data: {\"type\":\"image_edit.partial_image\",\"b64_json\":\"AAA\",\"background\":\"auto\","
      "\"created_at\":1,\"output_format\":\"png\",\"partial_image_index\":0,\"quality\":\"auto\",\"size\":\"auto\"}\n\n"
      "data: {\"type\":\"image_edit.completed\",\"b64_json\":\"BBB\",\"background\":\"auto\",\"created_at\":2,"
      "\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"auto\",\"usage\":{\"input_tokens\":2,"
      "\"output_tokens\":3,\"total_tokens\":5,\"input_tokens_details\":{\"image_tokens\":2,\"text_tokens\":0}}}\n\n"
      "data: [DONE]\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-edit-stream.png";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "binary";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageEditRequest request;
  request.image.purpose = "assistants";
  request.image.file_path = tmp.string();
  request.prompt = "Edit this";

  auto events = client.images().edit_stream(request);
  ASSERT_EQ(events.size(), 3u);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.headers.at("Accept"), "text/event-stream");
  EXPECT_NE(http_request.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);

  std::filesystem::remove(tmp);
}

TEST(ImagesResourceTest, EditStreamParsesTypedEvents) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body =
      "data: {\"type\":\"image_edit.partial_image\",\"b64_json\":\"AAA\",\"background\":\"auto\","
      "\"created_at\":1,\"output_format\":\"png\",\"partial_image_index\":0,\"quality\":\"auto\",\"size\":\"auto\"}\n\n"
      "data: {\"type\":\"image_edit.completed\",\"b64_json\":\"BBB\",\"background\":\"auto\",\"created_at\":2,"
      "\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"auto\",\"usage\":{\"input_tokens\":2,"
      "\"output_tokens\":3,\"total_tokens\":5,\"input_tokens_details\":{\"image_tokens\":2,\"text_tokens\":0}}}\n\n"
      "data: [DONE]\n\n";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-edit-stream-typed.png";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "binary";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageEditRequest request;
  request.image.purpose = "assistants";
  request.image.file_path = tmp.string();
  request.prompt = "Edit this";

  int partial_count = 0;
  int completed_count = 0;
  std::optional<ImageUsage> usage_from_event;

  client.images().edit_stream(request, [&](const ImageStreamEvent& event) {
    if (event.type == ImageStreamEvent::Type::ImageEditPartialImage) {
      ++partial_count;
      EXPECT_TRUE(event.edit_partial.has_value());
      if (event.edit_partial) {
        EXPECT_EQ(event.edit_partial->partial_image_index, 0);
      }
    } else if (event.type == ImageStreamEvent::Type::ImageEditCompleted) {
      ++completed_count;
      EXPECT_TRUE(event.edit_completed.has_value());
      if (event.edit_completed) {
        usage_from_event = event.edit_completed->usage;
      }
    }
    return true;
  });

  EXPECT_EQ(partial_count, 1);
  EXPECT_EQ(completed_count, 1);
  ASSERT_TRUE(usage_from_event.has_value());
  EXPECT_EQ(usage_from_event->input_tokens, 2);
  EXPECT_EQ(usage_from_event->output_tokens, 3);
  EXPECT_EQ(usage_from_event->total_tokens, 5);

  std::filesystem::remove(tmp);
}

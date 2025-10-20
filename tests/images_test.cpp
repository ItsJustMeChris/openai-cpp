#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/images.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace oait = openai::testing;

TEST(ImagesResourceTest, GenerateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"created":1,"data":[{"url":"https://example.com"}]})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  ImageGenerateRequest request;
  request.prompt = "A cute otter";

  auto response = client.images().generate(request);
  EXPECT_EQ(response.created, 1);
  ASSERT_EQ(response.data.size(), 1u);
  ASSERT_TRUE(response.data[0].url.has_value());
  EXPECT_EQ(*response.data[0].url, "https://example.com");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
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

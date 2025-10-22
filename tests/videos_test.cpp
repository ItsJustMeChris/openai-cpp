#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/client.hpp"
#include "openai/videos.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(VideosResourceTest, CreateSerializesMultipart) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  const std::string response_body = R"({
    "id": "vid_123",
    "created_at": 1700000000,
    "object": "video",
    "model": "sora-2",
    "progress": 0,
    "seconds": "4",
    "size": "720x1280",
    "status": "queued"
  })";

  mock_ptr->enqueue_response(openai::HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  VideoCreateRequest request;
  request.prompt = "A cat playing piano";
  request.model = VideoModel::Sora2Pro;
  request.seconds = VideoSeconds::Eight;
  request.size = VideoSize::Size1280x720;
  request.input_reference_data = std::vector<std::uint8_t>{'a', 'b', 'c'};
  request.input_reference_filename = "ref.png";
  request.input_reference_content_type = "image/png";

  auto video = client.videos().create(request);
  EXPECT_EQ(video.id, "vid_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(http_request.url.find("/videos"), std::string::npos);
}

TEST(VideosResourceTest, ListAppliesQueryParameters) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"data":[],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  VideoListParams params;
  params.limit = 10;
  params.order = std::string("desc");
  params.after = std::string("vid_prev");

  auto list = client.videos().list(params);
  EXPECT_FALSE(list.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=10"), std::string::npos);
  EXPECT_NE(request.url.find("order=desc"), std::string::npos);
  EXPECT_NE(request.url.find("after=vid_prev"), std::string::npos);
}

TEST(VideosResourceTest, DownloadSetsBinaryAcceptHeader) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse response;
  response.status_code = 200;
  response.headers["Content-Type"] = "application/binary";
  response.body = std::string("xyz", 3);
  mock_ptr->enqueue_response(response);

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  VideoDownloadContentParams params;
  params.variant = VideoDownloadVariant::Thumbnail;

  auto content = client.videos().download_content("vid_1", params);
  EXPECT_EQ(content.data.size(), 3u);
  EXPECT_EQ(content.headers.at("Content-Type"), "application/binary");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.headers.at("Accept"), "application/binary");
  EXPECT_NE(request.url.find("variant=thumbnail"), std::string::npos);
}

TEST(VideosResourceTest, DeleteParsesResponse) {
  using namespace openai;

  auto http_mock = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  const std::string response_body = R"({
    "id": "vid_123",
    "deleted": true,
    "object": "video.deleted"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto deleted = client.videos().remove("vid_123");
  EXPECT_TRUE(deleted.deleted);
  EXPECT_EQ(deleted.object, "video.deleted");
}


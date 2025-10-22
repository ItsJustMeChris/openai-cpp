#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/uploads.hpp"
#include "support/mock_http_client.hpp"

namespace oait = openai::testing;

TEST(UploadsResourceTest, CreateUploadSendsJson) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id":"upl_123",
    "bytes":2048,
    "created_at":1700000000,
    "expires_at":1700003600,
    "filename":"dataset.jsonl",
    "object":"upload",
    "purpose":"assistants",
    "status":"pending",
    "file": {
      "id":"file_123",
      "bytes":2048,
      "created_at":1700000000,
      "filename":"dataset.jsonl",
      "object":"file",
      "purpose":"assistants"
    }
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  UploadCreateParams params;
  params.bytes = 2048;
  params.filename = "dataset.jsonl";
  params.mime_type = "application/jsonl";
  params.purpose = "assistants";
  params.expires_after = UploadCreateExpiresAfter{.anchor = "created_at", .seconds = 7200};

  auto upload = client.uploads().create(params);
  EXPECT_EQ(upload.id, "upl_123");
  EXPECT_EQ(upload.bytes, 2048u);
  EXPECT_EQ(upload.status, "pending");
  ASSERT_TRUE(upload.file.has_value());
  EXPECT_EQ(upload.file->id, "file_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_EQ(request.url, "https://api.openai.com/v1/uploads");
  const auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("filename"), "dataset.jsonl");
  EXPECT_EQ(payload.at("mime_type"), "application/jsonl");
  EXPECT_EQ(payload.at("purpose"), "assistants");
  EXPECT_EQ(payload.at("expires_after").at("anchor"), "created_at");
  EXPECT_EQ(payload.at("expires_after").at("seconds"), 7200);
}

TEST(UploadsResourceTest, CancelAndCompleteUpload) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string cancel_body =
      R"({"id":"upl_123","bytes":10,"created_at":1,"expires_at":2,"filename":"file.txt","object":"upload","purpose":"assistants","status":"cancelled"})";
  const std::string complete_body =
      R"({"id":"upl_123","bytes":10,"created_at":1,"expires_at":2,"filename":"file.txt","object":"upload","purpose":"assistants","status":"completed"})";

  mock_ptr->enqueue_response(HttpResponse{200, {}, cancel_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, complete_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto cancelled = client.uploads().cancel("upl_123");
  EXPECT_EQ(cancelled.status, "cancelled");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto cancel_request = *mock_ptr->last_request();
  EXPECT_EQ(cancel_request.url, "https://api.openai.com/v1/uploads/upl_123/cancel");

  UploadCompleteParams complete_params;
  complete_params.part_ids = {"part_1", "part_2"};
  complete_params.md5 = "abc123";

  auto completed = client.uploads().complete("upl_123", complete_params);
  EXPECT_EQ(completed.status, "completed");
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& complete_request = *mock_ptr->last_request();
  EXPECT_EQ(complete_request.url, "https://api.openai.com/v1/uploads/upl_123/complete");
  const auto complete_payload = nlohmann::json::parse(complete_request.body);
  ASSERT_TRUE(complete_payload.contains("part_ids"));
  EXPECT_EQ(complete_payload.at("part_ids").size(), 2u);
  EXPECT_EQ(complete_payload.at("md5"), "abc123");
}

TEST(UploadPartsResourceTest, CreatePartSendsMultipart) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response =
      R"({"id":"part_1","created_at":1,"object":"upload.part","upload_id":"upl_123"})";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  UploadPartCreateParams params;
  params.data = {'t', 'e', 's', 't'};
  params.filename = "chunk.bin";
  params.content_type = "application/octet-stream";

  auto part = client.uploads().parts().create("upl_123", params);
  EXPECT_EQ(part.id, "part_1");
  EXPECT_EQ(part.upload_id, "upl_123");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.url, "https://api.openai.com/v1/uploads/upl_123/parts");
  ASSERT_TRUE(request.headers.count("Content-Type"));
  EXPECT_NE(request.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(request.body.find("chunk.bin"), std::string::npos);
  EXPECT_NE(request.body.find("test"), std::string::npos);
}

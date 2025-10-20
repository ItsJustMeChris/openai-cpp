#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/files.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace oait = openai::testing;

TEST(FilesResourceTest, ListParsesFiles) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "data": [
      {
        "id": "file-1",
        "bytes": 123,
        "created_at": 1700000000,
        "filename": "doc.txt",
        "object": "file",
        "purpose": "assistants",
        "status": "processed"
      }
    ],
    "has_more": false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));
  auto list = client.files().list();

  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_FALSE(list.has_more);
  const auto& file = list.data.front();
  EXPECT_EQ(file.id, "file-1");
  EXPECT_EQ(file.filename, "doc.txt");
  ASSERT_TRUE(file.status.has_value());
  EXPECT_EQ(*file.status, "processed");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
}

TEST(FilesResourceTest, RetrieveParsesFile) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "file-xyz",
    "bytes": 456,
    "created_at": 1700000001,
    "filename": "img.png",
    "object": "file",
    "purpose": "vision"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));
  auto file = client.files().retrieve("file-xyz");

  EXPECT_EQ(file.id, "file-xyz");
  EXPECT_EQ(file.bytes, 456u);
  EXPECT_EQ(file.purpose, "vision");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
}

TEST(FilesResourceTest, DeleteParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "file-del",
    "deleted": true,
    "object": "file"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));
  auto deleted = client.files().remove("file-del");

  EXPECT_EQ(deleted.id, "file-del");
  EXPECT_TRUE(deleted.deleted);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
}

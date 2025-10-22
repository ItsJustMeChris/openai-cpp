#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/files.hpp"
#include "openai/pagination.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

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

TEST(FilesResourceTest, ListPageSupportsCursorPagination) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string first_body = R"({
    "data": [
      {"id": "file-1", "bytes": 10, "created_at": 1, "filename": "a.txt", "object": "file", "purpose": "assistants"}
    ],
    "has_more": true,
    "next_cursor": "cursor-2"
  })";

  const std::string second_body = R"({
    "data": [
      {"id": "file-2", "bytes": 11, "created_at": 2, "filename": "b.txt", "object": "file", "purpose": "assistants"}
    ],
    "has_more": false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, first_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, second_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));
  auto page = client.files().list_page();

  EXPECT_EQ(mock_ptr->call_count(), 1u);
  EXPECT_TRUE(page.has_next_page());
  ASSERT_TRUE(page.next_cursor().has_value());
  EXPECT_EQ(*page.next_cursor(), "cursor-2");
  ASSERT_EQ(page.data().size(), 1u);
  EXPECT_EQ(page.data().front().id, "file-1");

  auto next_page = page.next_page();
  EXPECT_EQ(mock_ptr->call_count(), 2u);
  EXPECT_FALSE(next_page.has_next_page());
  ASSERT_EQ(next_page.data().size(), 1u);
  EXPECT_EQ(next_page.data().front().id, "file-2");

  const auto& last_request = mock_ptr->last_request();
  ASSERT_TRUE(last_request.has_value());
  EXPECT_NE(last_request->url.find("after=cursor-2"), std::string::npos);
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

TEST(FilesResourceTest, CreateBuildsMultipartBody) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"id":"file-upload","bytes":5,"created_at":1,"filename":"upload.txt","object":"file","purpose":"assistants"})"});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-upload.txt";
  {
    std::ofstream tmp_file(tmp, std::ios::binary);
    tmp_file << "hello";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  FileUploadRequest request;
  request.purpose = "assistants";
  request.file_path = tmp.string();
  request.file_name = "upload.txt";

  auto file = client.files().create(request);
  EXPECT_EQ(file.id, "file-upload");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  ASSERT_TRUE(last_request.headers.count("Content-Type"));
  EXPECT_NE(last_request.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(last_request.body.find("assistants"), std::string::npos);
  EXPECT_NE(last_request.body.find("hello"), std::string::npos);

  std::filesystem::remove(tmp);
}

TEST(FilesResourceTest, CreateSupportsInMemoryData) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"id":"file-bytes","bytes":4,"created_at":1,"filename":"memory.txt","object":"file","purpose":"assistants"})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  FileUploadRequest request;
  request.purpose = "assistants";
  request.file_data = openai::utils::UploadFile{
      std::vector<std::uint8_t>{'t', 'e', 's', 't'},
      "memory.txt",
      std::string("text/plain")};

  auto file = client.files().create(request);
  EXPECT_EQ(file.id, "file-bytes");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_NE(last_request.body.find("test"), std::string::npos);
  ASSERT_TRUE(last_request.headers.count("Content-Type"));
  EXPECT_NE(last_request.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
}

TEST(FilesResourceTest, ContentReturnsBinaryData) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, { {"Content-Type", "application/octet-stream"} }, std::string("data")});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto content = client.files().content("file-123");
  ASSERT_EQ(content.data.size(), 4u);
  EXPECT_EQ(std::string(content.data.begin(), content.data.end()), "data");
  ASSERT_TRUE(content.headers.count("Content-Type"));
  EXPECT_EQ(content.headers.at("Content-Type"), "application/octet-stream");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_NE(mock_ptr->last_request()->url.find("/files/file-123/content"), std::string::npos);
}

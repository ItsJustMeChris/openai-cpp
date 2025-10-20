#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/vector_stores.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(VectorStoresResourceTest, CreateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "vs_123",
    "name": "My Vector Store",
    "object": "vector_store",
    "created_at": 1,
    "metadata": {"project": "demo"}
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreCreateRequest request;
  request.name = "My Vector Store";
  request.metadata["project"] = "demo";

  auto store = client.vector_stores().create(request);
  EXPECT_EQ(store.id, "vs_123");
  EXPECT_EQ(store.metadata.at("project"), "demo");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

TEST(VectorStoresResourceTest, ListParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "data": [
      {"id": "vs_1", "name": "Store 1", "object": "vector_store", "created_at": 1},
      {"id": "vs_2", "name": "Store 2", "object": "vector_store", "created_at": 2}
    ],
    "has_more": false
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto list = client.vector_stores().list();
  ASSERT_EQ(list.data.size(), 2u);
  EXPECT_EQ(list.data[1].name, "Store 2");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

TEST(VectorStoresResourceTest, UpdateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "vs_123",
    "name": "Updated",
    "object": "vector_store",
    "created_at": 1
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreUpdateRequest request;
  request.name = "Updated";

  auto store = client.vector_stores().update("vs_123", request);
  EXPECT_EQ(store.name, "Updated");
}

TEST(VectorStoresResourceTest, DeleteParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({"id":"vs_123","deleted":true})";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto result = client.vector_stores().remove("vs_123");
  EXPECT_TRUE(result.deleted);
}

TEST(VectorStoresResourceTest, AttachFileParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "vsf_123",
    "object": "vector_store.file",
    "status": "completed",
    "file_id": "file_abc"
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreFileCreateRequest request;
  request.file_id = "file_abc";

  auto file = client.vector_stores().attach_file("vs_123", request);
  EXPECT_EQ(file.file_id, "file_abc");
  EXPECT_EQ(file.status, "completed");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

TEST(VectorStoresResourceTest, ListFilesParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "data": [
      {"id": "vsf_1", "file_id": "file1", "object": "vector_store.file", "status": "completed"}
    ],
    "has_more": false
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto list = client.vector_stores().list_files("vs_123");
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].file_id, "file1");
}

TEST(VectorStoresResourceTest, DeleteFileParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({"id":"vsf_123","deleted":true})";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto result = client.vector_stores().remove_file("vs_123", "vsf_123");
  EXPECT_TRUE(result.deleted);
}

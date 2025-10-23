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
  request.name = std::string("My Vector Store");
  request.metadata = Metadata{{"project", "demo"}};
  request.file_ids = std::vector<std::string>{"file_1"};

  auto store = client.vector_stores().create(request);
  EXPECT_EQ(store.id, "vs_123");
  ASSERT_TRUE(store.metadata.has_value());
  EXPECT_EQ(store.metadata->at("project"), "demo");

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
  ASSERT_TRUE(list.data[1].name.has_value());
  EXPECT_EQ(*list.data[1].name, "Store 2");

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
  ASSERT_TRUE(store.name.has_value());
  EXPECT_EQ(*store.name, "Updated");
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
    "created_at": 0,
    "object": "vector_store.file",
    "status": "completed",
    "file_id": "file_abc",
    "vector_store_id": "vs_123",
    "usage_bytes": 0
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreFileCreateRequest request;
  request.file_id = "file_abc";

  auto file = client.vector_stores().attach_file("vs_123", request);
  EXPECT_EQ(file.id, "vsf_123");
  EXPECT_EQ(file.vector_store_id, "vs_123");
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
      {
        "id": "vsf_1",
        "file_id": "file1",
        "object": "vector_store.file",
        "status": "completed",
        "vector_store_id": "vs_123",
        "usage_bytes": 0
      }
    ],
    "has_more": false
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto list = client.vector_stores().list_files("vs_123");
  ASSERT_EQ(list.data.size(), 1u);
  EXPECT_EQ(list.data[0].id, "vsf_1");
  EXPECT_EQ(list.data[0].vector_store_id, "vs_123");
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

TEST(VectorStoresResourceTest, CreateFileBatchParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "id": "vsfb_123",
    "object": "vector_store.file_batch",
    "status": "in_progress",
    "vector_store_id": "vs_123",
    "created_at": 0,
    "file_counts": {
      "in_progress": 1,
      "completed": 0,
      "failed": 0,
      "cancelled": 0,
      "total": 1
    }
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreFileBatchCreateRequest request;
  request.file_ids = std::vector<std::string>{"file_1", "file_2"};
  request.attributes = AttributeMap{{"priority", true}};

  auto batch = client.vector_stores().create_file_batch("vs_123", request);
  EXPECT_EQ(batch.id, "vsfb_123");
  EXPECT_EQ(batch.status, "in_progress");
  EXPECT_EQ(batch.file_counts.in_progress, 1);
  EXPECT_EQ(batch.file_counts.total, 1);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

TEST(VectorStoresResourceTest, RetrieveAndCancelFileBatch) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string retrieve_body = R"({
    "id": "vsfb_123",
    "object": "vector_store.file_batch",
    "status": "completed",
    "created_at": 0,
    "vector_store_id": "vs_123",
    "file_counts": {"in_progress": 0, "completed": 1, "failed": 0, "cancelled": 0, "total": 1}
  })";
  const std::string cancel_body = R"({
    "id": "vsfb_123",
    "object": "vector_store.file_batch",
    "status": "cancelled",
    "created_at": 0,
    "vector_store_id": "vs_123",
    "file_counts": {"in_progress": 0, "completed": 1, "failed": 0, "cancelled": 1, "total": 1}
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, retrieve_body});
  mock_ptr->enqueue_response(HttpResponse{200, {}, cancel_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  auto retrieved = client.vector_stores().retrieve_file_batch("vs_123", "vsfb_123");
  EXPECT_EQ(retrieved.status, "completed");

  auto cancelled = client.vector_stores().cancel_file_batch("vs_123", "vsfb_123");
  EXPECT_EQ(cancelled.status, "cancelled");
}

TEST(VectorStoresResourceTest, SearchReturnsResults) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string body = R"({
    "data": [
      {
        "file_id": "file123",
        "filename": "doc.txt",
        "score": 0.9,
        "content": [{"type": "text", "text": "matching content"}],
        "attributes": {"project": "demo"}
      }
    ]
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  VectorStoreSearchRequest request;
  request.query = std::string("hello");
  VectorStoreFilter filter;
  VectorStoreFilter::Comparison comparison;
  comparison.key = "project";
  comparison.op = VectorStoreFilter::Comparison::Operator::Eq;
  comparison.value = std::string("demo");
  filter.expression = comparison;
  request.filters = filter;
  VectorStoreSearchRequest::RankingOptions ranking;
  ranking.ranker = "auto";
  ranking.score_threshold = 0.5;
  request.ranking_options = ranking;

  auto results = client.vector_stores().search("vs_123", request);
  ASSERT_EQ(results.data.size(), 1u);
  EXPECT_EQ(results.data[0].file_id, "file123");
  ASSERT_FALSE(results.data[0].content.empty());
  EXPECT_EQ(results.data[0].content.front().text, "matching content");
  ASSERT_TRUE(results.data[0].attributes.has_value());
  auto attr_it = results.data[0].attributes->find("project");
  ASSERT_NE(attr_it, results.data[0].attributes->end());
  EXPECT_TRUE(std::holds_alternative<std::string>(attr_it->second));
  EXPECT_EQ(std::get<std::string>(attr_it->second), "demo");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("OpenAI-Beta"), "assistants=v2");
}

#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/containers.hpp"
#include "support/mock_http_client.hpp"

#include <nlohmann/json.hpp>

namespace oait = openai::testing;

TEST(ContainersResourceTest, CreateSerializesRequest) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "cont_123",
    "created_at": 1700000000,
    "name": "demo",
    "object": "container",
    "status": "active",
    "expires_after": {"anchor": "last_active_at", "minutes": 60}
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_client));

  ContainerCreateRequest request;
  request.name = "demo";
  request.file_ids = {"file_1", "file_2"};
  request.expires_after = ContainerCreateRequest::ExpiresAfter{"last_active_at", 60};

  auto container = client.containers().create(request);
  EXPECT_EQ(container.id, "cont_123");
  ASSERT_TRUE(container.expires_after.has_value());
  EXPECT_EQ(container.expires_after->minutes, 60);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& last_request = *mock_ptr->last_request();
  EXPECT_EQ(last_request.method, "POST");
  EXPECT_NE(last_request.url.find("/containers"), std::string::npos);

  auto payload = nlohmann::json::parse(last_request.body);
  EXPECT_EQ(payload.at("name"), "demo");
  EXPECT_EQ(payload.at("file_ids").size(), 2);
  EXPECT_EQ(payload.at("expires_after").at("minutes"), 60);
}

TEST(ContainersResourceTest, ListAppliesQueryParams) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"data":[],"has_more":false})"});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_client));

  ContainerListParams params;
  params.limit = 20;
  params.order = std::string("asc");
  params.after = std::string("cont_prev");

  auto list = client.containers().list(params);
  EXPECT_FALSE(list.has_more);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_NE(request.url.find("limit=20"), std::string::npos);
  EXPECT_NE(request.url.find("order=asc"), std::string::npos);
  EXPECT_NE(request.url.find("after=cont_prev"), std::string::npos);
}

TEST(ContainerFilesResourceTest, CreateWithFileIdSendsJsonBody) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({
    "id": "file_1",
    "bytes": 123,
    "container_id": "cont_1",
    "created_at": 1,
    "object": "container.file",
    "path": "foo.txt",
    "source": "user"
  })";

  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_client));

  ContainerFileCreateRequest request;
  request.file_id = std::string("file_123");

  auto file = client.containers().files().create("cont_1", request);
  EXPECT_EQ(file.id, "file_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& http_request = *mock_ptr->last_request();
  EXPECT_EQ(http_request.method, "POST");
  EXPECT_NE(http_request.url.find("/containers/cont_1/files"), std::string::npos);
  EXPECT_EQ(http_request.headers.at("Content-Type"), "application/json");

  auto payload = nlohmann::json::parse(http_request.body);
  EXPECT_EQ(payload.at("file_id"), "file_123");
}

TEST(ContainerFilesResourceTest, DeleteSetsWildcardAccept) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{204, {}, ""});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_client));

  client.containers().files().remove("cont_1", "file_1");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "DELETE");
  EXPECT_EQ(request.headers.at("Accept"), "*/*");
}

TEST(ContainerFilesContentResourceTest, RetrieveSetsBinaryAccept) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  HttpResponse response;
  response.status_code = 200;
  response.headers["Content-Type"] = "application/octet-stream";
  response.body = std::string("abc", 3);
  mock_ptr->enqueue_response(response);

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(mock_client));

  auto content = client.containers().files().content().retrieve("cont_1", "file_1");
  EXPECT_EQ(content.data.size(), 3u);
  EXPECT_EQ(content.headers.at("Content-Type"), "application/octet-stream");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.headers.at("Accept"), "application/octet-stream");
}


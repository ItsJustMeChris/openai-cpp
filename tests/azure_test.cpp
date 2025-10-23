#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/azure.hpp"
#include "openai/completions.hpp"
#include "support/mock_http_client.hpp"

namespace openai {
namespace {

using namespace openai::testing;

TEST(AzureOpenAIClientTest, UsesApiKeyHeaderAndDeploymentRouting) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();
  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"id":"cmpl","choices":[]})"});

  AzureClientOptions options;
  options.api_key = "azure-key";
  options.api_version = "2025-01-01";
  options.endpoint = "https://example-resource.azure.openai.com";
  options.deployment = "gpt-deploy";

  AzureOpenAIClient client(std::move(options), std::move(http));

  CompletionRequest request;
  request.model = "gpt-4o";
  request.prompt = "Hello";

  client.completions().create(request);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& recorded = *mock_ptr->last_request();
  EXPECT_EQ(recorded.method, "POST");
  EXPECT_NE(recorded.url.find("/deployments/gpt-deploy/completions"), std::string::npos);
  EXPECT_NE(recorded.url.find("api-version=2025-01-01"), std::string::npos);
  EXPECT_EQ(recorded.headers.at("api-key"), "azure-key");
  EXPECT_FALSE(recorded.headers.count("Authorization"));
}

TEST(AzureOpenAIClientTest, UsesTokenProviderForAuthorization) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();
  mock_ptr->enqueue_response(HttpResponse{200, {}, R"({"id":"cmpl","choices":[]})"});

  int provider_calls = 0;
  AzureClientOptions options;
  options.api_version = "2025-02-02";
  options.endpoint = "https://example-resource.azure.openai.com";
  options.azure_ad_token_provider = [&]() {
    ++provider_calls;
    return std::string("token-123");
  };

  AzureOpenAIClient client(std::move(options), std::move(http));

  CompletionRequest request;
  request.model = "my-deployment";
  request.prompt = "Test";

  client.completions().create(request);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& recorded = *mock_ptr->last_request();
  EXPECT_EQ(provider_calls, 1);
  EXPECT_NE(recorded.url.find("/deployments/my-deployment/completions"), std::string::npos);
  EXPECT_NE(recorded.url.find("api-version=2025-02-02"), std::string::npos);
  EXPECT_EQ(recorded.headers.at("Authorization"), "Bearer token-123");
  EXPECT_FALSE(recorded.headers.count("api-key"));
}

}  // namespace
}  // namespace openai

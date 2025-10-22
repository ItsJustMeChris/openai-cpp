#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/utils/platform.hpp"
#include "openai/error.hpp"

#include "support/mock_http_client.hpp"

using openai::ClientOptions;
using openai::HttpResponse;
using openai::OpenAIClient;
using openai::HttpError;
using openai::RequestOptions;
namespace mock = openai::testing;
namespace utils = openai::utils;

TEST(OpenAIClientPlatformTest, AddsPlatformHeadersAndUserAgent) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse response;
  response.status_code = 200;
  response.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(response);

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto list = client.models().list();
  (void)list;

  const auto& captured = mock_ptr->last_request();
  ASSERT_TRUE(captured.has_value());

  const auto& headers = captured->headers;
  const auto& expected_headers = utils::platform_headers();
  for (const auto& [key, value] : expected_headers) {
    auto it = headers.find(key);
    ASSERT_NE(it, headers.end()) << "Missing header: " << key;
    EXPECT_EQ(it->second, value);
  }

  auto retry_count_it = headers.find("X-Stainless-Retry-Count");
  ASSERT_NE(retry_count_it, headers.end());
  EXPECT_EQ(retry_count_it->second, "0");

  auto timeout_it = headers.find("X-Stainless-Timeout");
  ASSERT_NE(timeout_it, headers.end());
  EXPECT_EQ(timeout_it->second, "60");

  auto it = headers.find("User-Agent");
  ASSERT_NE(it, headers.end());
  EXPECT_EQ(it->second, utils::user_agent());
}

TEST(OpenAIClientRetryTest, RetriesOnServerError) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse retry_response;
  retry_response.status_code = 500;
  retry_response.headers["retry-after-ms"] = "1";
  retry_response.body = R"({"error":{"message":"temporary"}})";
  mock_ptr->enqueue_response(retry_response);

  HttpResponse success;
  success.status_code = 200;
  success.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(success);

  ClientOptions options;
  options.api_key = "sk-test";
  options.max_retries = 1;

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto list = client.models().list();
  (void)list;

  EXPECT_EQ(mock_ptr->call_count(), 2);
  const auto& captured = mock_ptr->last_request();
  ASSERT_TRUE(captured.has_value());
  auto retry_it = captured->headers.find("X-Stainless-Retry-Count");
  ASSERT_NE(retry_it, captured->headers.end());
  EXPECT_EQ(retry_it->second, "1");
}

TEST(OpenAIClientRetryTest, DoesNotRetryOnClientError) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse error_response;
  error_response.status_code = 400;
  error_response.body = R"({"error":{"message":"bad request"}})";
  mock_ptr->enqueue_response(error_response);

  ClientOptions options;
  options.api_key = "sk-test";
  options.max_retries = 2;

  OpenAIClient client(std::move(options), std::move(http_mock));

  EXPECT_THROW(client.models().list(), HttpError);
  EXPECT_EQ(mock_ptr->call_count(), 1);
}

TEST(OpenAIClientRetryTest, PerRequestMaxRetriesOverridesClientDefault) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse first;
  first.status_code = 500;
  first.headers["retry-after-ms"] = "1";
  first.body = R"({"error":{"message":"temporary"}})";
  mock_ptr->enqueue_response(first);

  HttpResponse success;
  success.status_code = 200;
  success.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(success);

  ClientOptions options;
  options.api_key = "sk-test";
  options.max_retries = 2;

  OpenAIClient client(std::move(options), std::move(http_mock));

  RequestOptions request_options;
  request_options.max_retries = 0;

  EXPECT_THROW(client.models().list(request_options), HttpError);
  EXPECT_EQ(mock_ptr->call_count(), 1);
}

TEST(OpenAIClientRequestOptionsTest, DefaultHeadersAppliedAndOverridable) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse success;
  success.status_code = 200;
  success.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(success);

  ClientOptions options;
  options.api_key = "sk-test";
  options.default_headers["X-Test-Default"] = "alpha";
  options.default_headers["X-Remove"] = "beta";

  OpenAIClient client(std::move(options), std::move(http_mock));

  RequestOptions request_options;
  request_options.headers["X-Remove"] = std::nullopt;
  request_options.headers["X-New"] = std::string("gamma");

  auto list = client.models().list(request_options);
  (void)list;

  const auto& captured = mock_ptr->last_request();
  ASSERT_TRUE(captured.has_value());

  const auto& headers = captured->headers;
  ASSERT_NE(headers.find("X-Test-Default"), headers.end());
  EXPECT_EQ(headers.at("X-Test-Default"), "alpha");

  EXPECT_EQ(headers.find("X-Remove"), headers.end());

  auto it = headers.find("X-New");
  ASSERT_NE(it, headers.end());
  EXPECT_EQ(it->second, "gamma");
}

TEST(OpenAIClientRequestOptionsTest, DefaultQueryParametersMerged) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse success;
  success.status_code = 200;
  success.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(success);

  ClientOptions options;
  options.api_key = "sk-test";
  options.default_query["foo"] = "bar";

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto list = client.models().list();
  (void)list;

  const auto& captured = mock_ptr->last_request();
  ASSERT_TRUE(captured.has_value());
  EXPECT_NE(captured->url.find("foo=bar"), std::string::npos);
}

TEST(OpenAIClientRequestOptionsTest, RequestOptionsCanRemoveDefaultQueryParameters) {
  auto http_mock = std::make_unique<mock::MockHttpClient>();
  auto* mock_ptr = http_mock.get();

  HttpResponse first;
  first.status_code = 200;
  first.body = R"({"object":"list","data":[]})";
  mock_ptr->enqueue_response(first);

  HttpResponse second = first;
  mock_ptr->enqueue_response(second);

  ClientOptions options;
  options.api_key = "sk-test";
  options.default_query["foo"] = "bar";

  OpenAIClient client(std::move(options), std::move(http_mock));

  auto baseline = client.models().list();
  (void)baseline;

  const auto first_request = mock_ptr->last_request();
  ASSERT_TRUE(first_request.has_value());
  ASSERT_NE(first_request->url.find("foo=bar"), std::string::npos);

  RequestOptions request_options;
  request_options.query_params["foo"] = std::nullopt;
  request_options.query_params["baz"] = std::string("buzz");

  auto list = client.models().list(request_options);
  (void)list;

  const auto& captured = mock_ptr->last_request();
  ASSERT_TRUE(captured.has_value());

  EXPECT_EQ(captured->url.find("foo=bar"), std::string::npos);
  EXPECT_NE(captured->url.find("baz=buzz"), std::string::npos);
}

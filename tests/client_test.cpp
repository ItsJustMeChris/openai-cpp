#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/utils/platform.hpp"

#include "support/mock_http_client.hpp"

using openai::ClientOptions;
using openai::HttpResponse;
using openai::OpenAIClient;
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

  auto it = headers.find("User-Agent");
  ASSERT_NE(it, headers.end());
  EXPECT_EQ(it->second, utils::user_agent());
}

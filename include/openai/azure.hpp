#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "openai/client.hpp"

namespace openai {

struct AzureClientOptions : ClientOptions {
  std::optional<std::string> api_version;
  std::optional<std::string> endpoint;
  std::optional<std::string> deployment;
  std::function<std::string()> azure_ad_token_provider;
};

class AzureOpenAIClient : public OpenAIClient {
public:
  explicit AzureOpenAIClient(AzureClientOptions options,
                             std::unique_ptr<HttpClient> http_client = nullptr);

  const std::string& api_version() const { return api_version_; }
  const std::optional<std::string>& deployment_name() const { return deployment_name_; }

private:
  std::string api_version_;
  std::optional<std::string> deployment_name_;
};

}  // namespace openai


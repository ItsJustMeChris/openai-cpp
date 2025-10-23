#include "openai/azure.hpp"

#include "openai/error.hpp"
#include "openai/utils/env.hpp"

#include <algorithm>

namespace openai {
namespace {

constexpr const char* kDefaultOpenAIBaseUrl = "https://api.openai.com/v1";

ClientOptions prepare_client_options(AzureClientOptions& options) {
  ClientOptions client = options;

  std::string api_version = options.api_version.value_or("");
  if (api_version.empty()) {
    if (auto env_version = utils::read_env("OPENAI_API_VERSION")) {
      api_version = *env_version;
    }
  }
  if (api_version.empty()) {
    throw OpenAIError("The OPENAI_API_VERSION environment variable is missing or empty; either provide it, or instantiate the AzureOpenAI client with an api_version option.");
  }
  client.default_query["api-version"] = api_version;
  options.api_version = api_version;

  const bool has_token_provider = static_cast<bool>(options.azure_ad_token_provider);

  if (has_token_provider && !client.api_key.empty()) {
    throw OpenAIError("The api_key and azure_ad_token_provider arguments are mutually exclusive; only one can be passed at a time.");
  }

  if (has_token_provider) {
    client.api_key_provider = options.azure_ad_token_provider;
    client.use_bearer_auth = true;
    client.api_key.clear();
    client.alternative_auth_header.reset();
  } else {
    std::string api_key = client.api_key;
    if (api_key.empty()) {
      if (auto env_key = utils::read_env("AZURE_OPENAI_API_KEY")) {
        api_key = *env_key;
      }
    }
    if (api_key.empty()) {
      throw OpenAIError("Missing credentials. Please pass one of api_key and azure_ad_token_provider, or set the AZURE_OPENAI_API_KEY environment variable.");
    }
    client.api_key = api_key;
    client.use_bearer_auth = false;
    client.alternative_auth_header = std::string("api-key");
    client.alternative_auth_prefix.clear();
  }

  std::string endpoint = options.endpoint.value_or("");
  if (client.base_url == kDefaultOpenAIBaseUrl || client.base_url.empty()) {
    if (endpoint.empty()) {
      if (auto env_endpoint = utils::read_env("AZURE_OPENAI_ENDPOINT")) {
        endpoint = *env_endpoint;
      }
    }
    if (endpoint.empty()) {
      throw OpenAIError("Must provide one of the base_url or endpoint arguments, or the AZURE_OPENAI_ENDPOINT environment variable");
    }
    if (endpoint.back() == '/') {
      endpoint.pop_back();
    }
    client.base_url = endpoint + "/openai";
  } else {
    if (!endpoint.empty()) {
      throw OpenAIError("base_url and endpoint are mutually exclusive");
    }
  }

  if (options.deployment && options.deployment->empty()) {
    options.deployment.reset();
  }

  client.azure_deployment_routing = true;
  if (options.deployment) {
    client.azure_deployment_name = options.deployment;
  }

  options.deployment = client.azure_deployment_name;

  return client;
}

}  // namespace

AzureOpenAIClient::AzureOpenAIClient(AzureClientOptions options, std::unique_ptr<HttpClient> http_client)
    : OpenAIClient(prepare_client_options(options), std::move(http_client)),
      api_version_(options.api_version.value()),
      deployment_name_(options.deployment) {}

}  // namespace openai


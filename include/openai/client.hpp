#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "openai/completions.hpp"
#include "openai/http_client.hpp"
#include "openai/error.hpp"
#include "openai/models.hpp"

namespace openai {

struct RequestOptions {
  std::map<std::string, std::string> headers;
  std::optional<std::string> idempotency_key;
  std::optional<std::chrono::milliseconds> timeout;
};

struct ClientOptions {
  std::string api_key;
  std::optional<std::string> organization;
  std::optional<std::string> project;
  std::string base_url = "https://api.openai.com/v1";
  std::chrono::milliseconds timeout{60000};
  std::size_t max_retries = 2;
};

class OpenAIClient;

class CompletionsResource {
public:
  explicit CompletionsResource(OpenAIClient& client) : client_(client) {}

  Completion create(const CompletionRequest& request,
                    const RequestOptions& options = {}) const;

private:
  OpenAIClient& client_;
};

class ModelsResource {
public:
  explicit ModelsResource(OpenAIClient& client) : client_(client) {}

  Model retrieve(const std::string& model, const RequestOptions& options = {}) const;

  ModelList list(const RequestOptions& options = {}) const;

  ModelDeleted Delete(const std::string& model, const RequestOptions& options = {}) const;

private:
  OpenAIClient& client_;
};

class OpenAIClient {
public:
  explicit OpenAIClient(ClientOptions options,
                        std::unique_ptr<HttpClient> http_client = nullptr);

  const ClientOptions& options() const { return options_; }

  CompletionsResource& completions() { return completions_; }
  const CompletionsResource& completions() const { return completions_; }

  ModelsResource& models() { return models_; }
  const ModelsResource& models() const { return models_; }

  Completion create_completion(const CompletionRequest& request,
                               const RequestOptions& options = {});

private:
  friend class CompletionsResource;
  friend class ModelsResource;

  HttpResponse perform_request(const std::string& method,
                               const std::string& path,
                               const std::string& body,
                               const RequestOptions& options) const;

  ClientOptions options_;
  std::unique_ptr<HttpClient> http_client_;
  CompletionsResource completions_;
  ModelsResource models_;
};

}  // namespace openai

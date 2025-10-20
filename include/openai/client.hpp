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
#include "openai/embeddings.hpp"
#include "openai/chat.hpp"
#include "openai/moderations.hpp"
#include "openai/responses.hpp"
#include "openai/files.hpp"
#include "openai/images.hpp"
#include "openai/audio.hpp"

namespace openai {

struct RequestOptions {
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> query_params;
  std::optional<std::string> idempotency_key;
  std::optional<std::chrono::milliseconds> timeout;
  std::function<void(const char*, std::size_t)> on_chunk;
  bool collect_body = true;
};

struct PageRequestOptions {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> query;
  std::string body;
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

class EmbeddingsResource {
public:
  explicit EmbeddingsResource(OpenAIClient& client) : client_(client) {}

  CreateEmbeddingResponse create(const EmbeddingRequest& request,
                                 const RequestOptions& options = {}) const;

private:
  OpenAIClient& client_;
};

class ModerationsResource {
public:
  explicit ModerationsResource(OpenAIClient& client) : client_(client) {}

  ModerationCreateResponse create(const ModerationRequest& request) const;
  ModerationCreateResponse create(const ModerationRequest& request,
                                  const RequestOptions& options) const;

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

  EmbeddingsResource& embeddings() { return embeddings_; }
  const EmbeddingsResource& embeddings() const { return embeddings_; }

  ModerationsResource& moderations() { return moderations_; }
  const ModerationsResource& moderations() const { return moderations_; }

  ResponsesResource& responses() { return responses_; }
  const ResponsesResource& responses() const { return responses_; }

  FilesResource& files() { return files_; }
  const FilesResource& files() const { return files_; }

  ImagesResource& images() { return images_; }
  const ImagesResource& images() const { return images_; }

  AudioResource& audio() { return audio_; }
  const AudioResource& audio() const { return audio_; }

  ChatResource& chat() { return chat_; }
  const ChatResource& chat() const { return chat_; }

  Completion create_completion(const CompletionRequest& request,
                               const RequestOptions& options = {});

private:
  friend class CompletionsResource;
  friend class ModelsResource;
  friend class EmbeddingsResource;
  friend class ModerationsResource;
  friend class ResponsesResource;
  friend class FilesResource;
  friend class ImagesResource;
  friend class AudioTranscriptionsResource;
  friend class ChatCompletionsResource;

  HttpResponse perform_request(const std::string& method,
                               const std::string& path,
                               const std::string& body,
                               const RequestOptions& options) const;

  HttpResponse perform_request(const PageRequestOptions& options) const;

  ClientOptions options_;
  std::unique_ptr<HttpClient> http_client_;
  CompletionsResource completions_;
  ModelsResource models_;
  EmbeddingsResource embeddings_;
  ModerationsResource moderations_;
  ResponsesResource responses_;
  FilesResource files_;
  ImagesResource images_;
  AudioResource audio_;
  ChatResource chat_;
};

}  // namespace openai

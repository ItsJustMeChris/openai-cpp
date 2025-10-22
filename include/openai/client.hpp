#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/completions.hpp"
#include "openai/logging.hpp"
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
#include "openai/containers.hpp"
#include "openai/videos.hpp"
#include "openai/vector_stores.hpp"
#include "openai/fine_tuning.hpp"
#include "openai/webhooks.hpp"
#include "openai/conversations.hpp"
#include "openai/beta.hpp"
#include "openai/assistants.hpp"
#include "openai/threads.hpp"
#include "openai/messages.hpp"
#include "openai/runs.hpp"
#include "openai/run_steps.hpp"
#include "openai/batches.hpp"
#include "openai/uploads.hpp"

namespace openai {

struct RequestOptions {
  std::map<std::string, std::optional<std::string>> headers;
  std::map<std::string, std::optional<std::string>> query_params;
  std::optional<nlohmann::json> query;
  std::optional<std::string> idempotency_key;
  std::optional<std::chrono::milliseconds> timeout;
  std::optional<std::size_t> max_retries;
  std::function<void(const char*, std::size_t)> on_chunk;
  bool collect_body = true;
};

struct PageRequestOptions {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  nlohmann::json query = nlohmann::json::object();
  std::string body;
};

struct ClientOptions {
  std::string api_key;
  std::optional<std::string> organization;
  std::optional<std::string> project;
  std::string base_url = "https://api.openai.com/v1";
  std::chrono::milliseconds timeout{60000};
  std::size_t max_retries = 2;
  std::map<std::string, std::string> default_headers;
  std::map<std::string, std::string> default_query;
  std::optional<std::string> webhook_secret;
  LogLevel log_level = LogLevel::Off;
  LoggerCallback logger;
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

  VectorStoresResource& vector_stores() { return vector_stores_; }
  const VectorStoresResource& vector_stores() const { return vector_stores_; }

  AssistantsResource& assistants() { return assistants_; }
  const AssistantsResource& assistants() const { return assistants_; }

  ThreadsResource& threads() { return threads_; }
  const ThreadsResource& threads() const { return threads_; }

  ThreadMessagesResource& thread_messages() { return thread_messages_; }
  const ThreadMessagesResource& thread_messages() const { return thread_messages_; }

  RunsResource& runs() { return runs_; }
  const RunsResource& runs() const { return runs_; }

  RunStepsResource& run_steps() { return run_steps_; }
  const RunStepsResource& run_steps() const { return run_steps_; }

  UploadsResource& uploads() { return uploads_; }
  const UploadsResource& uploads() const { return uploads_; }

  ChatResource& chat() { return chat_; }
  const ChatResource& chat() const { return chat_; }

  ContainersResource& containers() { return containers_; }
  const ContainersResource& containers() const { return containers_; }

  VideosResource& videos() { return videos_; }
  const VideosResource& videos() const { return videos_; }

  FineTuningResource& fine_tuning() { return fine_tuning_; }
  const FineTuningResource& fine_tuning() const { return fine_tuning_; }

  WebhooksResource& webhooks() { return webhooks_; }
  const WebhooksResource& webhooks() const { return webhooks_; }

  ConversationsResource& conversations() { return conversations_; }
  const ConversationsResource& conversations() const { return conversations_; }

  BetaResource& beta() { return beta_; }
  const BetaResource& beta() const { return beta_; }

  BatchesResource& batches() { return batches_; }
  const BatchesResource& batches() const { return batches_; }

  Completion create_completion(const CompletionRequest& request,
                               const RequestOptions& options = {});

private:
  friend class CompletionsResource;
  friend class ModelsResource;
  friend class EmbeddingsResource;
  friend class ModerationsResource;
  friend class ResponsesResource;
  friend class ResponsesResource::InputItemsResource;
  friend class FilesResource;
  friend class ImagesResource;
  friend class AudioTranscriptionsResource;
  friend class AudioTranslationsResource;
  friend class AudioSpeechResource;
  friend class VectorStoresResource;
  friend class AssistantsResource;
  friend class ThreadsResource;
  friend class ThreadMessagesResource;
  friend class RunsResource;
  friend class RunStepsResource;
  friend class ChatCompletionsResource;
  friend class ContainersResource;
  friend class ContainerFilesResource;
  friend class ContainerFilesContentResource;
  friend class VideosResource;
  friend class FineTuningJobsResource;
  friend class FineTuningJobCheckpointsResource;
  friend class FineTuningResource;
  friend class WebhooksResource;
  friend class ConversationsResource;
  friend class ConversationItemsResource;
  friend class BetaResource;
  friend class beta::RealtimeSessionsResource;
  friend class BatchesResource;
  friend class UploadsResource;
  friend class UploadPartsResource;

  HttpResponse perform_request(const std::string& method,
                               const std::string& path,
                               const std::string& body,
                               const RequestOptions& options) const;

  HttpResponse perform_request(const PageRequestOptions& options) const;

  void log(LogLevel level, const std::string& message, const nlohmann::json& details = {}) const;

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
  VectorStoresResource vector_stores_;
  AssistantsResource assistants_;
  ThreadsResource threads_;
  ThreadMessagesResource thread_messages_;
  RunsResource runs_;
  RunStepsResource run_steps_;
  ChatResource chat_;
  ContainersResource containers_;
  VideosResource videos_;
  FineTuningResource fine_tuning_;
  WebhooksResource webhooks_;
  ConversationsResource conversations_;
  BetaResource beta_;
  BatchesResource batches_;
  UploadsResource uploads_;
};

}  // namespace openai

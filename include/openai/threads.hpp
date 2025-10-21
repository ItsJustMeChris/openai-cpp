#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "openai/assistants.hpp"
#include "openai/thread_types.hpp"
#include "openai/runs.hpp"

namespace openai {

struct ThreadCreateAndRunRequest;
struct AssistantStreamSnapshot;
struct RequestOptions;
class OpenAIClient;

class ThreadsResource {
public:
  explicit ThreadsResource(OpenAIClient& client) : client_(client) {}

  Thread create(const ThreadCreateRequest& request) const;
  Thread create(const ThreadCreateRequest& request, const RequestOptions& options) const;

  Thread retrieve(const std::string& thread_id) const;
  Thread retrieve(const std::string& thread_id, const RequestOptions& options) const;

  Thread update(const std::string& thread_id, const ThreadUpdateRequest& request) const;
  Thread update(const std::string& thread_id, const ThreadUpdateRequest& request, const RequestOptions& options) const;

  ThreadDeleteResponse remove(const std::string& thread_id) const;
  ThreadDeleteResponse remove(const std::string& thread_id, const RequestOptions& options) const;

  Run create_and_run(const ThreadCreateAndRunRequest& request) const;
  Run create_and_run(const ThreadCreateAndRunRequest& request, const RequestOptions& options) const;

  std::vector<AssistantStreamEvent> create_and_run_stream(const ThreadCreateAndRunRequest& request) const;
  std::vector<AssistantStreamEvent> create_and_run_stream(const ThreadCreateAndRunRequest& request,
                                                         const RequestOptions& options) const;

  AssistantStreamSnapshot create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request) const;
  AssistantStreamSnapshot create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request,
                                                         const RequestOptions& options) const;

  Run create_and_run_poll(const ThreadCreateAndRunRequest& request) const;
  Run create_and_run_poll(const ThreadCreateAndRunRequest& request,
                          const RequestOptions& options,
                          std::chrono::milliseconds poll_interval) const;

private:
  OpenAIClient& client_;
};

Thread parse_thread_json(const nlohmann::json& payload);

}  // namespace openai

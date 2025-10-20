#pragma once

#include "openai/http_client.hpp"
#include "openai/error.hpp"

#include <mutex>
#include <optional>
#include <queue>
#include <variant>

namespace openai::testing {

/**
 * Simple in-memory HttpClient that replays queued responses.
 * Useful for unit tests that want to avoid real network calls while
 * mirroring the Node SDK's Prism-backed approach.
 */
class MockHttpClient final : public HttpClient {
public:
  struct EnqueuedError {
    std::string message;
  };

  using Enqueued = std::variant<HttpResponse, EnqueuedError>;

  HttpResponse request(const HttpRequest& request) override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_request_ = request;
    if (responses_.empty()) {
      throw OpenAIError("MockHttpClient queue underflow");
    }

    auto next = std::move(responses_.front());
    responses_.pop();

    if (std::holds_alternative<EnqueuedError>(next)) {
      throw OpenAIError(std::get<EnqueuedError>(next).message);
    }
    auto response = std::get<HttpResponse>(next);
    if (request.on_chunk) {
      request.on_chunk(response.body.data(), response.body.size());
    }
    if (!request.collect_body) {
      response.body.clear();
    }
    return response;
  }

  void enqueue_response(HttpResponse response) {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.push(std::move(response));
  }

  void enqueue_error(std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.push(EnqueuedError{std::move(message)});
  }

  [[nodiscard]] const std::optional<HttpRequest>& last_request() const {
    return last_request_;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_ = {};
    last_request_.reset();
  }

private:
  std::queue<Enqueued> responses_;
  std::optional<HttpRequest> last_request_;
  mutable std::mutex mutex_;
};

}  // namespace openai::testing

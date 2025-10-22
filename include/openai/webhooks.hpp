#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/webhooks/events.hpp"

namespace openai {

struct RequestOptions;
class OpenAIClient;

struct WebhookVerifyOptions {
  std::optional<std::string> secret;
  std::chrono::seconds tolerance{300};
};

class WebhooksResource {
public:
  explicit WebhooksResource(OpenAIClient& client) : client_(client) {}

  bool verify_signature(const std::string& payload,
                        const std::map<std::string, std::string>& headers,
                        const WebhookVerifyOptions& options = {}) const;

  webhooks::WebhookEvent unwrap(const std::string& payload,
                                const std::map<std::string, std::string>& headers,
                                const WebhookVerifyOptions& options = {}) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

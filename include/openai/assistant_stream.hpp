#pragma once

#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "openai/runs.hpp"
#include "openai/streaming.hpp"
#include "openai/threads.hpp"

namespace openai {

class AssistantStreamParser {
public:
  using EventCallback = std::function<void(const AssistantStreamEvent&)>;

  explicit AssistantStreamParser(EventCallback callback);

  void feed(const ServerSentEvent& event);

private:
  EventCallback callback_;
};

}  // namespace openai

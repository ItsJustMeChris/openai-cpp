#pragma once

#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "openai/messages.hpp"
#include "openai/run_steps.hpp"
#include "openai/runs.hpp"
#include "openai/streaming.hpp"
#include "openai/threads.hpp"

namespace openai {

struct AssistantThreadEvent {
  std::string name;
  Thread thread;
};

struct AssistantRunEvent {
  std::string name;
  Run run;
};

struct AssistantRunStepEvent {
  std::string name;
  RunStep run_step;
};

struct AssistantRunStepDeltaEvent {
  std::string name;
  RunStepDeltaEvent delta;
};

struct AssistantMessageEvent {
  std::string name;
  ThreadMessage message;
};

struct AssistantMessageDeltaEvent {
  std::string name;
  ThreadMessageDeltaEvent delta;
};

struct AssistantErrorEvent {
  std::string name;
  std::string error;
};

using AssistantStreamEvent = std::variant<AssistantThreadEvent,
                                          AssistantRunEvent,
                                          AssistantRunStepEvent,
                                          AssistantRunStepDeltaEvent,
                                          AssistantMessageEvent,
                                          AssistantMessageDeltaEvent,
                                          AssistantErrorEvent>;

class AssistantStreamParser {
public:
  using EventCallback = std::function<void(const AssistantStreamEvent&)>;

  explicit AssistantStreamParser(EventCallback callback);

  void feed(const ServerSentEvent& event);

private:
  EventCallback callback_;
};

}  // namespace openai

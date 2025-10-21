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
  std::string event;
  Thread thread;
};

struct AssistantRunEvent {
  std::string event;
  Run run;
};

struct AssistantRunStepEvent {
  std::string event;
  RunStep run_step;
};

struct AssistantMessageEvent {
  std::string event;
  ThreadMessage message;
};

struct AssistantErrorEvent {
  std::string event;
  std::string error;
};

using AssistantStreamEvent = std::variant<AssistantThreadEvent, AssistantRunEvent, AssistantRunStepEvent, AssistantMessageEvent,
                                          AssistantErrorEvent>;

class AssistantStreamParser {
public:
  using EventCallback = std::function<void(const AssistantStreamEvent&)>;

  explicit AssistantStreamParser(EventCallback callback);

  void feed(const ServerSentEvent& event);

private:
  EventCallback callback_;
  Thread last_thread_;
  Run last_run_;
  RunStep last_step_;
  ThreadMessage last_message_;

  void dispatch_thread(const std::string& event_name, const nlohmann::json& payload);
  void dispatch_run(const std::string& event_name, const nlohmann::json& payload);
  void dispatch_run_step(const std::string& event_name, const nlohmann::json& payload);
  void dispatch_message(const std::string& event_name, const nlohmann::json& payload);
};

}  // namespace openai


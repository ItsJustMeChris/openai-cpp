#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

struct AssistantStreamSnapshot {
  void ingest(const AssistantStreamEvent& event);

  const std::vector<AssistantStreamEvent>& events() const { return events_; }
  const std::optional<Thread>& latest_thread() const { return last_thread_; }
  const std::optional<Run>& latest_run() const { return last_run_; }

  std::optional<Run> final_run() const { return last_run_; }
  std::vector<ThreadMessage> final_messages() const;
  std::vector<RunStep> final_run_steps() const;

private:
  void apply_message_event(const ThreadMessage& message);
  void apply_message_delta(const AssistantMessageDeltaEvent& delta_event);
  void apply_run_step_event(const RunStep& step);
  void apply_run_step_delta(const AssistantRunStepDeltaEvent& delta_event);

  std::vector<AssistantStreamEvent> events_;
  std::optional<Thread> last_thread_;
  std::optional<Run> last_run_;
  std::map<std::string, ThreadMessage> message_snapshots_;
  std::vector<std::string> message_order_;
  std::map<std::string, RunStep> run_step_snapshots_;
  std::vector<std::string> run_step_order_;
};

}  // namespace openai

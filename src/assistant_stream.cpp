#include "openai/assistant_stream.hpp"

#include <nlohmann/json.hpp>

#include "openai/error.hpp"

namespace openai {
namespace {

using json = nlohmann::json;

json parse_json(const std::string& data) {
  try {
    return json::parse(data);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistant stream JSON: ") + ex.what());
  }
}

}  // namespace

AssistantStreamParser::AssistantStreamParser(EventCallback callback) : callback_(std::move(callback)) {}

void AssistantStreamParser::feed(const ServerSentEvent& event) {
  if (!event.event.has_value() || event.data.empty()) {
    return;
  }
  auto payload = parse_json(event.data);
  const auto& event_name = *event.event;
  AssistantStreamEvent variant;

  if (event_name == "error") {
    variant = AssistantErrorEvent{.name = event_name, .error = payload.value("message", "")};
  } else if (event_name.rfind("thread.run.step.delta", 0) == 0) {
    variant = AssistantRunStepDeltaEvent{.name = event_name, .delta = parse_run_step_delta_json(payload)};
  } else if (event_name.rfind("thread.run.step", 0) == 0) {
    variant = AssistantRunStepEvent{.name = event_name, .run_step = parse_run_step_json(payload)};
  } else if (event_name.rfind("thread.run.", 0) == 0) {
    variant = AssistantRunEvent{.name = event_name, .run = parse_run_json(payload)};
  } else if (event_name.rfind("thread.message.delta", 0) == 0) {
    variant = AssistantMessageDeltaEvent{.name = event_name, .delta = parse_thread_message_delta_json(payload)};
  } else if (event_name.rfind("thread.message.", 0) == 0) {
    variant = AssistantMessageEvent{.name = event_name, .message = parse_thread_message_json(payload)};
  } else if (event_name.rfind("thread.", 0) == 0) {
    variant = AssistantThreadEvent{.name = event_name, .thread = parse_thread_json(payload)};
  } else {
    return;  // Unknown event; ignore silently for now.
  }

  callback_(variant);
}

}  // namespace openai

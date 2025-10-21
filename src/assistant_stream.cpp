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

AssistantThreadEvent make_thread_event(const std::string& event_name, const json& payload) {
  AssistantThreadEvent event;
  event.event = event_name;
  event.thread = parse_thread_json(payload);
  return event;
}

AssistantRunEvent make_run_event(const std::string& event_name, const json& payload) {
  AssistantRunEvent event;
  event.event = event_name;
  event.run = parse_run_json(payload);
  return event;
}

AssistantRunStepEvent make_run_step_event(const std::string& event_name, const json& payload) {
  AssistantRunStepEvent event;
  event.event = event_name;
  if (payload.contains("delta")) {
    event.run_step = parse_run_step_json(payload.at("delta"));
  } else {
    event.run_step = parse_run_step_json(payload);
  }
  return event;
}

AssistantMessageEvent make_message_event(const std::string& event_name, const json& payload) {
  AssistantMessageEvent event;
  event.event = event_name;
  if (payload.contains("delta")) {
    event.message = parse_thread_message_json(payload.at("delta"));
  } else {
    event.message = parse_thread_message_json(payload);
  }
  return event;
}

}  // namespace

AssistantStreamParser::AssistantStreamParser(EventCallback callback) : callback_(std::move(callback)) {}

void AssistantStreamParser::feed(const ServerSentEvent& event) {
  if (!event.event.has_value() || event.data.empty()) {
    return;
  }
  auto payload = parse_json(event.data);
  const auto& event_name = *event.event;

  if (event_name.rfind("thread.", 0) == 0 && event_name.find("run") == std::string::npos &&
      event_name.find("message") == std::string::npos) {
    auto ev = make_thread_event(event_name, payload);
    last_thread_ = ev.thread;
    callback_(ev);
  } else if (event_name.find("thread.run.step") != std::string::npos) {
    auto ev = make_run_step_event(event_name, payload);
    last_step_ = ev.run_step;
    callback_(ev);
  } else if (event_name.find("thread.run") != std::string::npos) {
    auto ev = make_run_event(event_name, payload);
    last_run_ = ev.run;
    callback_(ev);
  } else if (event_name.find("thread.message") != std::string::npos) {
    auto ev = make_message_event(event_name, payload);
    last_message_ = ev.message;
    callback_(ev);
  } else if (event_name == "error") {
    AssistantErrorEvent error_event{.event = event_name, .error = payload.value("message", "")};
    callback_(error_event);
  }
}

}  // namespace openai


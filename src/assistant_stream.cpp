#include "openai/assistant_stream.hpp"

#include <algorithm>

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

AssistantErrorData parse_error_data(const json& payload) {
  AssistantErrorData data;
  if (payload.contains("code") && payload["code"].is_string()) {
    data.code = payload["code"].get<std::string>();
  }
  if (payload.contains("message") && payload["message"].is_string()) {
    data.message = payload["message"].get<std::string>();
  } else {
    data.message = "";
  }
  if (payload.contains("param") && payload["param"].is_string()) {
    data.param = payload["param"].get<std::string>();
  }
  if (payload.contains("type") && payload["type"].is_string()) {
    data.type = payload["type"].get<std::string>();
  }
  data.raw = payload;
  return data;
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
    variant = AssistantErrorEvent{.event = event_name, .data = parse_error_data(payload)};
  } else if (event_name.rfind("thread.run.step.delta", 0) == 0) {
    variant = AssistantRunStepDeltaEvent{.event = event_name, .data = parse_run_step_delta_json(payload)};
  } else if (event_name.rfind("thread.run.step", 0) == 0) {
    variant = AssistantRunStepEvent{.event = event_name, .data = parse_run_step_json(payload)};
  } else if (event_name.rfind("thread.run.", 0) == 0) {
    variant = AssistantRunEvent{.event = event_name, .data = parse_run_json(payload)};
  } else if (event_name.rfind("thread.message.delta", 0) == 0) {
    variant = AssistantMessageDeltaEvent{.event = event_name, .data = parse_thread_message_delta_json(payload)};
  } else if (event_name.rfind("thread.message.", 0) == 0) {
    variant = AssistantMessageEvent{.event = event_name, .data = parse_thread_message_json(payload)};
  } else if (event_name.rfind("thread.", 0) == 0) {
    AssistantThreadEvent thread_event;
    thread_event.event = event_name;
    thread_event.data = parse_thread_json(payload);
    if (payload.contains("enabled") && payload["enabled"].is_boolean()) {
      thread_event.enabled = payload["enabled"].get<bool>();
    }
    variant = thread_event;
  } else {
    return;  // Unknown event; ignore silently for now.
  }

  callback_(variant);
}

namespace {

MessageContentPart build_text_part(const MessageContentDeltaPart& delta_part) {
  MessageContentPart part;
  part.type = MessageContentPart::Type::Text;
  if (delta_part.text) {
    part.text.value = delta_part.text->value;
    part.text.annotations = delta_part.text->annotations;
  }
  return part;
}

MessageContentPart build_image_file_part(const MessageContentDeltaPart& delta_part) {
  MessageContentPart part;
  part.type = MessageContentPart::Type::ImageFile;
  if (delta_part.image_file) {
    MessageContentPart::ImageFileData data;
    data.file_id = delta_part.image_file->file_id;
    data.detail = delta_part.image_file->detail;
    part.image_file = data;
  }
  return part;
}

MessageContentPart build_image_url_part(const MessageContentDeltaPart& delta_part) {
  MessageContentPart part;
  part.type = MessageContentPart::Type::ImageURL;
  if (delta_part.image_url) {
    MessageContentPart::ImageURLData data;
    data.url = delta_part.image_url->url;
    data.detail = delta_part.image_url->detail;
    part.image_url = data;
  }
  return part;
}

MessageContentPart build_refusal_part(const MessageContentDeltaPart& delta_part) {
  MessageContentPart part;
  part.type = MessageContentPart::Type::Refusal;
  if (delta_part.refusal) part.refusal = *delta_part.refusal;
  return part;
}

MessageContentPart build_raw_part(const MessageContentDeltaPart& delta_part) {
  MessageContentPart part;
  part.type = MessageContentPart::Type::Raw;
  part.raw = delta_part.raw;
  return part;
}

void ensure_message_part(ThreadMessage& message, int index) {
  if (index < 0) return;
  if (static_cast<size_t>(index) >= message.content.size()) {
    message.content.resize(static_cast<size_t>(index) + 1);
  }
}

void ensure_tool_call(RunStepDetails& details, int index) {
  if (index < 0) return;
  if (static_cast<size_t>(index) >= details.tool_calls.size()) {
    details.tool_calls.resize(static_cast<size_t>(index) + 1);
  }
}

}  // namespace

void AssistantStreamSnapshot::ingest(const AssistantStreamEvent& event) {
  events_.push_back(event);
  std::visit(
      [&](auto&& ev) {
        using T = std::decay_t<decltype(ev)>;
        if constexpr (std::is_same_v<T, AssistantThreadEvent>) {
          last_thread_ = ev.data;
        } else if constexpr (std::is_same_v<T, AssistantRunEvent>) {
          last_run_ = ev.data;
        } else if constexpr (std::is_same_v<T, AssistantMessageEvent>) {
          apply_message_event(ev.data);
        } else if constexpr (std::is_same_v<T, AssistantMessageDeltaEvent>) {
          apply_message_delta(ev.data);
        } else if constexpr (std::is_same_v<T, AssistantRunStepEvent>) {
          apply_run_step_event(ev.data);
        } else if constexpr (std::is_same_v<T, AssistantRunStepDeltaEvent>) {
          apply_run_step_delta(ev.data);
        }
      },
      event);
}

std::vector<ThreadMessage> AssistantStreamSnapshot::final_messages() const {
  std::vector<ThreadMessage> result;
  result.reserve(message_order_.size());
  for (const auto& id : message_order_) {
    auto it = message_snapshots_.find(id);
    if (it != message_snapshots_.end()) {
      result.push_back(it->second);
    }
  }
  return result;
}

std::vector<RunStep> AssistantStreamSnapshot::final_run_steps() const {
  std::vector<RunStep> result;
  result.reserve(run_step_order_.size());
  for (const auto& id : run_step_order_) {
    auto it = run_step_snapshots_.find(id);
    if (it != run_step_snapshots_.end()) {
      result.push_back(it->second);
    }
  }
  return result;
}

void AssistantStreamSnapshot::apply_message_event(const ThreadMessage& message) {
  auto [it, inserted] = message_snapshots_.insert_or_assign(message.id, message);
  if (inserted) {
    message_order_.push_back(message.id);
  } else if (std::find(message_order_.begin(), message_order_.end(), message.id) == message_order_.end()) {
    message_order_.push_back(message.id);
  }
}

void AssistantStreamSnapshot::apply_message_delta(const ThreadMessageDeltaEvent& delta_event) {
  const std::string& id = delta_event.id;
  auto& snapshot = message_snapshots_[id];
  if (snapshot.id.empty()) snapshot.id = id;
  if (std::find(message_order_.begin(), message_order_.end(), id) == message_order_.end()) {
    message_order_.push_back(id);
  }

  const auto& delta_body = delta_event.delta;
  if (delta_body.role) snapshot.role = *delta_body.role;

  for (const auto& part : delta_body.content) {
    ensure_message_part(snapshot, part.index);
    auto& target = snapshot.content[static_cast<size_t>(part.index)];
    switch (part.type) {
      case MessageContentDeltaPart::Type::Text:
        target = build_text_part(part);
        break;
      case MessageContentDeltaPart::Type::ImageFile:
        target = build_image_file_part(part);
        break;
      case MessageContentDeltaPart::Type::ImageURL:
        target = build_image_url_part(part);
        break;
      case MessageContentDeltaPart::Type::Refusal:
        target = build_refusal_part(part);
        break;
      case MessageContentDeltaPart::Type::Raw:
        target = build_raw_part(part);
        break;
    }
  }
}

void AssistantStreamSnapshot::apply_run_step_event(const RunStep& step) {
  auto [it, inserted] = run_step_snapshots_.insert_or_assign(step.id, step);
  if (inserted) {
    run_step_order_.push_back(step.id);
  } else if (std::find(run_step_order_.begin(), run_step_order_.end(), step.id) == run_step_order_.end()) {
    run_step_order_.push_back(step.id);
  }
}

void AssistantStreamSnapshot::apply_run_step_delta(const RunStepDeltaEvent& delta_event) {
  const auto& delta_event_body = delta_event;
  const std::string& id = delta_event_body.id;
  auto& snapshot = run_step_snapshots_[id];
  if (snapshot.id.empty()) snapshot.id = id;
  if (std::find(run_step_order_.begin(), run_step_order_.end(), id) == run_step_order_.end()) {
    run_step_order_.push_back(id);
  }

  const auto& delta_opt = delta_event_body.delta.details;
  if (!delta_opt.has_value()) return;
  const auto& delta = *delta_opt;

  if (!snapshot.details.tool_calls.empty() || delta.type == RunStepDeltaDetails::Type::ToolCalls) {
    if (snapshot.details.tool_calls.empty()) {
      snapshot.details.type = RunStepDetails::Type::ToolCalls;
    }
    for (const auto& call_delta : delta.tool_calls) {
      ensure_tool_call(snapshot.details, call_delta.index);
      auto& target = snapshot.details.tool_calls[static_cast<size_t>(call_delta.index)];
      switch (call_delta.type) {
        case ToolCallDetails::Type::Function:
          target.type = ToolCallDetails::Type::Function;
          if (call_delta.function) {
            if (!target.function) target.function.emplace();
            target.function->id = call_delta.function->id;
            target.function->name = call_delta.function->name;
            target.function->arguments = call_delta.function->arguments;
            target.function->output = call_delta.function->output;
          } else if (call_delta.id) {
            if (!target.function) target.function.emplace();
            target.function->id = *call_delta.id;
          }
          break;
        case ToolCallDetails::Type::CodeInterpreter:
          target.type = ToolCallDetails::Type::CodeInterpreter;
          if (call_delta.code_interpreter) target.code_interpreter = call_delta.code_interpreter;
          break;
        case ToolCallDetails::Type::FileSearch:
          target.type = ToolCallDetails::Type::FileSearch;
          if (call_delta.file_search) target.file_search = call_delta.file_search;
          break;
      }
    }
  }

  if (delta.type == RunStepDeltaDetails::Type::MessageCreation) {
    snapshot.details.type = RunStepDetails::Type::MessageCreation;
    if (delta.message_creation && delta.message_creation->message_id) {
      if (!snapshot.details.message_creation) snapshot.details.message_creation.emplace();
      snapshot.details.message_creation->message_id = *delta.message_creation->message_id;
    }
  }
}

}  // namespace openai

#include "openai/response_stream.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "openai/error.hpp"

namespace openai {
namespace {

using json = nlohmann::json;

std::string join_messages(const std::vector<ResponseOutputMessage>& messages) {
  std::ostringstream stream;
  for (const auto& message : messages) {
    for (const auto& segment : message.text_segments) {
      stream << segment.text;
    }
  }
  return stream.str();
}

std::string extract_item_id(const ResponseOutputItem& item) {
  if (item.message && !item.message->id.empty()) return item.message->id;
  if (item.file_search_call && !item.file_search_call->id.empty()) return item.file_search_call->id;
  if (item.function_call && !item.function_call->id.empty()) return item.function_call->id;
  if (item.function_call_output && !item.function_call_output->id.empty()) return item.function_call_output->id;
  if (item.web_search_call && !item.web_search_call->id.empty()) return item.web_search_call->id;
  if (item.computer_call && !item.computer_call->id.empty()) return item.computer_call->id;
  if (item.computer_call_output && !item.computer_call_output->id.empty()) return item.computer_call_output->id;
  if (item.reasoning && !item.reasoning->id.empty()) return item.reasoning->id;
  if (item.image_generation_call && !item.image_generation_call->id.empty()) return item.image_generation_call->id;
  if (item.code_interpreter_call && !item.code_interpreter_call->id.empty()) return item.code_interpreter_call->id;
  if (item.local_shell_call && !item.local_shell_call->id.empty()) return item.local_shell_call->id;
  if (item.local_shell_output && !item.local_shell_output->id.empty()) return item.local_shell_output->id;
  if (item.mcp_call && !item.mcp_call->id.empty()) return item.mcp_call->id;
  if (item.mcp_list_tools && !item.mcp_list_tools->id.empty()) return item.mcp_list_tools->id;
  if (item.mcp_approval_request && !item.mcp_approval_request->id.empty()) return item.mcp_approval_request->id;
  if (item.mcp_approval_response && !item.mcp_approval_response->id.empty()) return item.mcp_approval_response->id;
  if (item.custom_tool_call && item.custom_tool_call->id && !item.custom_tool_call->id->empty()) {
    return *item.custom_tool_call->id;
  }
  return std::string{};
}

}  // namespace

void ResponseStreamSnapshot::ingest(const ResponseStreamEvent& event) {
  events_.push_back(event);
  switch (event.type) {
    case ResponseStreamEvent::Type::Created:
      if (event.created) handle_created(*event.created);
      break;
    case ResponseStreamEvent::Type::Completed:
      if (event.completed) handle_completed(*event.completed);
      break;
    case ResponseStreamEvent::Type::OutputItemAdded:
      if (event.output_item_added) handle_output_item_added(*event.output_item_added);
      break;
    case ResponseStreamEvent::Type::OutputItemDone:
      if (event.output_item_done) handle_output_item_done(*event.output_item_done);
      break;
    case ResponseStreamEvent::Type::ContentPartAdded:
      if (event.content_part_added) handle_content_part_added(*event.content_part_added);
      break;
    case ResponseStreamEvent::Type::ContentPartDone:
      if (event.content_part_done) handle_content_part_done(*event.content_part_done);
      break;
    case ResponseStreamEvent::Type::OutputTextDelta:
      if (event.text_delta) handle_text_delta(*event.text_delta);
      break;
    case ResponseStreamEvent::Type::OutputTextDone:
      if (event.text_done) handle_text_done(*event.text_done);
      break;
    case ResponseStreamEvent::Type::ReasoningTextDelta:
      if (event.reasoning_text_delta) handle_reasoning_text_delta(*event.reasoning_text_delta);
      break;
    case ResponseStreamEvent::Type::ReasoningTextDone:
      if (event.reasoning_text_done) handle_reasoning_text_done(*event.reasoning_text_done);
      break;
    case ResponseStreamEvent::Type::FunctionCallArgumentsDelta:
      if (event.function_arguments_delta) handle_function_arguments_delta(*event.function_arguments_delta);
      break;
    case ResponseStreamEvent::Type::FunctionCallArgumentsDone:
      if (event.function_arguments_done) handle_function_arguments_done(*event.function_arguments_done);
      break;
    case ResponseStreamEvent::Type::Unknown:
      break;
  }
}

Response& ResponseStreamSnapshot::ensure_response() {
  if (!current_response_) {
    throw OpenAIError("response stream received event before response.created");
  }
  return *current_response_;
}

ResponseOutputItem* ResponseStreamSnapshot::find_output_item(const std::string& item_id, int output_index) {
  Response& response = ensure_response();
  if (!item_id.empty()) {
    auto it = item_index_by_id_.find(item_id);
    if (it != item_index_by_id_.end() && it->second < response.output.size()) {
      return &response.output[it->second];
    }
  }
  if (output_index >= 0) {
    const auto index = static_cast<std::size_t>(output_index);
    if (index < response.output.size()) {
      return &response.output[index];
    }
  }
  return nullptr;
}

void ResponseStreamSnapshot::rebuild_item_index() {
  item_index_by_id_.clear();
  if (!current_response_) return;
  for (std::size_t i = 0; i < current_response_->output.size(); ++i) {
    const auto id = extract_item_id(current_response_->output[i]);
    if (!id.empty()) {
      item_index_by_id_[id] = i;
    }
  }
}

void ResponseStreamSnapshot::refresh_message_segments(ResponseOutputMessage& message) {
  message.text_segments.clear();
  for (const auto& content : message.content) {
    if (content.type == ResponseOutputContent::Type::Text && content.text) {
      message.text_segments.push_back(*content.text);
    }
  }
}

void ResponseStreamSnapshot::apply_message_part(ResponseOutputMessage& message,
                                                const ResponseContentPartAddedEvent& part) {
  if (!part.content_part) {
    return;
  }
  const auto index = part.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(part.content_index);
  if (message.content.size() <= index) {
    message.content.resize(index + 1);
  }
  message.content[index] = *part.content_part;
  refresh_message_segments(message);
}

void ResponseStreamSnapshot::apply_reasoning_part(ResponseReasoningItemDetails& reasoning,
                                                  const ResponseContentPartAddedEvent& part) {
  const auto index = part.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(part.content_index);
  if (reasoning.content.size() <= index) {
    reasoning.content.resize(index + 1);
  }
  if (part.reasoning_part) {
    reasoning.content[index] = *part.reasoning_part;
  } else if (part.content_part && part.content_part->type == ResponseOutputContent::Type::Text &&
             part.content_part->text) {
    ResponseReasoningContent content;
    content.type = "reasoning_text";
    content.text = part.content_part->text->text;
    content.raw = part.content_part->raw;
    reasoning.content[index] = std::move(content);
  }
}

void ResponseStreamSnapshot::rebuild_messages() {
  if (!current_response_) return;
  current_response_->messages.clear();
  for (auto& item : current_response_->output) {
    if (item.type == ResponseOutputItem::Type::Message && item.message) {
      refresh_message_segments(*item.message);
      current_response_->messages.push_back(*item.message);
    }
  }
  current_response_->output_text = join_messages(current_response_->messages);
}

void ResponseStreamSnapshot::handle_created(const ResponseCreatedEvent& created) {
  current_response_ = created.response;
  completed_response_.reset();
  rebuild_item_index();
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_completed(const ResponseCompletedEvent& completed) {
  current_response_ = completed.response;
  completed_response_ = completed.response;
  rebuild_item_index();
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_output_item_added(const ResponseOutputItemAddedEvent& added) {
  Response& response = ensure_response();
  if (added.output_index < 0) {
    throw OpenAIError("response.output_item.added provided negative output_index");
  }
  const auto index = static_cast<std::size_t>(added.output_index);
  if (response.output.size() <= index) {
    response.output.resize(index + 1);
  }
  response.output[index] = added.item;
  rebuild_item_index();
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_output_item_done(const ResponseOutputItemDoneEvent& done) {
  Response& response = ensure_response();
  if (done.output_index < 0) {
    throw OpenAIError("response.output_item.done provided negative output_index");
  }
  const auto index = static_cast<std::size_t>(done.output_index);
  if (response.output.size() <= index) {
    response.output.resize(index + 1);
  }
  response.output[index] = done.item;
  rebuild_item_index();
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_content_part_added(const ResponseContentPartAddedEvent& added) {
  auto* item = find_output_item(added.item_id, added.output_index);
  if (!item) {
    return;
  }
  if (item->type == ResponseOutputItem::Type::Message && item->message) {
    apply_message_part(*item->message, added);
  } else if (item->type == ResponseOutputItem::Type::Reasoning && item->reasoning) {
    apply_reasoning_part(*item->reasoning, added);
  }
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_content_part_done(const ResponseContentPartDoneEvent& done) {
  ResponseContentPartAddedEvent converted;
  converted.content_index = done.content_index;
  converted.output_index = done.output_index;
  converted.item_id = done.item_id;
  converted.content_part = done.content_part;
  converted.reasoning_part = done.reasoning_part;
  handle_content_part_added(converted);
}

void ResponseStreamSnapshot::handle_text_delta(const ResponseTextDeltaEvent& delta) {
  auto* item = find_output_item(delta.item_id, delta.output_index);
  if (!item || item->type != ResponseOutputItem::Type::Message || !item->message) {
    return;
  }
  ResponseOutputMessage& message = *item->message;
  const auto index = delta.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(delta.content_index);
  if (message.content.size() <= index) {
    message.content.resize(index + 1);
  }
  auto& content = message.content[index];
  if (content.type != ResponseOutputContent::Type::Text || !content.text) {
    content.type = ResponseOutputContent::Type::Text;
    content.text = ResponseOutputTextSegment{};
  }
  content.text->text += delta.delta;
  content.text->logprobs.insert(content.text->logprobs.end(), delta.logprobs.begin(), delta.logprobs.end());
  refresh_message_segments(message);
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_text_done(const ResponseTextDoneEvent& done) {
  auto* item = find_output_item(done.item_id, done.output_index);
  if (!item || item->type != ResponseOutputItem::Type::Message || !item->message) {
    return;
  }
  ResponseOutputMessage& message = *item->message;
  const auto index = done.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(done.content_index);
  if (message.content.size() <= index) {
    message.content.resize(index + 1);
  }
  auto& content = message.content[index];
  if (content.type != ResponseOutputContent::Type::Text || !content.text) {
    content.type = ResponseOutputContent::Type::Text;
    content.text = ResponseOutputTextSegment{};
  }
  content.text->text = done.text;
  if (!done.logprobs.empty()) {
    content.text->logprobs = done.logprobs;
  }
  refresh_message_segments(message);
  rebuild_messages();
}

void ResponseStreamSnapshot::handle_reasoning_text_delta(const ResponseReasoningTextDeltaEvent& delta) {
  auto* item = find_output_item(delta.item_id, delta.output_index);
  if (!item || item->type != ResponseOutputItem::Type::Reasoning || !item->reasoning) {
    return;
  }
  auto& reasoning = *item->reasoning;
  const auto index = delta.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(delta.content_index);
  if (reasoning.content.size() <= index) {
    reasoning.content.resize(index + 1);
  }
  auto& content = reasoning.content[index];
  if (content.type.empty()) {
    content.type = "reasoning_text";
  }
  content.text += delta.delta;
}

void ResponseStreamSnapshot::handle_reasoning_text_done(const ResponseReasoningTextDoneEvent& done) {
  auto* item = find_output_item(done.item_id, done.output_index);
  if (!item || item->type != ResponseOutputItem::Type::Reasoning || !item->reasoning) {
    return;
  }
  auto& reasoning = *item->reasoning;
  const auto index = done.content_index < 0 ? std::size_t{0} : static_cast<std::size_t>(done.content_index);
  if (reasoning.content.size() <= index) {
    reasoning.content.resize(index + 1);
  }
  auto& content = reasoning.content[index];
  content.type = "reasoning_text";
  content.text = done.text;
}

void ResponseStreamSnapshot::handle_function_arguments_delta(const ResponseFunctionCallArgumentsDeltaEvent& delta) {
  auto* item = find_output_item(delta.item_id, delta.output_index);
  if (!item || item->type != ResponseOutputItem::Type::FunctionToolCall || !item->function_call) {
    return;
  }
  auto& call = *item->function_call;
  call.arguments += delta.delta;
}

void ResponseStreamSnapshot::handle_function_arguments_done(const ResponseFunctionCallArgumentsDoneEvent& done) {
  auto* item = find_output_item(done.item_id, done.output_index);
  if (!item || item->type != ResponseOutputItem::Type::FunctionToolCall || !item->function_call) {
    return;
  }
  auto& call = *item->function_call;
  call.arguments = done.arguments;
  call.name = done.name;
  if (!call.arguments.empty()) {
    try {
      call.parsed_arguments = json::parse(call.arguments);
    } catch (const json::exception&) {
      call.parsed_arguments = std::nullopt;
    }
  } else {
    call.parsed_arguments.reset();
  }
}

}  // namespace openai


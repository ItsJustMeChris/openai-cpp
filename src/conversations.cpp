#include "openai/conversations.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"

#include <functional>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

json serialize_response_input_content(const ResponseInputContent& content) {
  json payload = content.raw.is_object() ? content.raw : json::object();
  switch (content.type) {
    case ResponseInputContent::Type::Text:
      payload["type"] = "input_text";
      payload["text"] = content.text;
      break;
    case ResponseInputContent::Type::Image:
      payload["type"] = "input_image";
      if (!content.image_url.empty()) payload["image_url"] = content.image_url;
      if (!content.image_detail.empty()) payload["detail"] = content.image_detail;
      if (!content.file_id.empty()) payload["file_id"] = content.file_id;
      break;
    case ResponseInputContent::Type::File:
      payload["type"] = "input_file";
      if (!content.file_id.empty()) payload["file_id"] = content.file_id;
      if (!content.file_url.empty()) payload["file_url"] = content.file_url;
      if (!content.filename.empty()) payload["filename"] = content.filename;
      break;
    case ResponseInputContent::Type::Audio:
      payload["type"] = "input_audio";
      if (!content.audio_data.empty()) {
        json audio = json::object();
        audio["data"] = content.audio_data;
        audio["format"] = content.audio_format;
        payload["input_audio"] = std::move(audio);
      }
      break;
    case ResponseInputContent::Type::Raw:
      if (!content.raw.is_null()) payload = content.raw;
      break;
  }
  if (content.id) payload["id"] = *content.id;
  return payload;
}

json serialize_response_input_message(const ResponseInputMessage& message) {
  json payload = message.raw.is_object() ? message.raw : json::object();
  payload["type"] = "message";
  payload["role"] = message.role;
  if (message.id) payload["id"] = *message.id;
  if (message.status) payload["status"] = *message.status;
  if (!message.metadata.empty()) payload["metadata"] = message.metadata;
  json content = json::array();
  for (const auto& fragment : message.content) {
    content.push_back(serialize_response_input_content(fragment));
  }
  payload["content"] = std::move(content);
  return payload;
}

json serialize_response_input_item(const ResponseInputItem& item) {
  json payload = item.raw.is_object() ? item.raw : json::object();
  switch (item.type) {
    case ResponseInputItem::Type::Message:
      payload = serialize_response_input_message(item.message);
      break;
    case ResponseInputItem::Type::InputText:
      payload["type"] = "input_text";
      if (item.input_text) payload["text"] = item.input_text->text;
      break;
    case ResponseInputItem::Type::InputImage:
      payload["type"] = "input_image";
      if (item.input_image) {
        if (item.input_image->image_url) payload["image_url"] = *item.input_image->image_url;
        if (item.input_image->file_id) payload["file_id"] = *item.input_image->file_id;
        if (item.input_image->detail) payload["detail"] = *item.input_image->detail;
      }
      break;
    case ResponseInputItem::Type::InputFile:
      payload["type"] = "input_file";
      if (item.input_file) {
        if (item.input_file->file_data) payload["file_data"] = *item.input_file->file_data;
        if (item.input_file->file_id) payload["file_id"] = *item.input_file->file_id;
        if (item.input_file->file_url) payload["file_url"] = *item.input_file->file_url;
        if (item.input_file->filename) payload["filename"] = *item.input_file->filename;
      }
      break;
    case ResponseInputItem::Type::InputAudio:
      payload["type"] = "input_audio";
      if (item.input_audio) {
        json audio = json::object();
        audio["data"] = item.input_audio->data;
        audio["format"] = item.input_audio->format;
        payload["input_audio"] = std::move(audio);
      }
      break;
    case ResponseInputItem::Type::Raw:
      if (!item.raw.is_null()) payload = item.raw;
      break;
  }
  return payload;
}

json serialize_response_input_items(const std::vector<ResponseInputItem>& items) {
  json array = json::array();
  for (const auto& item : items) {
    array.push_back(serialize_response_input_item(item));
  }
  return array;
}

std::string join_include(const std::optional<std::vector<std::string>>& include) {
  if (!include || include->empty()) return {};
  std::ostringstream oss;
  for (std::size_t i = 0; i < include->size(); ++i) {
    if (i > 0) oss << ",";
    oss << (*include)[i];
  }
  return oss.str();
}

ComputerScreenshotContent parse_computer_screenshot(const json& payload) {
  ComputerScreenshotContent screenshot;
  if (payload.contains("file_id") && payload.at("file_id").is_string()) {
    screenshot.file_id = payload.at("file_id").get<std::string>();
  }
  if (payload.contains("image_url") && payload.at("image_url").is_string()) {
    screenshot.image_url = payload.at("image_url").get<std::string>();
  }
  screenshot.type = payload.value("type", "");
  return screenshot;
}

ConversationMessageContent parse_message_content(const json& payload) {
  ConversationMessageContent content;
  content.raw = payload;
  const std::string type = payload.value("type", "");
  if (type == "input_text" || type == "output_text") {
    content.kind = (type == "input_text") ? ConversationMessageContent::Kind::InputText
                                          : ConversationMessageContent::Kind::OutputText;
    content.text = payload.value("text", "");
  } else if (type == "text") {
    content.kind = ConversationMessageContent::Kind::Text;
    content.text = payload.value("text", "");
  } else if (type == "summary_text") {
    content.kind = ConversationMessageContent::Kind::SummaryText;
    content.text = payload.value("text", "");
  } else if (type == "reasoning_text") {
    content.kind = ConversationMessageContent::Kind::ReasoningText;
    content.text = payload.value("text", "");
  } else if (type == "output_refusal") {
    content.kind = ConversationMessageContent::Kind::OutputRefusal;
    content.text = payload.value("text", "");
  } else if (type == "input_image") {
    content.kind = ConversationMessageContent::Kind::InputImage;
    if (payload.contains("image_url") && payload.at("image_url").is_string()) {
      content.image_url = payload.at("image_url").get<std::string>();
    }
    if (payload.contains("file_id") && payload.at("file_id").is_string()) {
      content.file_id = payload.at("file_id").get<std::string>();
    }
  } else if (type == "computer_screenshot") {
    content.kind = ConversationMessageContent::Kind::ComputerScreenshot;
    content.computer_screenshot = parse_computer_screenshot(payload);
  } else if (type == "input_file") {
    content.kind = ConversationMessageContent::Kind::InputFile;
    if (payload.contains("file_id") && payload.at("file_id").is_string()) {
      content.file_id = payload.at("file_id").get<std::string>();
    }
  } else {
    content.kind = ConversationMessageContent::Kind::Unknown;
  }
  return content;
}

ConversationMessage parse_message(const json& payload) {
  ConversationMessage message;
  message.raw = payload;
  message.id = payload.value("id", "");
  message.role = payload.value("role", "");
  message.status = payload.value("status", "");
  message.type = payload.value("type", "");
  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& entry : payload.at("content")) {
      message.content.push_back(parse_message_content(entry));
    }
  }
  return message;
}

ConversationLocalShellCallAction parse_local_shell_action(const json& payload) {
  ConversationLocalShellCallAction action;
  action.type = payload.value("type", "");
  if (payload.contains("command") && payload.at("command").is_array()) {
    for (const auto& arg : payload.at("command")) {
      if (arg.is_string()) action.command.push_back(arg.get<std::string>());
    }
  }
  if (payload.contains("env") && payload.at("env").is_object()) {
    for (auto it = payload.at("env").begin(); it != payload.at("env").end(); ++it) {
      if (it.value().is_string()) action.env[it.key()] = it.value().get<std::string>();
    }
  }
  if (payload.contains("timeout_ms") && payload.at("timeout_ms").is_number_integer()) {
    action.timeout_ms = payload.at("timeout_ms").get<int>();
  }
  if (payload.contains("user") && payload.at("user").is_string()) {
    action.user = payload.at("user").get<std::string>();
  }
  if (payload.contains("working_directory") && payload.at("working_directory").is_string()) {
    action.working_directory = payload.at("working_directory").get<std::string>();
  }
  return action;
}

ConversationLocalShellCall parse_local_shell_call(const json& payload) {
  ConversationLocalShellCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.call_id = payload.value("call_id", "");
  call.status = payload.value("status", "");
  call.type = payload.value("type", "");
  if (payload.contains("action") && payload.at("action").is_object()) {
    call.action = parse_local_shell_action(payload.at("action"));
  }
  return call;
}

ConversationLocalShellCallOutput parse_local_shell_output(const json& payload) {
  ConversationLocalShellCallOutput output;
  output.raw = payload;
  output.id = payload.value("id", "");
  output.output = payload.value("output", "");
  if (payload.contains("status") && payload.at("status").is_string()) {
    output.status = payload.at("status").get<std::string>();
  }
  output.type = payload.value("type", "");
  return output;
}

ConversationMcpListTool parse_mcp_list_tool(const json& payload) {
  ConversationMcpListTool tool;
  tool.name = payload.value("name", "");
  if (payload.contains("input_schema")) {
    tool.input_schema = payload.at("input_schema");
  }
  if (payload.contains("annotations")) {
    tool.annotations = payload.at("annotations");
  }
  if (payload.contains("description") && payload.at("description").is_string()) {
    tool.description = payload.at("description").get<std::string>();
  }
  return tool;
}

ConversationMcpListTools parse_mcp_list_tools(const json& payload) {
  ConversationMcpListTools list;
  list.raw = payload;
  list.id = payload.value("id", "");
  list.server_label = payload.value("server_label", "");
  list.type = payload.value("type", "");
  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
      list.tools.push_back(parse_mcp_list_tool(tool_json));
    }
  }
  if (payload.contains("error") && payload.at("error").is_string()) {
    list.error = payload.at("error").get<std::string>();
  }
  return list;
}

ConversationMcpApprovalRequest parse_mcp_approval_request(const json& payload) {
  ConversationMcpApprovalRequest request;
  request.raw = payload;
  request.id = payload.value("id", "");
  request.arguments = payload.value("arguments", "");
  request.name = payload.value("name", "");
  request.server_label = payload.value("server_label", "");
  request.type = payload.value("type", "");
  return request;
}

ConversationMcpApprovalResponse parse_mcp_approval_response(const json& payload) {
  ConversationMcpApprovalResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.approval_request_id = payload.value("approval_request_id", "");
  response.approve = payload.value("approve", false);
  response.type = payload.value("type", "");
  if (payload.contains("reason") && payload.at("reason").is_string()) {
    response.reason = payload.at("reason").get<std::string>();
  }
  return response;
}

ConversationMcpCall parse_mcp_call(const json& payload) {
  ConversationMcpCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.arguments = payload.value("arguments", "");
  call.name = payload.value("name", "");
  call.server_label = payload.value("server_label", "");
  call.type = payload.value("type", "");
  if (payload.contains("approval_request_id") && payload.at("approval_request_id").is_string()) {
    call.approval_request_id = payload.at("approval_request_id").get<std::string>();
  }
  if (payload.contains("error") && payload.at("error").is_string()) {
    call.error = payload.at("error").get<std::string>();
  }
  if (payload.contains("output") && payload.at("output").is_string()) {
    call.output = payload.at("output").get<std::string>();
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    call.status = payload.at("status").get<std::string>();
  }
  return call;
}

ConversationImageGenerationCall parse_image_generation_call(const json& payload) {
  ConversationImageGenerationCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  if (payload.contains("result") && payload.at("result").is_string()) {
    call.result = payload.at("result").get<std::string>();
  }
  call.status = payload.value("status", "");
  call.type = payload.value("type", "");
  return call;
}

ResponseFileSearchToolCallResult parse_file_search_result(const json& payload) {
  ResponseFileSearchToolCallResult result;
  result.raw = payload;
  if (payload.contains("attributes")) result.attributes = payload.at("attributes");
  if (payload.contains("file_id") && payload.at("file_id").is_string()) {
    result.file_id = payload.at("file_id").get<std::string>();
  }
  if (payload.contains("filename") && payload.at("filename").is_string()) {
    result.filename = payload.at("filename").get<std::string>();
  }
  if (payload.contains("score") && payload.at("score").is_number()) {
    result.score = payload.at("score").get<double>();
  }
  if (payload.contains("text") && payload.at("text").is_string()) {
    result.text = payload.at("text").get<std::string>();
  }
  return result;
}

ResponseFileSearchToolCall parse_file_search_call(const json& payload) {
  ResponseFileSearchToolCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.status = payload.value("status", "");
  if (payload.contains("queries") && payload.at("queries").is_array()) {
    for (const auto& query_json : payload.at("queries")) {
      if (query_json.is_string()) call.queries.push_back(query_json.get<std::string>());
    }
  }
  if (payload.contains("results") && payload.at("results").is_array()) {
    for (const auto& result_json : payload.at("results")) {
      call.results.push_back(parse_file_search_result(result_json));
    }
  }
  return call;
}

ResponseFunctionToolCall parse_function_tool_call(const json& payload) {
  ResponseFunctionToolCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.call_id = payload.value("call_id", "");
  call.name = payload.value("name", "");
  call.arguments = payload.value("arguments", "");
  if (payload.contains("status") && payload.at("status").is_string()) {
    call.status = payload.at("status").get<std::string>();
  }
  if (!call.arguments.empty()) {
    try {
      call.parsed_arguments = json::parse(call.arguments);
    } catch (const json::exception&) {
      call.parsed_arguments = std::nullopt;
    }
  }
  return call;
}

ResponseFunctionToolCallOutput parse_function_tool_call_output(const json& payload) {
  ResponseFunctionToolCallOutput output;
  output.raw = payload;
  output.id = payload.value("id", "");
  output.call_id = payload.value("call_id", "");
  if (payload.contains("output")) {
    const auto& out = payload.at("output");
    if (out.is_string()) {
      output.output_text = out.get<std::string>();
      if (output.output_text && !output.output_text->empty()) {
        try {
          output.parsed_output_json = json::parse(*output.output_text);
        } catch (const json::exception&) {
          output.parsed_output_json = std::nullopt;
        }
      }
    } else {
      output.output_content = out;
      if (out.is_array()) {
        for (const auto& item : out) {
          ResponseInputContent content_piece;
          content_piece.raw = item;
          const std::string type = item.value("type", "");
          if (type == "input_text") {
            content_piece.type = ResponseInputContent::Type::Text;
            content_piece.text = item.value("text", "");
          } else if (type == "input_image") {
            content_piece.type = ResponseInputContent::Type::Image;
            content_piece.image_url = item.value("image_url", "");
            content_piece.image_detail = item.value("detail", "");
            content_piece.file_id = item.value("file_id", "");
          } else if (type == "input_file") {
            content_piece.type = ResponseInputContent::Type::File;
            content_piece.file_id = item.value("file_id", "");
            content_piece.file_url = item.value("file_url", "");
            content_piece.filename = item.value("filename", "");
          } else if (type == "input_audio") {
            content_piece.type = ResponseInputContent::Type::Audio;
            if (item.contains("input_audio") && item.at("input_audio").is_object()) {
              const auto& audio = item.at("input_audio");
              content_piece.audio_data = audio.value("data", "");
              content_piece.audio_format = audio.value("format", "");
            }
          } else {
            content_piece.type = ResponseInputContent::Type::Raw;
          }
          output.structured_output.push_back(std::move(content_piece));
        }
      }
    }
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    output.status = payload.at("status").get<std::string>();
  }
  return output;
}

ResponseFunctionWebSearch parse_function_web_search(const json& payload) {
  ResponseFunctionWebSearch search;
  search.raw = payload;
  search.id = payload.value("id", "");
  search.status = payload.value("status", "");
  auto parse_action = [](const json& action_json) {
    ResponseFunctionWebSearch::Action action;
    action.raw = action_json;
    const std::string type = action_json.value("type", "");
    if (type == "search") {
      action.type = ResponseFunctionWebSearch::Action::Type::Search;
      if (action_json.contains("query") && action_json.at("query").is_string()) {
        action.query = action_json.at("query").get<std::string>();
      }
      if (action_json.contains("sources") && action_json.at("sources").is_array()) {
        for (const auto& source_json : action_json.at("sources")) {
          ResponseFunctionWebSearch::Action::Source source;
          source.raw = source_json;
          source.url = source_json.value("url", "");
          action.sources.push_back(std::move(source));
        }
      }
    } else if (type == "open_page") {
      action.type = ResponseFunctionWebSearch::Action::Type::OpenPage;
      if (action_json.contains("url") && action_json.at("url").is_string()) {
        action.url = action_json.at("url").get<std::string>();
      }
    } else if (type == "find") {
      action.type = ResponseFunctionWebSearch::Action::Type::Find;
      if (action_json.contains("pattern") && action_json.at("pattern").is_string()) {
        action.pattern = action_json.at("pattern").get<std::string>();
      }
      if (action_json.contains("url") && action_json.at("url").is_string()) {
        action.url = action_json.at("url").get<std::string>();
      }
    } else {
      action.type = ResponseFunctionWebSearch::Action::Type::Unknown;
    }
    return action;
  };

  if (payload.contains("action") && payload.at("action").is_object()) {
    search.actions.push_back(parse_action(payload.at("action")));
  }
  if (payload.contains("actions")) {
    const auto& actions_json = payload.at("actions");
    if (actions_json.is_array()) {
      for (const auto& action_json : actions_json) {
        if (action_json.is_object()) {
          search.actions.push_back(parse_action(action_json));
        }
      }
    } else if (actions_json.is_object()) {
      search.actions.push_back(parse_action(actions_json));
    }
  }
  return search;
}

ResponseReasoningSummary parse_reasoning_summary(const json& payload) {
  ResponseReasoningSummary summary;
  summary.raw = payload;
  summary.text = payload.value("text", "");
  summary.type = payload.value("type", "");
  return summary;
}

ResponseReasoningContent parse_reasoning_content(const json& payload) {
  ResponseReasoningContent content;
  content.raw = payload;
  content.text = payload.value("text", "");
  content.type = payload.value("type", "");
  return content;
}

ResponseReasoningItemDetails parse_reasoning_item(const json& payload) {
  ResponseReasoningItemDetails reasoning;
  reasoning.raw = payload;
  reasoning.id = payload.value("id", "");
  if (payload.contains("summary") && payload.at("summary").is_array()) {
    for (const auto& summary_json : payload.at("summary")) {
      reasoning.summary.push_back(parse_reasoning_summary(summary_json));
    }
  }
  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& content_json : payload.at("content")) {
      reasoning.content.push_back(parse_reasoning_content(content_json));
    }
  }
  if (payload.contains("encrypted_content") && payload.at("encrypted_content").is_string()) {
    reasoning.encrypted_content = payload.at("encrypted_content").get<std::string>();
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    reasoning.status = payload.at("status").get<std::string>();
  }
  return reasoning;
}

ResponseCodeInterpreterToolCall parse_code_interpreter_tool_call(const json& payload) {
  ResponseCodeInterpreterToolCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  if (payload.contains("code")) {
    if (payload.at("code").is_string()) {
      call.code = payload.at("code").get<std::string>();
    } else if (payload.at("code").is_null()) {
      call.code = std::nullopt;
    }
  }
  call.container_id = payload.value("container_id", "");
  if (payload.contains("outputs") && payload.at("outputs").is_array()) {
    for (const auto& output_json : payload.at("outputs")) {
      const std::string output_type = output_json.value("type", "");
      if (output_type == "logs") {
        ResponseCodeInterpreterLogOutput log;
        log.raw = output_json;
        log.logs = output_json.value("logs", "");
        call.log_outputs.push_back(std::move(log));
      } else if (output_type == "image") {
        ResponseCodeInterpreterImageOutput image;
        image.raw = output_json;
        image.url = output_json.value("url", "");
        call.image_outputs.push_back(std::move(image));
      }
    }
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    call.status = payload.at("status").get<std::string>();
  }
  return call;
}

ResponseComputerToolCall parse_computer_tool_call(const json& payload) {
  ResponseComputerToolCall call;
  call.raw = payload;
  call.id = payload.value("id", "");
  call.call_id = payload.value("call_id", "");
  call.status = payload.value("status", "");

  auto parse_action = [](const json& action_json) {
    ResponseComputerToolCall::Action action;
    action.raw = action_json;
    const std::string type = action_json.value("type", "");
    if (type == "click") {
      action.type = ResponseComputerToolCall::Action::Type::Click;
      if (action_json.contains("button") && action_json.at("button").is_string()) {
        action.button = action_json.at("button").get<std::string>();
      }
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) action.x = action_json.at("x").get<int>();
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) action.y = action_json.at("y").get<int>();
    } else if (type == "double_click") {
      action.type = ResponseComputerToolCall::Action::Type::DoubleClick;
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) action.x = action_json.at("x").get<int>();
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) action.y = action_json.at("y").get<int>();
    } else if (type == "drag") {
      action.type = ResponseComputerToolCall::Action::Type::Drag;
      if (action_json.contains("path") && action_json.at("path").is_array()) {
        for (const auto& point : action_json.at("path")) {
          if (point.contains("x") && point.contains("y") && point.at("x").is_number_integer() &&
              point.at("y").is_number_integer()) {
            action.path.push_back({point.at("x").get<int>(), point.at("y").get<int>()});
          }
        }
      }
    } else if (type == "keypress") {
      action.type = ResponseComputerToolCall::Action::Type::Keypress;
      if (action_json.contains("keys") && action_json.at("keys").is_array()) {
        for (const auto& key : action_json.at("keys")) {
          if (key.is_string()) action.keys.push_back(key.get<std::string>());
        }
      }
    } else if (type == "move") {
      action.type = ResponseComputerToolCall::Action::Type::Move;
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) action.x = action_json.at("x").get<int>();
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) action.y = action_json.at("y").get<int>();
    } else if (type == "screenshot") {
      action.type = ResponseComputerToolCall::Action::Type::Screenshot;
    } else if (type == "scroll") {
      action.type = ResponseComputerToolCall::Action::Type::Scroll;
      if (action_json.contains("scroll_x") && action_json.at("scroll_x").is_number_integer()) {
        action.scroll_x = action_json.at("scroll_x").get<int>();
      }
      if (action_json.contains("scroll_y") && action_json.at("scroll_y").is_number_integer()) {
        action.scroll_y = action_json.at("scroll_y").get<int>();
      }
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) action.x = action_json.at("x").get<int>();
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) action.y = action_json.at("y").get<int>();
    } else if (type == "type") {
      action.type = ResponseComputerToolCall::Action::Type::Type;
      if (action_json.contains("text") && action_json.at("text").is_string()) {
        action.text = action_json.at("text").get<std::string>();
      }
    } else if (type == "wait") {
      action.type = ResponseComputerToolCall::Action::Type::Wait;
    } else {
      action.type = ResponseComputerToolCall::Action::Type::Unknown;
    }
    return action;
  };

  if (payload.contains("action") && payload.at("action").is_object()) {
    call.action = parse_action(payload.at("action"));
  }
  if (payload.contains("pending_safety_checks") && payload.at("pending_safety_checks").is_array()) {
    for (const auto& entry : payload.at("pending_safety_checks")) {
      ResponseComputerToolCall::PendingSafetyCheck check;
      check.raw = entry;
      check.id = entry.value("id", "");
      check.code = entry.value("code", "");
      check.message = entry.value("message", "");
      call.pending_safety_checks.push_back(std::move(check));
    }
  }
  return call;
}

ResponseComputerToolCallOutputScreenshot parse_computer_tool_call_output_screenshot(const json& payload) {
  ResponseComputerToolCallOutputScreenshot screenshot;
  screenshot.raw = payload;
  if (payload.contains("file_id") && payload.at("file_id").is_string()) {
    screenshot.file_id = payload.at("file_id").get<std::string>();
  }
  if (payload.contains("image_url") && payload.at("image_url").is_string()) {
    screenshot.image_url = payload.at("image_url").get<std::string>();
  }
  return screenshot;
}

ResponseComputerToolCallOutput parse_computer_tool_call_output(const json& payload) {
  ResponseComputerToolCallOutput output;
  output.raw = payload;
  output.id = payload.value("id", "");
  output.call_id = payload.value("call_id", "");
  if (payload.contains("output") && payload.at("output").is_object()) {
    output.screenshot = parse_computer_tool_call_output_screenshot(payload.at("output"));
  }
  if (payload.contains("acknowledged_safety_checks") && payload.at("acknowledged_safety_checks").is_array()) {
    for (const auto& entry : payload.at("acknowledged_safety_checks")) {
      ResponseComputerToolCall::PendingSafetyCheck check;
      check.raw = entry;
      check.id = entry.value("id", "");
      check.code = entry.value("code", "");
      check.message = entry.value("message", "");
      output.acknowledged_safety_checks.push_back(std::move(check));
    }
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    output.status = payload.at("status").get<std::string>();
  }
  return output;
}

ResponseCustomToolCall parse_custom_tool_call(const json& payload) {
  ResponseCustomToolCall call;
  call.raw = payload;
  call.call_id = payload.value("call_id", "");
  call.input = payload.value("input", "");
  call.name = payload.value("name", "");
  if (payload.contains("id") && payload.at("id").is_string()) {
    call.id = payload.at("id").get<std::string>();
  }
  return call;
}

ConversationItem parse_conversation_item(const json& payload) {
  ConversationItem item;
  item.raw = payload;
  item.type = payload.value("type", "");

  if (item.type == "message") {
    item.kind = ConversationItem::Kind::Message;
    item.message = parse_message(payload);
  } else if (item.type == "function_call") {
    item.kind = ConversationItem::Kind::FunctionToolCall;
    item.function_tool_call = parse_function_tool_call(payload);
  } else if (item.type == "function_call_output") {
    item.kind = ConversationItem::Kind::FunctionToolCallOutput;
    item.function_tool_call_output = parse_function_tool_call_output(payload);
  } else if (item.type == "file_search_call") {
    item.kind = ConversationItem::Kind::FileSearchToolCall;
    item.file_search_tool_call = parse_file_search_call(payload);
  } else if (item.type == "web_search_call") {
    item.kind = ConversationItem::Kind::FunctionWebSearch;
    item.function_web_search = parse_function_web_search(payload);
  } else if (item.type == "image_generation_call") {
    item.kind = ConversationItem::Kind::ImageGenerationCall;
    item.image_generation_call = parse_image_generation_call(payload);
  } else if (item.type == "computer_call") {
    item.kind = ConversationItem::Kind::ComputerToolCall;
    item.computer_tool_call = parse_computer_tool_call(payload);
  } else if (item.type == "computer_call_output") {
    item.kind = ConversationItem::Kind::ComputerToolCallOutput;
    item.computer_tool_call_output = parse_computer_tool_call_output(payload);
  } else if (item.type == "reasoning") {
    item.kind = ConversationItem::Kind::Reasoning;
    item.reasoning = parse_reasoning_item(payload);
  } else if (item.type == "code_interpreter_call") {
    item.kind = ConversationItem::Kind::CodeInterpreterToolCall;
    item.code_interpreter_tool_call = parse_code_interpreter_tool_call(payload);
  } else if (item.type == "local_shell_call") {
    item.kind = ConversationItem::Kind::LocalShellCall;
    item.local_shell_call = parse_local_shell_call(payload);
  } else if (item.type == "local_shell_call_output") {
    item.kind = ConversationItem::Kind::LocalShellCallOutput;
    item.local_shell_output = parse_local_shell_output(payload);
  } else if (item.type == "mcp_list_tools") {
    item.kind = ConversationItem::Kind::McpListTools;
    item.mcp_list_tools = parse_mcp_list_tools(payload);
  } else if (item.type == "mcp_approval_request") {
    item.kind = ConversationItem::Kind::McpApprovalRequest;
    item.mcp_approval_request = parse_mcp_approval_request(payload);
  } else if (item.type == "mcp_approval_response") {
    item.kind = ConversationItem::Kind::McpApprovalResponse;
    item.mcp_approval_response = parse_mcp_approval_response(payload);
  } else if (item.type == "mcp_call") {
    item.kind = ConversationItem::Kind::McpCall;
    item.mcp_call = parse_mcp_call(payload);
  } else if (item.type == "custom_tool_call") {
    item.kind = ConversationItem::Kind::CustomToolCall;
    item.custom_tool_call = parse_custom_tool_call(payload);
  } else {
    item.kind = ConversationItem::Kind::Unknown;
  }

  return item;
}

ConversationItemList parse_item_list(const json& payload) {
  ConversationItemList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      list.data.push_back(parse_conversation_item(entry));
    }
  }
  if (payload.contains("first_id") && payload.at("first_id").is_string()) {
    list.first_id = payload.at("first_id").get<std::string>();
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  list.object = payload.value("object", "");
  return list;
}

ConversationItemsPage parse_items_page(const json& payload) {
  ConversationItemsPage page;
  page.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& entry : payload.at("data")) {
      page.data.push_back(parse_conversation_item(entry));
    }
  }
  if (payload.contains("first_id") && payload.at("first_id").is_string()) {
    page.first_id = payload.at("first_id").get<std::string>();
  }
  page.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    page.last_id = payload.at("last_id").get<std::string>();
  }
  if (payload.contains("last_id") && payload.at("last_id").is_null()) {
    page.last_id.reset();
  }
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    page.next_cursor = payload.at("next_cursor").get<std::string>();
  }
  if (payload.contains("object") && payload.at("object").is_string()) {
    page.object = payload.at("object").get<std::string>();
  }
  if (!page.next_cursor && page.last_id) {
    page.next_cursor = page.last_id;
  }
  return page;
}

Conversation parse_conversation(const json& payload) {
  Conversation convo;
  convo.raw = payload;
  convo.id = payload.value("id", "");
  convo.created_at = payload.value("created_at", 0);
  if (payload.contains("metadata")) {
    convo.metadata = payload.at("metadata");
  }
  convo.object = payload.value("object", "");
  return convo;
}

ConversationDeleted parse_conversation_deleted(const json& payload) {
  ConversationDeleted deleted;
  deleted.raw = payload;
  deleted.id = payload.value("id", "");
  deleted.deleted = payload.value("deleted", false);
  deleted.object = payload.value("object", "");
  return deleted;
}

json conversation_create_to_json(const ConversationCreateParams& params) {
  json body = json::object();
  if (params.items) {
    body["items"] = serialize_response_input_items(*params.items);
  }
  if (params.metadata) body["metadata"] = *params.metadata;
  return body;
}

json conversation_update_to_json(const ConversationUpdateParams& params) {
  json body = json::object();
  if (params.metadata) body["metadata"] = *params.metadata;
  return body;
}

}  // namespace

ConversationsResource::ConversationsResource(OpenAIClient& client)
    : client_(client), items_(std::make_unique<ConversationItemsResource>(client)) {}

Conversation ConversationsResource::create(const ConversationCreateParams& params) const {
  return create(params, RequestOptions{});
}

Conversation ConversationsResource::create(const ConversationCreateParams& params,
                                           const RequestOptions& options) const {
  auto body = conversation_create_to_json(params).dump();
  auto response = client_.perform_request("POST", "/conversations", body, options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation create response: ") + ex.what());
  }
}

Conversation ConversationsResource::create() const {
  return create(ConversationCreateParams{});
}

Conversation ConversationsResource::retrieve(const std::string& conversation_id) const {
  return retrieve(conversation_id, RequestOptions{});
}

Conversation ConversationsResource::retrieve(const std::string& conversation_id,
                                             const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation retrieve response: ") + ex.what());
  }
}

Conversation ConversationsResource::update(const std::string& conversation_id,
                                           const ConversationUpdateParams& params) const {
  return update(conversation_id, params, RequestOptions{});
}

Conversation ConversationsResource::update(const std::string& conversation_id,
                                           const ConversationUpdateParams& params,
                                           const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto body = conversation_update_to_json(params).dump();
  auto response = client_.perform_request("POST", path, body, options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation update response: ") + ex.what());
  }
}

ConversationDeleted ConversationsResource::remove(const std::string& conversation_id) const {
  return remove(conversation_id, RequestOptions{});
}

ConversationDeleted ConversationsResource::remove(const std::string& conversation_id,
                                                  const RequestOptions& options) const {
  auto path = std::string("/conversations/") + conversation_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_conversation_deleted(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation delete response: ") + ex.what());
  }
}

ConversationItemsResource& ConversationsResource::items() { return *items_; }

const ConversationItemsResource& ConversationsResource::items() const { return *items_; }

ConversationItemList ConversationItemsResource::create(const std::string& conversation_id,
                                                       const ItemCreateParams& params) const {
  return create(conversation_id, params, RequestOptions{});
}

ConversationItemList ConversationItemsResource::create(const std::string& conversation_id,
                                                       const ItemCreateParams& params,
                                                       const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.include) request_options.query_params["include"] = join_include(params.include);

  json body = json::object();
  body["items"] = serialize_response_input_items(params.items);

  auto path = std::string("/conversations/") + conversation_id + "/items";
  auto response = client_.perform_request("POST", path, body.dump(), request_options);
  try {
    return parse_item_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation items create response: ") + ex.what());
  }
}

ConversationItem ConversationItemsResource::retrieve(const std::string& conversation_id,
                                                     const std::string& item_id,
                                                     const ItemRetrieveParams& params) const {
  return retrieve(conversation_id, item_id, params, RequestOptions{});
}

ConversationItem ConversationItemsResource::retrieve(const std::string& conversation_id,
                                                     const std::string& item_id,
                                                     const ItemRetrieveParams& params,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.query_params["include"] = join_include(params.include);

  auto path = std::string("/conversations/") + params.conversation_id + "/items/" + item_id;
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_conversation_item(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation item retrieve response: ") + ex.what());
  }
}

ConversationItemsPage ConversationItemsResource::list(const std::string& conversation_id,
                                                      const ItemListParams& params) const {
  return list(conversation_id, params, RequestOptions{});
}

ConversationItemsPage ConversationItemsResource::list(const std::string& conversation_id,
                                                      const ItemListParams& params,
                                                      const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.include) request_options.query_params["include"] = join_include(params.include);

  auto path = std::string("/conversations/") + conversation_id + "/items";
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_items_page(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation items list response: ") + ex.what());
  }
}

ConversationItemsPage ConversationItemsResource::list(const std::string& conversation_id) const {
  return list(conversation_id, ItemListParams{});
}

CursorPage<ConversationItem> ConversationItemsResource::list_page(const std::string& conversation_id,
                                                                  const ItemListParams& params) const {
  return list_page(conversation_id, params, RequestOptions{});
}

CursorPage<ConversationItem> ConversationItemsResource::list_page(const std::string& conversation_id,
                                                                  const ItemListParams& params,
                                                                  const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.include) request_options.query_params["include"] = join_include(params.include);

  auto fetch_impl = std::make_shared<std::function<CursorPage<ConversationItem>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<ConversationItem> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    ConversationItemsPage page;
    try {
      page = parse_items_page(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse conversation items list response: ") + ex.what());
    }

    return CursorPage<ConversationItem>(std::move(page.data),
                                        page.has_more,
                                        page.next_cursor,
                                        request_options,
                                        *fetch_impl,
                                        "after",
                                        std::move(page.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = std::string("/conversations/") + conversation_id + "/items";
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

Conversation ConversationItemsResource::remove(const std::string& conversation_id, const std::string& item_id) const {
  return remove(conversation_id, item_id, ItemDeleteParams{conversation_id}, RequestOptions{});
}

Conversation ConversationItemsResource::remove(const std::string& conversation_id,
                                               const std::string& item_id,
                                               const ItemDeleteParams& params,
                                               const RequestOptions& options) const {
  auto path = std::string("/conversations/") + params.conversation_id + "/items/" + item_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_conversation(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse conversation item delete response: ") + ex.what());
  }
}

}  // namespace openai

#include "openai/messages.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

std::string build_thread_messages_path(const std::string& thread_id) {
  return "/threads/" + thread_id + "/messages";
}

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

json attachments_to_json(const std::vector<MessageAttachment>& attachments) {
  json array = json::array();
  for (const auto& attachment : attachments) {
    json item;
    if (!attachment.file_id.empty()) item["file_id"] = attachment.file_id;
    if (!attachment.tools.empty()) {
      json tools = json::array();
      for (const auto& tool : attachment.tools) {
        switch (tool.type) {
          case ThreadMessageAttachmentTool::Type::CodeInterpreter:
            tools.push_back(json::object({{"type", "code_interpreter"}}));
            break;
          case ThreadMessageAttachmentTool::Type::FileSearch:
            tools.push_back(json::object({{"type", "file_search"}}));
            break;
        }
      }
      item["tools"] = std::move(tools);
    }
    array.push_back(std::move(item));
  }
  return array;
}

json content_part_to_json(const ThreadMessageContentPart& part) {
  json value;
  switch (part.type) {
    case ThreadMessageContentPart::Type::Text:
      value["type"] = "text";
      value["text"] = part.text;
      break;
    case ThreadMessageContentPart::Type::ImageFile:
      value["type"] = "image_file";
      if (part.image_file) {
        json image;
        image["file_id"] = part.image_file->file_id;
        if (part.image_file->detail) image["detail"] = *part.image_file->detail;
        value["image_file"] = std::move(image);
      }
      break;
    case ThreadMessageContentPart::Type::ImageURL:
      value["type"] = "image_url";
      if (part.image_url) {
        json image;
        image["url"] = part.image_url->url;
        if (part.image_url->detail) image["detail"] = *part.image_url->detail;
        value["image_url"] = std::move(image);
      }
      break;
    case ThreadMessageContentPart::Type::Raw:
      value = part.raw;
      break;
  }
  if (part.type != ThreadMessageContentPart::Type::Raw) {
    for (auto it = part.raw.begin(); it != part.raw.end(); ++it) {
      value[it.key()] = it.value();
    }
  }
  return value;
}

json message_create_content_to_json(const std::variant<std::string, std::vector<ThreadMessageContentPart>>& content) {
  if (std::holds_alternative<std::string>(content)) {
    return json(std::get<std::string>(content));
  }
  json array = json::array();
  for (const auto& part : std::get<std::vector<ThreadMessageContentPart>>(content)) {
    array.push_back(content_part_to_json(part));
  }
  return array;
}

json metadata_to_json(const std::map<std::string, std::string>& metadata) {
  json value = json::object();
  for (const auto& entry : metadata) {
    value[entry.first] = entry.second;
  }
  return value;
}

json create_request_to_json(const MessageCreateRequest& request) {
  json body;
  body["role"] = request.role;
  body["content"] = message_create_content_to_json(request.content);
  if (!request.attachments.empty()) body["attachments"] = attachments_to_json(request.attachments);
  if (!request.metadata.empty()) body["metadata"] = metadata_to_json(request.metadata);
  return body;
}

json update_request_to_json(const MessageUpdateRequest& request) {
  json body;
  if (request.metadata) body["metadata"] = metadata_to_json(*request.metadata);
  return body;
}

MessageTextAnnotation parse_annotation(const json& payload) {
  MessageTextAnnotation annotation;
  annotation.raw = payload;
  const std::string type = payload.value("type", "");
  if (type == "file_citation") {
    annotation.type = MessageTextAnnotation::Type::FileCitation;
  } else if (type == "file_path") {
    annotation.type = MessageTextAnnotation::Type::FilePath;
  } else {
    annotation.type = MessageTextAnnotation::Type::Raw;
  }
  annotation.text = payload.value("text", "");
  annotation.start_index = payload.value("start_index", 0);
  annotation.end_index = payload.value("end_index", 0);
  if (payload.contains("file_citation") && payload["file_citation"].is_object()) {
    MessageTextAnnotation::FileCitation file_citation;
    if (payload.at("file_citation").contains("file_id") && payload.at("file_citation").at("file_id").is_string()) {
      file_citation.file_id = payload.at("file_citation").at("file_id").get<std::string>();
      annotation.file_id = file_citation.file_id;
    }
    if (payload.at("file_citation").contains("quote") && payload.at("file_citation").at("quote").is_string()) {
      file_citation.quote = payload.at("file_citation").at("quote").get<std::string>();
      annotation.quote = file_citation.quote;
    }
    annotation.file_citation = std::move(file_citation);
  }
  if (payload.contains("file_path") && payload["file_path"].is_object()) {
    MessageTextAnnotation::FilePath file_path;
    if (payload.at("file_path").contains("file_id") && payload.at("file_path").at("file_id").is_string()) {
      file_path.file_id = payload.at("file_path").at("file_id").get<std::string>();
      annotation.file_id = file_path.file_id;
    }
    annotation.file_path = std::move(file_path);
  }
  return annotation;
}

MessageTextContent parse_text_content(const json& payload) {
  MessageTextContent content;
  content.value = payload.value("value", "");
  if (payload.contains("annotations") && payload["annotations"].is_array()) {
    for (const auto& item : payload.at("annotations")) {
      content.annotations.push_back(parse_annotation(item));
    }
  }
  return content;
}

MessageContentPart parse_message_content(const json& payload) {
  MessageContentPart part;
  part.raw = payload;
  const std::string type = payload.value("type", "");
  if (type == "text") {
    part.type = MessageContentPart::Type::Text;
    if (payload.contains("text")) part.text = parse_text_content(payload.at("text"));
  } else if (type == "image_file") {
    part.type = MessageContentPart::Type::ImageFile;
    if (payload.contains("image_file") && payload["image_file"].is_object()) {
      MessageContentPart::ImageFileData data;
      data.file_id = payload.at("image_file").value("file_id", "");
      if (payload.at("image_file").contains("detail") && payload.at("image_file").at("detail").is_string()) {
        data.detail = payload.at("image_file").at("detail").get<std::string>();
      }
      part.image_file = data;
    }
  } else if (type == "image_url") {
    part.type = MessageContentPart::Type::ImageURL;
    if (payload.contains("image_url") && payload["image_url"].is_object()) {
      MessageContentPart::ImageURLData data;
      data.url = payload.at("image_url").value("url", "");
      if (payload.at("image_url").contains("detail") && payload.at("image_url").at("detail").is_string()) {
        data.detail = payload.at("image_url").at("detail").get<std::string>();
      }
      part.image_url = data;
    }
  } else if (type == "refusal") {
    part.type = MessageContentPart::Type::Refusal;
    part.refusal = payload.value("refusal", "");
  } else {
    part.type = MessageContentPart::Type::Raw;
  }
  return part;
}

MessageAttachment parse_attachment(const json& payload) {
  MessageAttachment attachment;
  attachment.file_id = payload.value("file_id", "");
  if (payload.contains("tools") && payload["tools"].is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
      ThreadMessageAttachmentTool tool;
      const std::string type = tool_json.value("type", "code_interpreter");
      if (type == "file_search") {
        tool.type = ThreadMessageAttachmentTool::Type::FileSearch;
      } else {
        tool.type = ThreadMessageAttachmentTool::Type::CodeInterpreter;
      }
      attachment.tools.push_back(tool);
    }
  }
  return attachment;
}

MessageContentDeltaPart parse_message_content_delta(const json& payload) {
  MessageContentDeltaPart part;
  part.raw = payload;
  part.index = payload.value("index", 0);
  const std::string type = payload.value("type", "");
  if (type == "text") {
    part.type = MessageContentDeltaPart::Type::Text;
    if (payload.contains("text") && payload["text"].is_object()) {
      part.text = parse_text_content(payload.at("text"));
    }
  } else if (type == "image_file") {
    part.type = MessageContentDeltaPart::Type::ImageFile;
    if (payload.contains("image_file") && payload["image_file"].is_object()) {
      MessageContentPart::ImageFileData data;
      data.file_id = payload.at("image_file").value("file_id", "");
      if (payload.at("image_file").contains("detail") && payload.at("image_file").at("detail").is_string()) {
        data.detail = payload.at("image_file").at("detail").get<std::string>();
      }
      part.image_file = data;
    }
  } else if (type == "image_url") {
    part.type = MessageContentDeltaPart::Type::ImageURL;
    if (payload.contains("image_url") && payload["image_url"].is_object()) {
      MessageContentPart::ImageURLData data;
      data.url = payload.at("image_url").value("url", "");
      if (payload.at("image_url").contains("detail") && payload.at("image_url").at("detail").is_string()) {
        data.detail = payload.at("image_url").at("detail").get<std::string>();
      }
      part.image_url = data;
    }
  } else if (type == "refusal") {
    part.type = MessageContentDeltaPart::Type::Refusal;
    if (payload.contains("refusal") && payload["refusal"].is_string()) {
      part.refusal = payload["refusal"].get<std::string>();
    }
  } else {
    part.type = MessageContentDeltaPart::Type::Raw;
  }
  return part;
}

ThreadMessage parse_thread_message_impl(const json& payload) {
  ThreadMessage message;
  message.raw = payload;
  message.id = payload.value("id", "");
  if (payload.contains("assistant_id") && !payload["assistant_id"].is_null()) {
    message.assistant_id = payload["assistant_id"].get<std::string>();
  }
  if (payload.contains("attachments") && payload["attachments"].is_array()) {
    for (const auto& item : payload.at("attachments")) {
      message.attachments.push_back(parse_attachment(item));
    }
  }
  if (payload.contains("completed_at") && payload["completed_at"].is_number_integer()) {
    message.completed_at = payload["completed_at"].get<int>();
  }
  if (payload.contains("content") && payload["content"].is_array()) {
    for (const auto& item : payload.at("content")) {
      message.content.push_back(parse_message_content(item));
    }
  }
  message.created_at = payload.value("created_at", 0);
  if (payload.contains("incomplete_at") && payload["incomplete_at"].is_number_integer()) {
    message.incomplete_at = payload["incomplete_at"].get<int>();
  }
  if (payload.contains("incomplete_details") && payload["incomplete_details"].is_object()) {
    ThreadMessageIncompleteDetails details;
    details.raw = payload.at("incomplete_details");
    if (details.raw.contains("reason") && details.raw.at("reason").is_string()) {
      details.reason = details.raw.at("reason").get<std::string>();
      message.incomplete_reason = details.reason;
    }
    message.incomplete_details = std::move(details);
  }
  if (!message.incomplete_reason && payload.contains("incomplete_reason") && payload["incomplete_reason"].is_string()) {
    message.incomplete_reason = payload["incomplete_reason"].get<std::string>();
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) message.metadata[it.key()] = it.value().get<std::string>();
    }
  }
  message.object = payload.value("object", "");
  message.role = payload.value("role", "");
  if (payload.contains("run_id") && !payload["run_id"].is_null()) {
    message.run_id = payload["run_id"].get<std::string>();
  }
  message.status = payload.value("status", "");
  message.thread_id = payload.value("thread_id", "");
  return message;
}

MessageList parse_list(const json& payload) {
  MessageList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("first_id") && payload["first_id"].is_string()) list.first_id = payload["first_id"].get<std::string>();
  if (payload.contains("last_id") && payload["last_id"].is_string()) list.last_id = payload["last_id"].get<std::string>();
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_thread_message_impl(item));
    }
  }
  return list;
}

MessageDeleteResponse parse_delete_response(const json& payload) {
  MessageDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

ThreadMessageDeltaEvent parse_thread_message_delta_impl(const nlohmann::json& payload) {
  ThreadMessageDeltaEvent event;
  event.raw = payload;
  event.id = payload.value("id", "");
  event.object = payload.value("object", "");
  if (payload.contains("delta") && payload["delta"].is_object()) {
    const auto& delta = payload.at("delta");
    ThreadMessageDelta message_delta;
    message_delta.raw = delta;
    if (delta.contains("role") && delta["role"].is_string()) {
      message_delta.role = delta["role"].get<std::string>();
    }
    if (delta.contains("content") && delta["content"].is_array()) {
      for (const auto& item : delta.at("content")) {
        message_delta.content.push_back(parse_message_content_delta(item));
      }
    }
    event.delta = message_delta;
  }
  return event;
}

}  // namespace

ThreadMessage parse_thread_message_json(const nlohmann::json& payload) {
  return parse_thread_message_impl(payload);
}

ThreadMessageDeltaEvent parse_thread_message_delta_json(const nlohmann::json& payload) {
  return parse_thread_message_delta_impl(payload);
}

ThreadMessage ThreadMessagesResource::create(const std::string& thread_id, const MessageCreateRequest& request) const {
  return create(thread_id, request, RequestOptions{});
}

ThreadMessage ThreadMessagesResource::create(const std::string& thread_id,
                                              const MessageCreateRequest& request,
                                              const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = create_request_to_json(request);
  auto response = client_.perform_request("POST", build_thread_messages_path(thread_id), body.dump(), request_options);
  try {
    return parse_thread_message_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread message: ") + ex.what());
  }
}

ThreadMessage ThreadMessagesResource::retrieve(const std::string& thread_id, const std::string& message_id) const {
  return retrieve(thread_id, message_id, RequestOptions{});
}

ThreadMessage ThreadMessagesResource::retrieve(const std::string& thread_id,
                                               const std::string& message_id,
                                               const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", build_thread_messages_path(thread_id) + "/" + message_id, "",
                                          request_options);
  try {
    return parse_thread_message_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread message: ") + ex.what());
  }
}

ThreadMessage ThreadMessagesResource::update(const std::string& thread_id,
                                             const std::string& message_id,
                                             const MessageUpdateRequest& request) const {
  return update(thread_id, message_id, request, RequestOptions{});
}

ThreadMessage ThreadMessagesResource::update(const std::string& thread_id,
                                             const std::string& message_id,
                                             const MessageUpdateRequest& request,
                                             const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = update_request_to_json(request);
  auto response = client_.perform_request("POST", build_thread_messages_path(thread_id) + "/" + message_id, body.dump(),
                                          request_options);
  try {
    return parse_thread_message_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread message: ") + ex.what());
  }
}

MessageDeleteResponse ThreadMessagesResource::remove(const std::string& thread_id, const std::string& message_id) const {
  return remove(thread_id, message_id, RequestOptions{});
}

MessageDeleteResponse ThreadMessagesResource::remove(const std::string& thread_id,
                                                     const std::string& message_id,
                                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("DELETE", build_thread_messages_path(thread_id) + "/" + message_id, "",
                                          request_options);
  try {
    return parse_delete_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread message delete response: ") + ex.what());
  }
}

MessageList ThreadMessagesResource::list(const std::string& thread_id) const {
  return list(thread_id, MessageListParams{}, RequestOptions{});
}

MessageList ThreadMessagesResource::list(const std::string& thread_id, const MessageListParams& params) const {
  return list(thread_id, params, RequestOptions{});
}

MessageList ThreadMessagesResource::list(const std::string& thread_id, const RequestOptions& options) const {
  return list(thread_id, MessageListParams{}, options);
}

MessageList ThreadMessagesResource::list(const std::string& thread_id,
                                         const MessageListParams& params,
                                         const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.before) request_options.query_params["before"] = *params.before;
  if (params.run_id) request_options.query_params["run_id"] = *params.run_id;
  auto response = client_.perform_request("GET", build_thread_messages_path(thread_id), "", request_options);
  try {
    return parse_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread message list: ") + ex.what());
  }
}

}  // namespace openai

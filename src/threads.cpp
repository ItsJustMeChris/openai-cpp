#include "openai/threads.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kThreadsPath = "/threads";
constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

json attachments_to_json(const std::vector<ThreadMessageAttachment>& attachments) {
  if (attachments.empty()) {
    return json::array();
  }
  json array = json::array();
  for (const auto& attachment : attachments) {
    json item = json::object();
    if (!attachment.file_id.empty()) {
      item["file_id"] = attachment.file_id;
    }
    if (!attachment.tools.empty()) {
      json tools = json::array();
      for (const auto& tool : attachment.tools) {
        if (tool.type == ThreadMessageAttachmentTool::Type::CodeInterpreter) {
          tools.push_back(json::object({{"type", "code_interpreter"}}));
        } else {
          tools.push_back(json::object({{"type", "file_search"}}));
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

json content_to_json(const std::variant<std::monostate, std::string, std::vector<ThreadMessageContentPart>>& content) {
  if (std::holds_alternative<std::string>(content)) {
    return json(std::get<std::string>(content));
  }
  if (std::holds_alternative<std::vector<ThreadMessageContentPart>>(content)) {
    json array = json::array();
    for (const auto& part : std::get<std::vector<ThreadMessageContentPart>>(content)) {
      array.push_back(content_part_to_json(part));
    }
    return array;
  }
  return json();
}

json thread_resources_to_json(const ThreadToolResources& resources) {
  json value = json::object();
  if (!resources.code_interpreter_file_ids.empty()) {
    value["code_interpreter"] = json::object({{"file_ids", resources.code_interpreter_file_ids}});
  }
  if (!resources.file_search_vector_store_ids.empty()) {
    value["file_search"] = json::object({{"vector_store_ids", resources.file_search_vector_store_ids}});
  }
  return value;
}

json metadata_to_json(const std::map<std::string, std::string>& metadata) {
  json value = json::object();
  for (const auto& entry : metadata) {
    value[entry.first] = entry.second;
  }
  return value;
}

json create_request_to_json(const ThreadCreateRequest& request) {
  json body;
  if (!request.messages.empty()) {
    json messages = json::array();
    for (const auto& message : request.messages) {
      json message_json;
      message_json["role"] = message.role;
      message_json["content"] = content_to_json(message.content);
      if (!message.attachments.empty()) {
        message_json["attachments"] = attachments_to_json(message.attachments);
      }
      if (!message.metadata.empty()) {
        message_json["metadata"] = metadata_to_json(message.metadata);
      }
      messages.push_back(std::move(message_json));
    }
    body["messages"] = std::move(messages);
  }
  if (!request.metadata.empty()) {
    body["metadata"] = metadata_to_json(request.metadata);
  }
  if (request.tool_resources) {
    const auto tools = thread_resources_to_json(*request.tool_resources);
    if (!tools.empty()) body["tool_resources"] = tools;
  }
  return body;
}

json update_request_to_json(const ThreadUpdateRequest& request) {
  json body;
  if (request.metadata) body["metadata"] = metadata_to_json(*request.metadata);
  if (request.tool_resources) body["tool_resources"] = thread_resources_to_json(*request.tool_resources);
  return body;
}

ThreadToolResources parse_tool_resources(const json& payload) {
  ThreadToolResources resources;
  if (payload.contains("code_interpreter") && payload["code_interpreter"].is_object()) {
    const auto& ci = payload.at("code_interpreter");
    if (ci.contains("file_ids") && ci["file_ids"].is_array()) {
      for (const auto& item : ci.at("file_ids")) {
        if (item.is_string()) resources.code_interpreter_file_ids.push_back(item.get<std::string>());
      }
    }
  }
  if (payload.contains("file_search") && payload["file_search"].is_object()) {
    const auto& fs = payload.at("file_search");
    if (fs.contains("vector_store_ids") && fs["vector_store_ids"].is_array()) {
      for (const auto& item : fs.at("vector_store_ids")) {
        if (item.is_string()) resources.file_search_vector_store_ids.push_back(item.get<std::string>());
      }
    }
  }
  return resources;
}

Thread parse_thread_impl(const json& payload) {
  Thread thread;
  thread.raw = payload;
  thread.id = payload.value("id", "");
  thread.created_at = payload.value("created_at", 0);
  thread.object = payload.value("object", "");
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) thread.metadata[it.key()] = it.value().get<std::string>();
    }
  }
  if (payload.contains("tool_resources") && payload["tool_resources"].is_object()) {
    thread.tool_resources = parse_tool_resources(payload.at("tool_resources"));
  }
  return thread;
}

ThreadDeleteResponse parse_thread_delete(const json& payload) {
  ThreadDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

}  // namespace

Thread parse_thread_json(const nlohmann::json& payload) {
  return parse_thread_impl(payload);
}

Thread ThreadsResource::create(const ThreadCreateRequest& request) const {
  return create(request, RequestOptions{});
}

Thread ThreadsResource::create(const ThreadCreateRequest& request, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = create_request_to_json(request);
  auto response = client_.perform_request("POST", kThreadsPath, body.dump(), request_options);
  try {
    return parse_thread_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread: ") + ex.what());
  }
}

Thread ThreadsResource::retrieve(const std::string& thread_id) const {
  return retrieve(thread_id, RequestOptions{});
}

Thread ThreadsResource::retrieve(const std::string& thread_id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", std::string(kThreadsPath) + "/" + thread_id, "", request_options);
  try {
    return parse_thread_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread: ") + ex.what());
  }
}

Thread ThreadsResource::update(const std::string& thread_id, const ThreadUpdateRequest& request) const {
  return update(thread_id, request, RequestOptions{});
}

Thread ThreadsResource::update(const std::string& thread_id,
                               const ThreadUpdateRequest& request,
                               const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = update_request_to_json(request);
  auto response = client_.perform_request("POST", std::string(kThreadsPath) + "/" + thread_id, body.dump(), request_options);
  try {
    return parse_thread_impl(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread: ") + ex.what());
  }
}

ThreadDeleteResponse ThreadsResource::remove(const std::string& thread_id) const {
  return remove(thread_id, RequestOptions{});
}

ThreadDeleteResponse ThreadsResource::remove(const std::string& thread_id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("DELETE", std::string(kThreadsPath) + "/" + thread_id, "", request_options);
  try {
    return parse_thread_delete(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse thread delete response: ") + ex.what());
  }
}

}  // namespace openai

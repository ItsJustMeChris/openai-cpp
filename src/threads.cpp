#include "openai/threads.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/runs.hpp"
#include "openai/assistant_stream.hpp"
#include "openai/streaming.hpp"

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

json chunking_strategy_to_json(const ThreadToolResources::FileSearchVectorStoreChunkingStrategy& strategy) {
  json chunking = json::object();
  chunking["type"] = strategy.type;
  if (strategy.chunk_overlap_tokens) chunking["chunk_overlap_tokens"] = *strategy.chunk_overlap_tokens;
  if (strategy.max_chunk_size_tokens) chunking["max_chunk_size_tokens"] = *strategy.max_chunk_size_tokens;
  return chunking;
}

json vector_store_to_json(const ThreadToolResources::FileSearchVectorStore& store) {
  json value = json::object();
  if (store.chunking_strategy) {
    value["chunking_strategy"] = chunking_strategy_to_json(*store.chunking_strategy);
  }
  if (!store.file_ids.empty()) value["file_ids"] = store.file_ids;
  if (store.metadata && !store.metadata->empty()) value["metadata"] = *store.metadata;
  return value;
}

json thread_resources_to_json(const ThreadToolResources& resources) {
  json value = json::object();
  if (resources.code_interpreter && !resources.code_interpreter->file_ids.empty()) {
    value["code_interpreter"] = json::object({{"file_ids", resources.code_interpreter->file_ids}});
  }
  if (resources.file_search) {
    json file_search = json::object();
    if (!resources.file_search->vector_store_ids.empty()) {
      file_search["vector_store_ids"] = resources.file_search->vector_store_ids;
    }
    if (!resources.file_search->vector_stores.empty()) {
      json stores = json::array();
      for (const auto& store : resources.file_search->vector_stores) {
        json store_json = vector_store_to_json(store);
        if (!store_json.empty()) stores.push_back(std::move(store_json));
      }
      if (!stores.empty()) file_search["vector_stores"] = std::move(stores);
    }
    if (!file_search.empty()) value["file_search"] = std::move(file_search);
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

json build_thread_create_body(const ThreadCreateRequest& request) {
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

ThreadToolResources::FileSearchVectorStoreChunkingStrategy parse_chunking_strategy(const json& payload) {
  ThreadToolResources::FileSearchVectorStoreChunkingStrategy strategy;
  strategy.type = payload.value("type", "");
  if (payload.contains("chunk_overlap_tokens") && payload.at("chunk_overlap_tokens").is_number_integer()) {
    strategy.chunk_overlap_tokens = payload.at("chunk_overlap_tokens").get<int>();
  }
  if (payload.contains("max_chunk_size_tokens") && payload.at("max_chunk_size_tokens").is_number_integer()) {
    strategy.max_chunk_size_tokens = payload.at("max_chunk_size_tokens").get<int>();
  }
  if (payload.contains("static") && payload.at("static").is_object()) {
    const auto& static_obj = payload.at("static");
    if (static_obj.contains("chunk_overlap_tokens") && static_obj.at("chunk_overlap_tokens").is_number_integer()) {
      strategy.chunk_overlap_tokens = static_obj.at("chunk_overlap_tokens").get<int>();
    }
    if (static_obj.contains("max_chunk_size_tokens") && static_obj.at("max_chunk_size_tokens").is_number_integer()) {
      strategy.max_chunk_size_tokens = static_obj.at("max_chunk_size_tokens").get<int>();
    }
  }
  return strategy;
}

std::map<std::string, std::string> parse_metadata_object(const json& payload) {
  std::map<std::string, std::string> metadata;
  if (!payload.is_object()) return metadata;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    if (it.value().is_string()) metadata[it.key()] = it.value().get<std::string>();
  }
  return metadata;
}

ThreadToolResources parse_tool_resources(const json& payload) {
  ThreadToolResources resources;
  if (payload.contains("code_interpreter") && payload["code_interpreter"].is_object()) {
    ThreadToolResources::CodeInterpreter code_interpreter;
    const auto& ci = payload.at("code_interpreter");
    if (ci.contains("file_ids") && ci["file_ids"].is_array()) {
      for (const auto& item : ci.at("file_ids")) {
        if (item.is_string()) code_interpreter.file_ids.push_back(item.get<std::string>());
      }
    }
    if (!code_interpreter.file_ids.empty()) resources.code_interpreter = std::move(code_interpreter);
  }
  if (payload.contains("file_search") && payload["file_search"].is_object()) {
    ThreadToolResources::FileSearch file_search;
    const auto& fs = payload.at("file_search");
    if (fs.contains("vector_store_ids") && fs["vector_store_ids"].is_array()) {
      for (const auto& item : fs.at("vector_store_ids")) {
        if (item.is_string()) file_search.vector_store_ids.push_back(item.get<std::string>());
      }
    }
    if (fs.contains("vector_stores") && fs["vector_stores"].is_array()) {
      for (const auto& store_json : fs.at("vector_stores")) {
        if (!store_json.is_object()) continue;
        ThreadToolResources::FileSearchVectorStore store;
        if (store_json.contains("chunking_strategy") && store_json.at("chunking_strategy").is_object()) {
          store.chunking_strategy = parse_chunking_strategy(store_json.at("chunking_strategy"));
        }
        if (store_json.contains("file_ids") && store_json.at("file_ids").is_array()) {
          for (const auto& id : store_json.at("file_ids")) {
            if (id.is_string()) store.file_ids.push_back(id.get<std::string>());
          }
        }
        if (store_json.contains("metadata") && store_json.at("metadata").is_object()) {
          auto metadata = parse_metadata_object(store_json.at("metadata"));
          if (!metadata.empty()) store.metadata = std::move(metadata);
        }
        file_search.vector_stores.push_back(std::move(store));
      }
    }
    if (!file_search.vector_store_ids.empty() || !file_search.vector_stores.empty()) {
      resources.file_search = std::move(file_search);
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
  const auto body = build_thread_create_body(request);
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

Run ThreadsResource::create_and_run(const ThreadCreateAndRunRequest& request) const {
  return create_and_run(request, RequestOptions{});
}

Run ThreadsResource::create_and_run(const ThreadCreateAndRunRequest& request, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);

  if (request.run.include && !request.run.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < request.run.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*request.run.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }

  json body;
  if (request.thread) {
    body["thread"] = build_thread_create_body(*request.thread);
  }

  auto run_body = build_run_create_body(request.run);
  for (auto it = run_body.begin(); it != run_body.end(); ++it) {
    body[it.key()] = it.value();
  }

  auto response = client_.perform_request("POST", std::string(kThreadsPath) + "/runs", body.dump(), request_options);
  try {
    return parse_run_json(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse run: ") + ex.what());
  }
}

std::vector<AssistantStreamEvent> ThreadsResource::create_and_run_stream(const ThreadCreateAndRunRequest& request) const {
  return create_and_run_stream_snapshot(request).events();
}

std::vector<AssistantStreamEvent> ThreadsResource::create_and_run_stream(const ThreadCreateAndRunRequest& request,
                                                                        const RequestOptions& options) const {
  return create_and_run_stream_snapshot(request, options).events();
}

AssistantStreamSnapshot ThreadsResource::create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request) const {
  return create_and_run_stream_snapshot(request, RequestOptions{});
}

AssistantStreamSnapshot ThreadsResource::create_and_run_stream_snapshot(const ThreadCreateAndRunRequest& request,
                                                                       const RequestOptions& options) const {
  ThreadCreateAndRunRequest streaming_request = request;
  streaming_request.run.stream = true;

  RequestOptions request_options = options;
  apply_beta_header(request_options);
  request_options.collect_body = false;
  request_options.headers["X-Stainless-Helper-Method"] = "stream";

  if (streaming_request.run.include && !streaming_request.run.include->empty()) {
    std::string joined;
    for (size_t i = 0; i < streaming_request.run.include->size(); ++i) {
      if (i > 0) joined += ",";
      joined += (*streaming_request.run.include)[i];
    }
    request_options.query_params["include"] = std::move(joined);
  }

  AssistantStreamSnapshot snapshot;
  AssistantStreamParser parser([&](const AssistantStreamEvent& ev) { snapshot.ingest(ev); });
  SSEEventStream stream([&](const ServerSentEvent& sse_event) {
    parser.feed(sse_event);
    return true;
  });
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  json body;
  if (streaming_request.thread) {
    body["thread"] = build_thread_create_body(*streaming_request.thread);
  }
  auto run_body = build_run_create_body(streaming_request.run);
  for (auto it = run_body.begin(); it != run_body.end(); ++it) {
    body[it.key()] = it.value();
  }

  client_.perform_request("POST", std::string(kThreadsPath) + "/runs", body.dump(), request_options);

  stream.finalize();

  return snapshot;
}

Run ThreadsResource::create_and_run_poll(const ThreadCreateAndRunRequest& request) const {
  return create_and_run_poll(request, RequestOptions{}, std::chrono::milliseconds(5000));
}

Run ThreadsResource::create_and_run_poll(const ThreadCreateAndRunRequest& request,
                                         const RequestOptions& options,
                                         std::chrono::milliseconds poll_interval) const {
  auto run = create_and_run(request, options);
  if (run.thread_id.empty()) {
    throw OpenAIError("Run response missing thread_id for polling");
  }

  RunRetrieveParams params;
  params.thread_id = run.thread_id;
  return client_.runs().poll(run.id, params, options, poll_interval);
}

}  // namespace openai

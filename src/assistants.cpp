#include "openai/assistants.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"

#include <nlohmann/json.hpp>

#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kAssistantsPath = "/assistants";
constexpr const char* kBetaHeaderName = "OpenAI-Beta";
constexpr const char* kBetaHeaderValue = "assistants=v2";

void apply_beta_header(RequestOptions& options) {
  options.headers[kBetaHeaderName] = kBetaHeaderValue;
}

const char* to_string(AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type type) {
  switch (type) {
    case AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Auto:
      return "auto";
    case AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Static:
      return "static";
  }
  return "auto";
}

AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type chunking_strategy_type_from_string(
    const std::string& value) {
  if (value == "static") {
    return AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Static;
  }
  return AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Auto;
}

json tool_to_json(const AssistantTool& tool) {
  json value;
  switch (tool.type) {
    case AssistantTool::Type::CodeInterpreter:
      value["type"] = "code_interpreter";
      break;
    case AssistantTool::Type::FileSearch:
      value["type"] = "file_search";
      if (tool.file_search) {
        json overrides;
        if (tool.file_search->max_num_results) overrides["max_num_results"] = *tool.file_search->max_num_results;
        if (tool.file_search->ranking_options) {
          json ranking;
          if (tool.file_search->ranking_options->ranker) ranking["ranker"] = *tool.file_search->ranking_options->ranker;
          if (tool.file_search->ranking_options->score_threshold) {
            ranking["score_threshold"] = *tool.file_search->ranking_options->score_threshold;
          }
          overrides["ranking_options"] = std::move(ranking);
        }
        value["file_search"] = std::move(overrides);
      }
      break;
    case AssistantTool::Type::Function:
      value["type"] = "function";
      if (tool.function) {
        json fn;
        fn["name"] = tool.function->name;
        if (tool.function->description) fn["description"] = *tool.function->description;
        if (!tool.function->parameters.is_null() && !tool.function->parameters.empty()) {
          fn["parameters"] = tool.function->parameters;
        }
        value["function"] = std::move(fn);
      }
      break;
  }
  return value;
}

AssistantTool parse_tool(const json& payload) {
  AssistantTool tool;
  const std::string type = payload.value("type", "");
  if (type == "code_interpreter") {
    tool.type = AssistantTool::Type::CodeInterpreter;
  } else if (type == "file_search") {
    tool.type = AssistantTool::Type::FileSearch;
    if (payload.contains("file_search") && payload["file_search"].is_object()) {
      AssistantTool::FileSearchOverrides overrides;
      const auto& obj = payload.at("file_search");
      if (obj.contains("max_num_results") && obj["max_num_results"].is_number_integer()) {
        overrides.max_num_results = obj["max_num_results"].get<int>();
      }
      if (obj.contains("ranking_options") && obj["ranking_options"].is_object()) {
        const auto& ranking = obj.at("ranking_options");
        AssistantTool::FileSearchOverrides::RankingOptions options;
        bool has_options = false;
        if (ranking.contains("ranker") && ranking["ranker"].is_string()) {
          options.ranker = ranking["ranker"].get<std::string>();
          has_options = true;
        }
        if (ranking.contains("score_threshold") && ranking["score_threshold"].is_number()) {
          options.score_threshold = ranking["score_threshold"].get<double>();
          has_options = true;
        }
        if (has_options) {
          overrides.ranking_options = options;
        }
      }
      tool.file_search = overrides;
    }
  } else if (type == "function") {
    tool.type = AssistantTool::Type::Function;
    if (payload.contains("function") && payload["function"].is_object()) {
      AssistantTool::FunctionDefinition fn;
      const auto& fn_obj = payload.at("function");
      fn.name = fn_obj.value("name", "");
      if (fn_obj.contains("description") && fn_obj["description"].is_string()) {
        fn.description = fn_obj["description"].get<std::string>();
      }
      if (fn_obj.contains("parameters")) {
        fn.parameters = fn_obj.at("parameters");
      }
      tool.function = fn;
    }
  }
  return tool;
}

json tool_resources_to_json(const AssistantToolResources& resources) {
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
      json vector_stores = json::array();
      for (const auto& store : resources.file_search->vector_stores) {
        json store_json = json::object();
        if (store.chunking_strategy) {
          json chunking;
          const auto& strategy = *store.chunking_strategy;
          chunking["type"] = to_string(strategy.type);
          if (strategy.type ==
                  AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Static &&
              strategy.static_options) {
            json static_json;
            static_json["chunk_overlap_tokens"] = strategy.static_options->chunk_overlap_tokens;
            static_json["max_chunk_size_tokens"] = strategy.static_options->max_chunk_size_tokens;
            chunking["static"] = std::move(static_json);
          }
          store_json["chunking_strategy"] = std::move(chunking);
        }
        if (!store.file_ids.empty()) {
          store_json["file_ids"] = store.file_ids;
        }
        if (store.metadata && !store.metadata->empty()) {
          json metadata = json::object();
          for (const auto& entry : *store.metadata) {
            metadata[entry.first] = entry.second;
          }
          store_json["metadata"] = std::move(metadata);
        }
        vector_stores.push_back(std::move(store_json));
      }
      file_search["vector_stores"] = std::move(vector_stores);
    }
    if (!file_search.empty()) {
      value["file_search"] = std::move(file_search);
    }
  }
  return value;
}

AssistantToolResources parse_tool_resources(const json& payload) {
  AssistantToolResources resources;
  if (payload.contains("code_interpreter") && payload["code_interpreter"].is_object()) {
    const auto& ci = payload.at("code_interpreter");
    AssistantToolResources::CodeInterpreter code_interpreter;
    if (ci.contains("file_ids") && ci["file_ids"].is_array()) {
      for (const auto& item : ci.at("file_ids")) {
        if (item.is_string()) code_interpreter.file_ids.push_back(item.get<std::string>());
      }
    }
    if (!code_interpreter.file_ids.empty()) resources.code_interpreter = code_interpreter;
  }
  if (payload.contains("file_search") && payload["file_search"].is_object()) {
    const auto& fs = payload.at("file_search");
    AssistantToolResources::FileSearch file_search;
    if (fs.contains("vector_store_ids") && fs["vector_store_ids"].is_array()) {
      for (const auto& item : fs.at("vector_store_ids")) {
        if (item.is_string()) file_search.vector_store_ids.push_back(item.get<std::string>());
      }
    }
    if (fs.contains("vector_stores") && fs["vector_stores"].is_array()) {
      for (const auto& entry : fs.at("vector_stores")) {
        if (!entry.is_object()) continue;
        AssistantToolResources::FileSearch::VectorStore store;
        if (entry.contains("chunking_strategy") && entry["chunking_strategy"].is_object()) {
          const auto& chunking = entry.at("chunking_strategy");
          AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy strategy;
          strategy.type = chunking_strategy_type_from_string(chunking.value("type", "auto"));
          if (strategy.type ==
                  AssistantToolResources::FileSearch::VectorStore::ChunkingStrategy::Type::Static &&
              chunking.contains("static") && chunking["static"].is_object()) {
            const auto& static_obj = chunking.at("static");
            AssistantToolResources::FileSearch::VectorStore::StaticChunking static_opts;
            static_opts.chunk_overlap_tokens = static_obj.value("chunk_overlap_tokens", 0);
            static_opts.max_chunk_size_tokens = static_obj.value("max_chunk_size_tokens", 0);
            strategy.static_options = static_opts;
          }
          store.chunking_strategy = strategy;
        }
        if (entry.contains("file_ids") && entry["file_ids"].is_array()) {
          for (const auto& file_id : entry.at("file_ids")) {
            if (file_id.is_string()) store.file_ids.push_back(file_id.get<std::string>());
          }
        }
        if (entry.contains("metadata") && entry["metadata"].is_object()) {
          std::map<std::string, std::string> metadata;
          for (auto it = entry["metadata"].begin(); it != entry["metadata"].end(); ++it) {
            if (it.value().is_string()) metadata[it.key()] = it.value().get<std::string>();
          }
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

json response_format_to_json(const AssistantResponseFormat& format) {
  json value;
  value["type"] = format.type;
  if (!format.json_schema.is_null() && !format.json_schema.empty()) {
    value["json_schema"] = format.json_schema;
  }
  return value;
}

AssistantResponseFormat parse_response_format(const json& payload) {
  AssistantResponseFormat format;
  format.type = payload.value("type", "");
  if (payload.contains("json_schema")) {
    format.json_schema = payload.at("json_schema");
  }
  return format;
}

json metadata_to_json(const std::map<std::string, std::string>& metadata) {
  json value = json::object();
  for (const auto& entry : metadata) {
    value[entry.first] = entry.second;
  }
  return value;
}

Assistant parse_assistant(const json& payload) {
  Assistant assistant;
  assistant.raw = payload;
  assistant.id = payload.value("id", "");
  assistant.created_at = payload.value("created_at", 0);
  if (payload.contains("description") && !payload["description"].is_null()) {
    assistant.description = payload["description"].get<std::string>();
  }
  if (payload.contains("instructions") && !payload["instructions"].is_null()) {
    assistant.instructions = payload["instructions"].get<std::string>();
  }
  if (payload.contains("metadata") && payload["metadata"].is_object()) {
    for (auto it = payload["metadata"].begin(); it != payload["metadata"].end(); ++it) {
      if (it.value().is_string()) assistant.metadata[it.key()] = it.value().get<std::string>();
    }
  }
  assistant.model = payload.value("model", "");
  if (payload.contains("name") && !payload["name"].is_null()) {
    assistant.name = payload["name"].get<std::string>();
  }
  assistant.object = payload.value("object", "");
  if (payload.contains("tools") && payload["tools"].is_array()) {
    for (const auto& item : payload.at("tools")) {
      assistant.tools.push_back(parse_tool(item));
    }
  }
  if (payload.contains("response_format") && !payload["response_format"].is_null()) {
    assistant.response_format = parse_response_format(payload.at("response_format"));
  }
  if (payload.contains("temperature") && payload["temperature"].is_number()) {
    assistant.temperature = payload["temperature"].get<double>();
  }
  if (payload.contains("top_p") && payload["top_p"].is_number()) {
    assistant.top_p = payload["top_p"].get<double>();
  }
  if (payload.contains("reasoning_effort") && !payload["reasoning_effort"].is_null()) {
    assistant.reasoning_effort = payload["reasoning_effort"].get<std::string>();
  }
  if (payload.contains("tool_resources") && payload["tool_resources"].is_object()) {
    assistant.tool_resources = parse_tool_resources(payload.at("tool_resources"));
  }
  return assistant;
}

AssistantList parse_list(const json& payload) {
  AssistantList list;
  list.raw = payload;
  list.has_more = payload.value("has_more", false);
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_assistant(item));
    }
  }
  return list;
}

AssistantDeleteResponse parse_delete_response(const json& payload) {
  AssistantDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

json create_request_to_json(const AssistantCreateRequest& request) {
  json body;
  body["model"] = request.model;
  if (request.description) body["description"] = *request.description;
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.name) body["name"] = *request.name;
  if (!request.metadata.empty()) body["metadata"] = metadata_to_json(request.metadata);
  if (!request.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : request.tools) tools.push_back(tool_to_json(tool));
    body["tools"] = std::move(tools);
  }
  if (request.tool_resources) {
    const auto tool_resource_json = tool_resources_to_json(*request.tool_resources);
    if (!tool_resource_json.empty()) body["tool_resources"] = tool_resource_json;
  }
  if (request.response_format) body["response_format"] = response_format_to_json(*request.response_format);
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.reasoning_effort) body["reasoning_effort"] = *request.reasoning_effort;
  return body;
}

json update_request_to_json(const AssistantUpdateRequest& request) {
  json body;
  if (request.model) body["model"] = *request.model;
  if (request.description) body["description"] = *request.description;
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.name) body["name"] = *request.name;
  if (request.metadata) body["metadata"] = metadata_to_json(*request.metadata);
  if (request.tools) {
    json tools = json::array();
    for (const auto& tool : *request.tools) tools.push_back(tool_to_json(tool));
    body["tools"] = std::move(tools);
  }
  if (request.tool_resources) {
    const auto tool_resource_json = tool_resources_to_json(*request.tool_resources);
    body["tool_resources"] = tool_resource_json;
  }
  if (request.response_format) body["response_format"] = response_format_to_json(*request.response_format);
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (request.reasoning_effort) body["reasoning_effort"] = *request.reasoning_effort;
  return body;
}

}  // namespace

Assistant AssistantsResource::create(const AssistantCreateRequest& request) const {
  return create(request, RequestOptions{});
}

Assistant AssistantsResource::create(const AssistantCreateRequest& request, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = create_request_to_json(request);
  auto response = client_.perform_request("POST", kAssistantsPath, body.dump(), request_options);
  try {
    return parse_assistant(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistant: ") + ex.what());
  }
}

Assistant AssistantsResource::retrieve(const std::string& assistant_id) const {
  return retrieve(assistant_id, RequestOptions{});
}

Assistant AssistantsResource::retrieve(const std::string& assistant_id, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("GET", std::string(kAssistantsPath) + "/" + assistant_id, "", request_options);
  try {
    return parse_assistant(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistant: ") + ex.what());
  }
}

Assistant AssistantsResource::update(const std::string& assistant_id, const AssistantUpdateRequest& request) const {
  return update(assistant_id, request, RequestOptions{});
}

Assistant AssistantsResource::update(const std::string& assistant_id,
                                     const AssistantUpdateRequest& request,
                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  const auto body = update_request_to_json(request);
  auto response = client_.perform_request("POST", std::string(kAssistantsPath) + "/" + assistant_id, body.dump(),
                                          request_options);
  try {
    return parse_assistant(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistant: ") + ex.what());
  }
}

AssistantDeleteResponse AssistantsResource::remove(const std::string& assistant_id) const {
  return remove(assistant_id, RequestOptions{});
}

AssistantDeleteResponse AssistantsResource::remove(const std::string& assistant_id,
                                                   const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  auto response = client_.perform_request("DELETE", std::string(kAssistantsPath) + "/" + assistant_id, "",
                                          request_options);
  try {
    return parse_delete_response(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistant delete response: ") + ex.what());
  }
}

AssistantList AssistantsResource::list() const {
  return list(AssistantListParams{}, RequestOptions{});
}

AssistantList AssistantsResource::list(const RequestOptions& options) const {
  return list(AssistantListParams{}, options);
}

AssistantList AssistantsResource::list(const AssistantListParams& params) const {
  return list(params, RequestOptions{});
}

AssistantList AssistantsResource::list(const AssistantListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  apply_beta_header(request_options);
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;
  if (params.before) request_options.query_params["before"] = *params.before;
  auto response = client_.perform_request("GET", kAssistantsPath, "", request_options);
  try {
    return parse_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse assistants list: ") + ex.what());
  }
}

}  // namespace openai

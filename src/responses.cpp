#include "openai/responses.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"
#include "openai/streaming.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <sstream>
#include <utility>

namespace openai {
namespace {

using json = nlohmann::json;

const char* kResponseEndpoint = "/responses";

std::string to_string(ResponseComputerEnvironment environment) {
  switch (environment) {
    case ResponseComputerEnvironment::Windows:
      return "windows";
    case ResponseComputerEnvironment::Mac:
      return "mac";
    case ResponseComputerEnvironment::Linux:
      return "linux";
    case ResponseComputerEnvironment::Ubuntu:
      return "ubuntu";
    case ResponseComputerEnvironment::Browser:
      return "browser";
    case ResponseComputerEnvironment::Unknown:
    default:
      return "";
  }
}

ResponseComputerEnvironment computer_environment_from_string(const std::string& value) {
  if (value == "windows") return ResponseComputerEnvironment::Windows;
  if (value == "mac") return ResponseComputerEnvironment::Mac;
  if (value == "linux") return ResponseComputerEnvironment::Linux;
  if (value == "ubuntu") return ResponseComputerEnvironment::Ubuntu;
  if (value == "browser") return ResponseComputerEnvironment::Browser;
  return ResponseComputerEnvironment::Unknown;
}

std::string to_string(ResponseCustomToolFormat::Type type) {
  switch (type) {
    case ResponseCustomToolFormat::Type::Text:
      return "text";
    case ResponseCustomToolFormat::Type::Grammar:
      return "grammar";
    case ResponseCustomToolFormat::Type::Unknown:
    default:
      return "";
  }
}

ResponseCustomToolFormat::Type custom_tool_format_from_string(const std::string& value) {
  if (value == "text") return ResponseCustomToolFormat::Type::Text;
  if (value == "grammar") return ResponseCustomToolFormat::Type::Grammar;
  return ResponseCustomToolFormat::Type::Unknown;
}

std::string to_string(ResponseToolChoiceSimpleOption option) {
  switch (option) {
    case ResponseToolChoiceSimpleOption::None:
      return "none";
    case ResponseToolChoiceSimpleOption::Auto:
      return "auto";
    case ResponseToolChoiceSimpleOption::Required:
      return "required";
  }
  return "auto";
}

std::optional<ResponseToolChoiceSimpleOption> tool_choice_simple_from_string(const std::string& value) {
  if (value == "none") return ResponseToolChoiceSimpleOption::None;
  if (value == "auto") return ResponseToolChoiceSimpleOption::Auto;
  if (value == "required") return ResponseToolChoiceSimpleOption::Required;
  return std::nullopt;
}

std::string to_string(ResponseToolChoiceAllowedMode mode) {
  switch (mode) {
    case ResponseToolChoiceAllowedMode::Auto:
      return "auto";
    case ResponseToolChoiceAllowedMode::Required:
      return "required";
  }
  return "auto";
}

std::optional<ResponseToolChoiceAllowedMode> tool_choice_allowed_mode_from_string(const std::string& value) {
  if (value == "auto") return ResponseToolChoiceAllowedMode::Auto;
  if (value == "required") return ResponseToolChoiceAllowedMode::Required;
  return std::nullopt;
}

nlohmann::json serialize_search_filter(const ResponseSearchFilter& filter);
ResponseSearchFilter parse_search_filter(const json& payload);

nlohmann::json serialize_custom_tool_format(const ResponseCustomToolFormat& format) {
  json data = format.raw.is_object() ? format.raw : json::object();
  const auto type_str = to_string(format.type);
  if (!type_str.empty()) {
    data["type"] = type_str;
  }
  if (format.definition) {
    data["definition"] = *format.definition;
  }
  if (format.syntax) {
    data["syntax"] = *format.syntax;
  }
  return data;
}

nlohmann::json serialize_mcp_tool_filter(const ResponseMcpToolFilter& filter) {
  json data = filter.raw.is_object() ? filter.raw : json::object();
  if (filter.read_only.has_value()) {
    data["read_only"] = *filter.read_only;
  }
  if (!filter.tool_names.empty()) {
    data["tool_names"] = filter.tool_names;
  }
  return data;
}

nlohmann::json serialize_mcp_tool_approval_filter(const ResponseMcpToolApprovalFilter& filter) {
  json data = filter.raw.is_object() ? filter.raw : json::object();
  if (filter.always) {
    data["always"] = serialize_mcp_tool_filter(*filter.always);
  }
  if (filter.never) {
    data["never"] = serialize_mcp_tool_filter(*filter.never);
  }
  return data;
}

nlohmann::json serialize_web_search_filters(const ResponseWebSearchToolFilters& filters) {
  json data = filters.raw.is_object() ? filters.raw : json::object();
  if (!filters.allowed_domains.empty()) {
    data["allowed_domains"] = filters.allowed_domains;
  }
  return data;
}

nlohmann::json serialize_web_search_location(const ResponseWebSearchToolUserLocation& location) {
  json data = location.raw.is_object() ? location.raw : json::object();
  if (location.city) data["city"] = *location.city;
  if (location.country) data["country"] = *location.country;
  if (location.region) data["region"] = *location.region;
  if (location.timezone) data["timezone"] = *location.timezone;
  if (location.type) data["type"] = *location.type;
  return data;
}

nlohmann::json serialize_web_search_preview_location(const ResponseWebSearchPreviewUserLocation& location) {
  json data = location.raw.is_object() ? location.raw : json::object();
  if (location.city) data["city"] = *location.city;
  if (location.country) data["country"] = *location.country;
  if (location.region) data["region"] = *location.region;
  if (location.timezone) data["timezone"] = *location.timezone;
  if (location.type) data["type"] = *location.type;
  return data;
}

nlohmann::json serialize_code_interpreter_container(const ResponseCodeInterpreterToolDefinition& code_interpreter) {
  json data = code_interpreter.raw.is_object() ? code_interpreter.raw : json::object();
  if (std::holds_alternative<std::string>(code_interpreter.container)) {
    data["container"] = std::get<std::string>(code_interpreter.container);
  } else if (std::holds_alternative<ResponseCodeInterpreterAutoContainer>(code_interpreter.container)) {
    const auto& container = std::get<ResponseCodeInterpreterAutoContainer>(code_interpreter.container);
    json container_json = container.raw.is_object() ? container.raw : json::object();
    container_json["type"] = "auto";
    if (!container.file_ids.empty()) {
      container_json["file_ids"] = container.file_ids;
    }
    data["container"] = std::move(container_json);
  }
  return data;
}

nlohmann::json serialize_image_generation_mask(const ResponseImageGenerationToolMask& mask) {
  json data = mask.raw.is_object() ? mask.raw : json::object();
  if (mask.image_url) data["image_url"] = *mask.image_url;
  if (mask.file_id) data["file_id"] = *mask.file_id;
  return data;
}

std::string to_string(ResponseSearchFilter::Comparison::Operator op) {
  switch (op) {
    case ResponseSearchFilter::Comparison::Operator::Eq:
      return "eq";
    case ResponseSearchFilter::Comparison::Operator::Ne:
      return "ne";
    case ResponseSearchFilter::Comparison::Operator::Gt:
      return "gt";
    case ResponseSearchFilter::Comparison::Operator::Gte:
      return "gte";
    case ResponseSearchFilter::Comparison::Operator::Lt:
      return "lt";
    case ResponseSearchFilter::Comparison::Operator::Lte:
      return "lte";
    case ResponseSearchFilter::Comparison::Operator::In:
      return "in";
    case ResponseSearchFilter::Comparison::Operator::Nin:
      return "nin";
    case ResponseSearchFilter::Comparison::Operator::Unknown:
    default:
      return "unknown";
  }
}

ResponseSearchFilter::Comparison::Operator search_filter_operator_from_string(const std::string& value) {
  if (value == "eq") return ResponseSearchFilter::Comparison::Operator::Eq;
  if (value == "ne") return ResponseSearchFilter::Comparison::Operator::Ne;
  if (value == "gt") return ResponseSearchFilter::Comparison::Operator::Gt;
  if (value == "gte") return ResponseSearchFilter::Comparison::Operator::Gte;
  if (value == "lt") return ResponseSearchFilter::Comparison::Operator::Lt;
  if (value == "lte") return ResponseSearchFilter::Comparison::Operator::Lte;
  if (value == "in") return ResponseSearchFilter::Comparison::Operator::In;
  if (value == "nin") return ResponseSearchFilter::Comparison::Operator::Nin;
  return ResponseSearchFilter::Comparison::Operator::Unknown;
}

std::string to_string(ResponseSearchFilter::Compound::Logical logical) {
  switch (logical) {
    case ResponseSearchFilter::Compound::Logical::And:
      return "and";
    case ResponseSearchFilter::Compound::Logical::Or:
      return "or";
    case ResponseSearchFilter::Compound::Logical::Unknown:
    default:
      return "unknown";
  }
}

ResponseSearchFilter::Compound::Logical search_filter_logical_from_string(const std::string& value) {
  if (value == "and") return ResponseSearchFilter::Compound::Logical::And;
  if (value == "or") return ResponseSearchFilter::Compound::Logical::Or;
  return ResponseSearchFilter::Compound::Logical::Unknown;
}

nlohmann::json serialize_search_filter(const ResponseSearchFilter& filter) {
  if (filter.type == ResponseSearchFilter::Type::Comparison && filter.comparison) {
    json data = filter.raw.is_object() ? filter.raw : json::object();
    data["key"] = filter.comparison->key;
    data["type"] = to_string(filter.comparison->op);
    data["value"] = filter.comparison->value;
    return data;
  }
  if (filter.type == ResponseSearchFilter::Type::Compound && filter.compound) {
    json data = filter.raw.is_object() ? filter.raw : json::object();
    data["type"] = to_string(filter.compound->logical);
    json filters = json::array();
    for (const auto& child : filter.compound->filters) {
      filters.push_back(serialize_search_filter(child));
    }
    data["filters"] = std::move(filters);
    return data;
  }
  if (!filter.raw.is_null()) {
    return filter.raw;
  }
  return json::object();
}

ResponseSearchFilter parse_search_filter(const json& payload) {
  ResponseSearchFilter filter;
  filter.raw = payload;
  if (!payload.is_object()) {
    filter.type = ResponseSearchFilter::Type::Unknown;
    return filter;
  }

  const std::string type = payload.value("type", std::string{});
  if ((type == "and" || type == "or") && payload.contains("filters") && payload.at("filters").is_array()) {
    ResponseSearchFilter::Compound compound;
    compound.logical = search_filter_logical_from_string(type);
    for (const auto& child : payload.at("filters")) {
      compound.filters.push_back(parse_search_filter(child));
    }
    filter.type = ResponseSearchFilter::Type::Compound;
    filter.compound = std::move(compound);
    return filter;
  }

  ResponseSearchFilter::Comparison comparison;
  comparison.key = payload.value("key", std::string{});
  comparison.op = search_filter_operator_from_string(type);
  if (payload.contains("value")) {
    comparison.value = payload.at("value");
  }
  filter.type = ResponseSearchFilter::Type::Comparison;
  filter.comparison = std::move(comparison);
  return filter;
}

nlohmann::json tool_definition_to_json(const ResponseToolDefinition& tool) {
  json data = tool.raw.is_object() ? tool.raw : json::object();
  data["type"] = tool.type;

  if (tool.function) {
    const auto& fn = *tool.function;
    data["name"] = fn.name;
    if (fn.description) data["description"] = *fn.description;
    if (fn.parameters) data["parameters"] = *fn.parameters;
    if (fn.strict.has_value()) data["strict"] = *fn.strict;
  }

  if (tool.custom) {
    const auto& custom = *tool.custom;
    data["name"] = custom.name;
    if (custom.description) data["description"] = *custom.description;
    if (custom.format) {
      data["format"] = serialize_custom_tool_format(*custom.format);
    }
  }

  if (tool.file_search) {
    const auto& file_search = *tool.file_search;
    data["vector_store_ids"] = file_search.vector_store_ids;
    if (file_search.filters) data["filters"] = serialize_search_filter(*file_search.filters);
    if (file_search.max_num_results) data["max_num_results"] = *file_search.max_num_results;
    if (file_search.ranking_options) {
      const auto& ranking = *file_search.ranking_options;
      json ranking_json = ranking.raw.is_object() ? ranking.raw : json::object();
      if (ranking.ranker) ranking_json["ranker"] = *ranking.ranker;
      if (ranking.score_threshold) ranking_json["score_threshold"] = *ranking.score_threshold;
      data["ranking_options"] = std::move(ranking_json);
    }
  }

  if (tool.computer) {
    const auto& computer = *tool.computer;
    data["display_height"] = computer.display_height;
    data["display_width"] = computer.display_width;
    const auto environment = to_string(computer.environment);
    if (!environment.empty()) {
      data["environment"] = environment;
    }
  }

  if (tool.web_search) {
    const auto& web_search = *tool.web_search;
    if (web_search.filters) data["filters"] = serialize_web_search_filters(*web_search.filters);
    if (web_search.search_context_size) data["search_context_size"] = *web_search.search_context_size;
    if (web_search.user_location) data["user_location"] = serialize_web_search_location(*web_search.user_location);
  }

  if (tool.web_search_preview) {
    const auto& preview = *tool.web_search_preview;
    if (preview.search_context_size) data["search_context_size"] = *preview.search_context_size;
    if (preview.user_location) data["user_location"] = serialize_web_search_preview_location(*preview.user_location);
  }

  if (tool.code_interpreter) {
    data = serialize_code_interpreter_container(*tool.code_interpreter);
    data["type"] = tool.type;
  }

  if (tool.image_generation) {
    const auto& image = *tool.image_generation;
    if (image.background) data["background"] = *image.background;
    if (image.input_fidelity) data["input_fidelity"] = *image.input_fidelity;
    if (image.input_image_mask) data["input_image_mask"] = serialize_image_generation_mask(*image.input_image_mask);
    if (image.model) data["model"] = *image.model;
    if (image.moderation) data["moderation"] = *image.moderation;
    if (image.output_compression) data["output_compression"] = *image.output_compression;
    if (image.output_format) data["output_format"] = *image.output_format;
    if (image.visual_quality) data["visual_quality"] = *image.visual_quality;
    if (image.width) data["width"] = *image.width;
    if (image.height) data["height"] = *image.height;
    if (image.aspect_ratio) data["aspect_ratio"] = *image.aspect_ratio;
    if (image.seed) data["seed"] = *image.seed;
  }

  if (tool.mcp) {
    const auto& mcp = *tool.mcp;
    data["server_label"] = mcp.server_label;
    if (mcp.allowed_tool_names) data["allowed_tools"] = *mcp.allowed_tool_names;
    else if (mcp.allowed_tool_filter) data["allowed_tools"] = serialize_mcp_tool_filter(*mcp.allowed_tool_filter);
    if (mcp.authorization) data["authorization"] = *mcp.authorization;
    if (mcp.connector_id) data["connector_id"] = *mcp.connector_id;
    if (!mcp.headers.empty()) data["headers"] = mcp.headers;
    if (std::holds_alternative<std::string>(mcp.require_approval)) {
      data["require_approval"] = std::get<std::string>(mcp.require_approval);
    } else if (std::holds_alternative<ResponseMcpToolApprovalFilter>(mcp.require_approval)) {
      data["require_approval"] = serialize_mcp_tool_approval_filter(
        std::get<ResponseMcpToolApprovalFilter>(mcp.require_approval));
    }
    if (mcp.server_description) data["server_description"] = *mcp.server_description;
    if (mcp.server_url) data["server_url"] = *mcp.server_url;
  }

  return data;
}

ResponseCustomToolFormat parse_custom_tool_format(const json& payload) {
  ResponseCustomToolFormat format;
  format.raw = payload;
  if (payload.contains("type") && payload.at("type").is_string()) {
    format.type = custom_tool_format_from_string(payload.at("type").get<std::string>());
  }
  if (payload.contains("definition") && payload.at("definition").is_string()) {
    format.definition = payload.at("definition").get<std::string>();
  }
  if (payload.contains("syntax") && payload.at("syntax").is_string()) {
    format.syntax = payload.at("syntax").get<std::string>();
  }
  return format;
}

ResponseMcpToolFilter parse_mcp_tool_filter(const json& payload) {
  ResponseMcpToolFilter filter;
  filter.raw = payload;
  if (payload.contains("read_only") && payload.at("read_only").is_boolean()) {
    filter.read_only = payload.at("read_only").get<bool>();
  }
  if (payload.contains("tool_names") && payload.at("tool_names").is_array()) {
    for (const auto& value : payload.at("tool_names")) {
      if (value.is_string()) {
        filter.tool_names.push_back(value.get<std::string>());
      }
    }
  }
  return filter;
}

ResponseMcpToolApprovalFilter parse_mcp_tool_approval_filter(const json& payload) {
  ResponseMcpToolApprovalFilter filter;
  filter.raw = payload;
  if (payload.contains("always") && payload.at("always").is_object()) {
    filter.always = parse_mcp_tool_filter(payload.at("always"));
  }
  if (payload.contains("never") && payload.at("never").is_object()) {
    filter.never = parse_mcp_tool_filter(payload.at("never"));
  }
  return filter;
}

ResponseWebSearchToolFilters parse_web_search_filters(const json& payload) {
  ResponseWebSearchToolFilters filters;
  filters.raw = payload;
  if (payload.contains("allowed_domains") && payload.at("allowed_domains").is_array()) {
    for (const auto& value : payload.at("allowed_domains")) {
      if (value.is_string()) {
        filters.allowed_domains.push_back(value.get<std::string>());
      }
    }
  }
  return filters;
}

ResponseWebSearchToolUserLocation parse_web_search_location(const json& payload) {
  ResponseWebSearchToolUserLocation location;
  location.raw = payload;
  if (payload.contains("city") && payload.at("city").is_string()) {
    location.city = payload.at("city").get<std::string>();
  }
  if (payload.contains("country") && payload.at("country").is_string()) {
    location.country = payload.at("country").get<std::string>();
  }
  if (payload.contains("region") && payload.at("region").is_string()) {
    location.region = payload.at("region").get<std::string>();
  }
  if (payload.contains("timezone") && payload.at("timezone").is_string()) {
    location.timezone = payload.at("timezone").get<std::string>();
  }
  if (payload.contains("type") && payload.at("type").is_string()) {
    location.type = payload.at("type").get<std::string>();
  }
  return location;
}

ResponseWebSearchPreviewUserLocation parse_web_search_preview_location(const json& payload) {
  ResponseWebSearchPreviewUserLocation location;
  location.raw = payload;
  if (payload.contains("city") && payload.at("city").is_string()) {
    location.city = payload.at("city").get<std::string>();
  }
  if (payload.contains("country") && payload.at("country").is_string()) {
    location.country = payload.at("country").get<std::string>();
  }
  if (payload.contains("region") && payload.at("region").is_string()) {
    location.region = payload.at("region").get<std::string>();
  }
  if (payload.contains("timezone") && payload.at("timezone").is_string()) {
    location.timezone = payload.at("timezone").get<std::string>();
  }
  if (payload.contains("type") && payload.at("type").is_string()) {
    location.type = payload.at("type").get<std::string>();
  }
  return location;
}

ResponseCodeInterpreterAutoContainer parse_code_interpreter_auto_container(const json& payload) {
  ResponseCodeInterpreterAutoContainer container;
  container.raw = payload;
  if (payload.contains("file_ids") && payload.at("file_ids").is_array()) {
    for (const auto& value : payload.at("file_ids")) {
      if (value.is_string()) {
        container.file_ids.push_back(value.get<std::string>());
      }
    }
  }
  return container;
}

ResponseImageGenerationToolMask parse_image_generation_mask(const json& payload) {
  ResponseImageGenerationToolMask mask;
  mask.raw = payload;
  if (payload.contains("image_url") && payload.at("image_url").is_string()) {
    mask.image_url = payload.at("image_url").get<std::string>();
  }
  if (payload.contains("file_id") && payload.at("file_id").is_string()) {
    mask.file_id = payload.at("file_id").get<std::string>();
  }
  return mask;
}

ResponseToolDefinition parse_tool_definition(const json& payload) {
  ResponseToolDefinition tool;
  tool.raw = payload;
  if (payload.contains("type") && payload.at("type").is_string()) {
    tool.type = payload.at("type").get<std::string>();
  }

  if (tool.type == "function") {
    ResponseFunctionToolDefinition fn;
    fn.raw = payload;
    fn.name = payload.value("name", "");
    if (payload.contains("description") && payload.at("description").is_string()) {
      fn.description = payload.at("description").get<std::string>();
    }
    if (payload.contains("parameters") && !payload.at("parameters").is_null()) {
      fn.parameters = payload.at("parameters");
    }
    if (payload.contains("strict") && payload.at("strict").is_boolean()) {
      fn.strict = payload.at("strict").get<bool>();
    }
    tool.function = std::move(fn);
  } else if (tool.type == "custom") {
    ResponseCustomToolDefinition custom;
    custom.raw = payload;
    custom.name = payload.value("name", "");
    if (payload.contains("description") && payload.at("description").is_string()) {
      custom.description = payload.at("description").get<std::string>();
    }
    if (payload.contains("format") && payload.at("format").is_object()) {
      custom.format = parse_custom_tool_format(payload.at("format"));
    }
    tool.custom = std::move(custom);
  } else if (tool.type == "file_search") {
    ResponseFileSearchToolDefinition file_search;
    file_search.raw = payload;
    if (payload.contains("vector_store_ids") && payload.at("vector_store_ids").is_array()) {
      for (const auto& value : payload.at("vector_store_ids")) {
        if (value.is_string()) {
          file_search.vector_store_ids.push_back(value.get<std::string>());
        }
      }
    }
    if (payload.contains("filters") && !payload.at("filters").is_null()) {
      file_search.filters = parse_search_filter(payload.at("filters"));
    }
    if (payload.contains("max_num_results") && payload.at("max_num_results").is_number_integer()) {
      file_search.max_num_results = payload.at("max_num_results").get<int>();
    }
    if (payload.contains("ranking_options") && payload.at("ranking_options").is_object()) {
      ResponseFileSearchRankingOptions ranking;
      ranking.raw = payload.at("ranking_options");
      if (ranking.raw.contains("ranker") && ranking.raw.at("ranker").is_string()) {
        ranking.ranker = ranking.raw.at("ranker").get<std::string>();
      }
      if (ranking.raw.contains("score_threshold") && ranking.raw.at("score_threshold").is_number()) {
        ranking.score_threshold = ranking.raw.at("score_threshold").get<double>();
      }
      file_search.ranking_options = std::move(ranking);
    }
    tool.file_search = std::move(file_search);
  } else if (tool.type == "computer_use_preview") {
    ResponseComputerToolDefinition computer;
    computer.raw = payload;
    computer.display_height = payload.value("display_height", 0);
    computer.display_width = payload.value("display_width", 0);
    if (payload.contains("environment") && payload.at("environment").is_string()) {
      computer.environment = computer_environment_from_string(payload.at("environment").get<std::string>());
    }
    tool.computer = std::move(computer);
  } else if (tool.type == "web_search" || tool.type == "web_search_2025_08_26") {
    ResponseWebSearchToolDefinition web_search;
    web_search.raw = payload;
    web_search.type = tool.type;
    if (payload.contains("filters") && payload.at("filters").is_object()) {
      web_search.filters = parse_web_search_filters(payload.at("filters"));
    }
    if (payload.contains("search_context_size") && payload.at("search_context_size").is_string()) {
      web_search.search_context_size = payload.at("search_context_size").get<std::string>();
    }
    if (payload.contains("user_location") && payload.at("user_location").is_object()) {
      web_search.user_location = parse_web_search_location(payload.at("user_location"));
    }
    tool.web_search = std::move(web_search);
  } else if (tool.type == "web_search_preview" || tool.type == "web_search_preview_2025_03_11") {
    ResponseWebSearchPreviewToolDefinition preview;
    preview.raw = payload;
    preview.type = tool.type;
    if (payload.contains("search_context_size") && payload.at("search_context_size").is_string()) {
      preview.search_context_size = payload.at("search_context_size").get<std::string>();
    }
    if (payload.contains("user_location") && payload.at("user_location").is_object()) {
      preview.user_location = parse_web_search_preview_location(payload.at("user_location"));
    }
    tool.web_search_preview = std::move(preview);
  } else if (tool.type == "code_interpreter") {
    ResponseCodeInterpreterToolDefinition code_interpreter;
    code_interpreter.raw = payload;
    if (payload.contains("container")) {
      const auto& container = payload.at("container");
      if (container.is_string()) {
        code_interpreter.container = container.get<std::string>();
      } else if (container.is_object()) {
        code_interpreter.container = parse_code_interpreter_auto_container(container);
      }
    }
    tool.code_interpreter = std::move(code_interpreter);
  } else if (tool.type == "image_generation") {
    ResponseImageGenerationToolDefinition image;
    image.raw = payload;
    if (payload.contains("background") && payload.at("background").is_string()) {
      image.background = payload.at("background").get<std::string>();
    }
    if (payload.contains("input_fidelity") && payload.at("input_fidelity").is_string()) {
      image.input_fidelity = payload.at("input_fidelity").get<std::string>();
    }
    if (payload.contains("input_image_mask") && payload.at("input_image_mask").is_object()) {
      image.input_image_mask = parse_image_generation_mask(payload.at("input_image_mask"));
    }
    if (payload.contains("model") && payload.at("model").is_string()) {
      image.model = payload.at("model").get<std::string>();
    }
    if (payload.contains("moderation") && payload.at("moderation").is_string()) {
      image.moderation = payload.at("moderation").get<std::string>();
    }
    if (payload.contains("output_compression") && payload.at("output_compression").is_number_integer()) {
      image.output_compression = payload.at("output_compression").get<int>();
    }
    if (payload.contains("output_format") && payload.at("output_format").is_string()) {
      image.output_format = payload.at("output_format").get<std::string>();
    }
    if (payload.contains("visual_quality") && payload.at("visual_quality").is_string()) {
      image.visual_quality = payload.at("visual_quality").get<std::string>();
    }
    if (payload.contains("width") && payload.at("width").is_number_integer()) {
      image.width = payload.at("width").get<int>();
    }
    if (payload.contains("height") && payload.at("height").is_number_integer()) {
      image.height = payload.at("height").get<int>();
    }
    if (payload.contains("aspect_ratio") && payload.at("aspect_ratio").is_string()) {
      image.aspect_ratio = payload.at("aspect_ratio").get<std::string>();
    }
    if (payload.contains("seed") && payload.at("seed").is_number_integer()) {
      image.seed = payload.at("seed").get<int>();
    }
    tool.image_generation = std::move(image);
  } else if (tool.type == "local_shell") {
    ResponseLocalShellToolDefinition local_shell;
    local_shell.raw = payload;
    tool.local_shell = std::move(local_shell);
  } else if (tool.type == "mcp") {
    ResponseMcpToolDefinition mcp;
    mcp.raw = payload;
    mcp.server_label = payload.value("server_label", "");
    if (payload.contains("allowed_tools")) {
      const auto& allowed = payload.at("allowed_tools");
      if (allowed.is_array()) {
        std::vector<std::string> names;
        for (const auto& value : allowed) {
          if (value.is_string()) {
            names.push_back(value.get<std::string>());
          }
        }
        if (!names.empty()) {
          mcp.allowed_tool_names = std::move(names);
        }
      } else if (allowed.is_object()) {
        mcp.allowed_tool_filter = parse_mcp_tool_filter(allowed);
      }
    }
    if (payload.contains("authorization") && payload.at("authorization").is_string()) {
      mcp.authorization = payload.at("authorization").get<std::string>();
    }
    if (payload.contains("connector_id") && payload.at("connector_id").is_string()) {
      mcp.connector_id = payload.at("connector_id").get<std::string>();
    }
    if (payload.contains("headers") && payload.at("headers").is_object()) {
      for (auto it = payload.at("headers").begin(); it != payload.at("headers").end(); ++it) {
        if (it.value().is_string()) {
          mcp.headers[it.key()] = it.value().get<std::string>();
        }
      }
    }
    if (payload.contains("require_approval")) {
      const auto& approval = payload.at("require_approval");
      if (approval.is_string()) {
        mcp.require_approval = approval.get<std::string>();
      } else if (approval.is_object()) {
        mcp.require_approval = parse_mcp_tool_approval_filter(approval);
      }
    }
    if (payload.contains("server_description") && payload.at("server_description").is_string()) {
      mcp.server_description = payload.at("server_description").get<std::string>();
    }
    if (payload.contains("server_url") && payload.at("server_url").is_string()) {
      mcp.server_url = payload.at("server_url").get<std::string>();
    }
    tool.mcp = std::move(mcp);
  }

  return tool;
}

ResponseToolChoice parse_tool_choice(const json& payload) {
  ResponseToolChoice choice;
  choice.raw = payload;
  if (payload.is_string()) {
    if (auto option = tool_choice_simple_from_string(payload.get<std::string>()); option) {
      choice.kind = ResponseToolChoice::Kind::Simple;
      choice.simple = *option;
    } else {
      choice.kind = ResponseToolChoice::Kind::Unknown;
    }
    return choice;
  }

  if (!payload.is_object()) {
    choice.kind = ResponseToolChoice::Kind::Unknown;
    return choice;
  }

  const std::string type = payload.value("type", "");
  if (type == "allowed_tools") {
    ResponseToolChoiceAllowed allowed;
    allowed.raw = payload;
    if (payload.contains("mode") && payload.at("mode").is_string()) {
      if (auto mode = tool_choice_allowed_mode_from_string(payload.at("mode").get<std::string>()); mode) {
        allowed.mode = *mode;
      }
    }
    if (payload.contains("tools") && payload.at("tools").is_array()) {
      for (const auto& definition : payload.at("tools")) {
        allowed.tools.push_back(parse_tool_definition(definition));
      }
    }
    choice.kind = ResponseToolChoice::Kind::Allowed;
    choice.allowed = std::move(allowed);
    return choice;
  }

  if (type == "function") {
    ResponseToolChoiceFunction function;
    function.raw = payload;
    function.name = payload.value("name", "");
    choice.kind = ResponseToolChoice::Kind::Function;
    choice.function = std::move(function);
    return choice;
  }

  if (type == "custom") {
    ResponseToolChoiceCustom custom;
    custom.raw = payload;
    custom.name = payload.value("name", "");
    choice.kind = ResponseToolChoice::Kind::Custom;
    choice.custom = std::move(custom);
    return choice;
  }

  if (type == "mcp") {
    ResponseToolChoiceMcp mcp;
    mcp.raw = payload;
    mcp.server_label = payload.value("server_label", "");
    if (payload.contains("name") && payload.at("name").is_string()) {
      mcp.name = payload.at("name").get<std::string>();
    }
    choice.kind = ResponseToolChoice::Kind::Mcp;
    choice.mcp = std::move(mcp);
    return choice;
  }

  if (!type.empty()) {
    ResponseToolChoiceTypes types;
    types.raw = payload;
    types.type = type;
    choice.kind = ResponseToolChoice::Kind::Types;
    choice.types = std::move(types);
    return choice;
  }

  choice.kind = ResponseToolChoice::Kind::Unknown;
  return choice;
}

nlohmann::json tool_choice_to_json(const ResponseToolChoice& choice) {
  switch (choice.kind) {
    case ResponseToolChoice::Kind::Simple:
      return to_string(choice.simple);
    case ResponseToolChoice::Kind::Allowed: {
      json data = choice.raw.is_object() ? choice.raw : json::object();
      data["type"] = "allowed_tools";
      data["mode"] = to_string(choice.allowed ? choice.allowed->mode : ResponseToolChoiceAllowedMode::Auto);
      json tools = json::array();
      if (choice.allowed) {
        for (const auto& tool : choice.allowed->tools) {
          tools.push_back(tool_definition_to_json(tool));
        }
      }
      data["tools"] = std::move(tools);
      return data;
    }
    case ResponseToolChoice::Kind::Function: {
      json data = choice.raw.is_object() ? choice.raw : json::object();
      data["type"] = "function";
      if (choice.function) {
        data["name"] = choice.function->name;
      }
      return data;
    }
    case ResponseToolChoice::Kind::Custom: {
      json data = choice.raw.is_object() ? choice.raw : json::object();
      data["type"] = "custom";
      if (choice.custom) {
        data["name"] = choice.custom->name;
      }
      return data;
    }
    case ResponseToolChoice::Kind::Mcp: {
      json data = choice.raw.is_object() ? choice.raw : json::object();
      data["type"] = "mcp";
      if (choice.mcp) {
        data["server_label"] = choice.mcp->server_label;
        if (choice.mcp->name) data["name"] = *choice.mcp->name;
      }
      return data;
    }
    case ResponseToolChoice::Kind::Types: {
      json data = choice.raw.is_object() ? choice.raw : json::object();
      if (choice.types) {
        data["type"] = choice.types->type;
      }
      return data;
    }
    case ResponseToolChoice::Kind::Unknown:
    default:
      return choice.raw;
  }
}

ResponseOutputTextAnnotation parse_output_text_annotation(const json& payload) {
  ResponseOutputTextAnnotation annotation;
  annotation.raw = payload;
  const std::string type = payload.value("type", "");
  if (type == "file_citation") {
    annotation.type = ResponseOutputTextAnnotation::Type::FileCitation;
    if (payload.contains("file_id") && payload.at("file_id").is_string()) {
      annotation.file_id = payload.at("file_id").get<std::string>();
    }
    if (payload.contains("filename") && payload.at("filename").is_string()) {
      annotation.filename = payload.at("filename").get<std::string>();
    }
    if (payload.contains("index") && payload.at("index").is_number_integer()) {
      annotation.index = payload.at("index").get<int>();
    }
  } else if (type == "url_citation") {
    annotation.type = ResponseOutputTextAnnotation::Type::UrlCitation;
    if (payload.contains("start_index") && payload.at("start_index").is_number_integer()) {
      annotation.start_index = payload.at("start_index").get<int>();
    }
    if (payload.contains("end_index") && payload.at("end_index").is_number_integer()) {
      annotation.end_index = payload.at("end_index").get<int>();
    }
    if (payload.contains("title") && payload.at("title").is_string()) {
      annotation.title = payload.at("title").get<std::string>();
    }
    if (payload.contains("url") && payload.at("url").is_string()) {
      annotation.url = payload.at("url").get<std::string>();
    }
  } else if (type == "container_file_citation") {
    annotation.type = ResponseOutputTextAnnotation::Type::ContainerFileCitation;
    if (payload.contains("container_id") && payload.at("container_id").is_string()) {
      annotation.container_id = payload.at("container_id").get<std::string>();
    }
    if (payload.contains("file_id") && payload.at("file_id").is_string()) {
      annotation.file_id = payload.at("file_id").get<std::string>();
    }
    if (payload.contains("filename") && payload.at("filename").is_string()) {
      annotation.filename = payload.at("filename").get<std::string>();
    }
    if (payload.contains("start_index") && payload.at("start_index").is_number_integer()) {
      annotation.start_index = payload.at("start_index").get<int>();
    }
    if (payload.contains("end_index") && payload.at("end_index").is_number_integer()) {
      annotation.end_index = payload.at("end_index").get<int>();
    }
  } else if (type == "file_path") {
    annotation.type = ResponseOutputTextAnnotation::Type::FilePath;
    if (payload.contains("file_id") && payload.at("file_id").is_string()) {
      annotation.file_id = payload.at("file_id").get<std::string>();
    }
    if (payload.contains("index") && payload.at("index").is_number_integer()) {
      annotation.index = payload.at("index").get<int>();
    }
  } else {
    annotation.type = ResponseOutputTextAnnotation::Type::Unknown;
  }
  return annotation;
}

ResponseOutputTextLogprobTop parse_output_text_logprob_top(const json& payload) {
  ResponseOutputTextLogprobTop top;
  top.raw = payload;
  top.token = payload.value("token", std::string{});
  if (payload.contains("bytes") && payload.at("bytes").is_array()) {
    for (const auto& value : payload.at("bytes")) {
      if (value.is_number_integer()) {
        const auto byte_value = value.get<int>();
        if (byte_value >= 0 && byte_value <= 255) {
          top.bytes.push_back(static_cast<std::uint8_t>(byte_value));
        }
      }
    }
  }
  if (payload.contains("logprob") && payload.at("logprob").is_number()) {
    top.logprob = payload.at("logprob").get<double>();
  }
  return top;
}

ResponseOutputTextLogprob parse_output_text_logprob(const json& payload) {
  ResponseOutputTextLogprob logprob;
  logprob.raw = payload;
  logprob.token = payload.value("token", std::string{});
  if (payload.contains("bytes") && payload.at("bytes").is_array()) {
    for (const auto& value : payload.at("bytes")) {
      if (value.is_number_integer()) {
        const auto byte_value = value.get<int>();
        if (byte_value >= 0 && byte_value <= 255) {
          logprob.bytes.push_back(static_cast<std::uint8_t>(byte_value));
        }
      }
    }
  }
  if (payload.contains("logprob") && payload.at("logprob").is_number()) {
    logprob.logprob = payload.at("logprob").get<double>();
  }
  if (payload.contains("top_logprobs") && payload.at("top_logprobs").is_array()) {
    for (const auto& entry : payload.at("top_logprobs")) {
      logprob.top_logprobs.push_back(parse_output_text_logprob_top(entry));
    }
  }
  return logprob;
}

ResponseOutputContent parse_output_content(const json& payload) {
  ResponseOutputContent content;
  content.raw = payload;
  const std::string type = payload.value("type", "");
  if (type == "output_text") {
    content.type = ResponseOutputContent::Type::Text;
    ResponseOutputTextSegment segment;
    segment.raw = payload;
    segment.text = payload.value("text", std::string{});
    if (payload.contains("annotations") && payload.at("annotations").is_array()) {
      for (const auto& annotation_json : payload.at("annotations")) {
        segment.annotations.push_back(parse_output_text_annotation(annotation_json));
      }
    }
    if (payload.contains("logprobs") && payload.at("logprobs").is_array()) {
      for (const auto& logprob_json : payload.at("logprobs")) {
        segment.logprobs.push_back(parse_output_text_logprob(logprob_json));
      }
    }
    content.text = std::move(segment);
  } else if (type == "refusal") {
    content.type = ResponseOutputContent::Type::Refusal;
    ResponseOutputRefusalSegment refusal;
    refusal.raw = payload;
    refusal.refusal = payload.value("refusal", std::string{});
    content.refusal = std::move(refusal);
  } else {
    content.type = ResponseOutputContent::Type::Raw;
  }
  return content;
}

ResponseFileSearchToolCallResult parse_file_search_result(const json& payload) {
  ResponseFileSearchToolCallResult result;
  result.raw = payload;
  if (payload.contains("attributes")) {
    result.attributes = payload.at("attributes");
  }
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
  call.id = payload.value("id", std::string{});
  call.status = payload.value("status", std::string{});
  if (payload.contains("queries") && payload.at("queries").is_array()) {
    for (const auto& q : payload.at("queries")) {
      if (q.is_string()) {
        call.queries.push_back(q.get<std::string>());
      }
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
  call.id = payload.value("id", std::string{});
  call.call_id = payload.value("call_id", std::string{});
  call.name = payload.value("name", std::string{});
  call.arguments = payload.value("arguments", std::string{});
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
  output.id = payload.value("id", std::string{});
  output.call_id = payload.value("call_id", std::string{});
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
          ResponseInputContent content;
          content.raw = item;
          const std::string type = item.value("type", std::string{});
          if (type == "input_text") {
            content.type = ResponseInputContent::Type::Text;
            content.text = item.value("text", std::string{});
          } else if (type == "input_image") {
            content.type = ResponseInputContent::Type::Image;
            content.image_url = item.value("image_url", std::string{});
            content.image_detail = item.value("detail", std::string{});
            content.file_id = item.value("file_id", std::string{});
          } else if (type == "input_file") {
            content.type = ResponseInputContent::Type::File;
            content.file_id = item.value("file_id", std::string{});
            content.file_url = item.value("file_url", std::string{});
            content.filename = item.value("filename", std::string{});
          } else if (type == "input_audio") {
            content.type = ResponseInputContent::Type::Audio;
            if (item.contains("input_audio") && item.at("input_audio").is_object()) {
              const auto& audio = item.at("input_audio");
              content.audio_data = audio.value("data", std::string{});
              content.audio_format = audio.value("format", std::string{});
            }
          } else {
            content.type = ResponseInputContent::Type::Raw;
          }
          output.structured_output.push_back(std::move(content));
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
  search.id = payload.value("id", std::string{});
  search.status = payload.value("status", std::string{});
  auto parse_action = [](const json& action_json) {
    ResponseFunctionWebSearch::Action action;
    action.raw = action_json;
    const std::string type = action_json.value("type", std::string{});
    if (type == "search") {
      action.type = ResponseFunctionWebSearch::Action::Type::Search;
      if (action_json.contains("query") && action_json.at("query").is_string()) {
        action.query = action_json.at("query").get<std::string>();
      }
      if (action_json.contains("sources") && action_json.at("sources").is_array()) {
        for (const auto& source_json : action_json.at("sources")) {
          ResponseFunctionWebSearch::Action::Source source;
          source.raw = source_json;
          source.url = source_json.value("url", std::string{});
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
  summary.text = payload.value("text", std::string{});
  summary.type = payload.value("type", std::string{});
  return summary;
}

ResponseReasoningContent parse_reasoning_content(const json& payload) {
  ResponseReasoningContent content;
  content.raw = payload;
  content.text = payload.value("text", std::string{});
  content.type = payload.value("type", std::string{});
  return content;
}

ResponseReasoningItemDetails parse_reasoning_item(const json& payload) {
  ResponseReasoningItemDetails reasoning;
  reasoning.raw = payload;
  reasoning.id = payload.value("id", std::string{});
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
  call.id = payload.value("id", std::string{});
  if (payload.contains("code")) {
    if (payload.at("code").is_string()) {
      call.code = payload.at("code").get<std::string>();
    } else if (payload.at("code").is_null()) {
      call.code = std::nullopt;
    }
  }
  call.container_id = payload.value("container_id", std::string{});
  if (payload.contains("outputs") && payload.at("outputs").is_array()) {
    for (const auto& output_json : payload.at("outputs")) {
      const auto output_type = output_json.value("type", std::string{});
      if (output_type == "logs") {
        ResponseCodeInterpreterLogOutput log;
        log.raw = output_json;
        log.logs = output_json.value("logs", std::string{});
        call.log_outputs.push_back(std::move(log));
      } else if (output_type == "image") {
        ResponseCodeInterpreterImageOutput image;
        image.raw = output_json;
        image.url = output_json.value("url", std::string{});
        call.image_outputs.push_back(std::move(image));
      }
    }
  }
  if (payload.contains("status") && payload.at("status").is_string()) {
    call.status = payload.at("status").get<std::string>();
  }
  return call;
}

ResponseImageGenerationCall parse_image_generation_call(const json& payload) {
  ResponseImageGenerationCall call;
  call.raw = payload;
  call.id = payload.value("id", std::string{});
  if (payload.contains("result") && !payload.at("result").is_null() && payload.at("result").is_string()) {
    call.result = payload.at("result").get<std::string>();
  }
  call.status = payload.value("status", std::string{});
  return call;
}

ResponseComputerToolCall parse_computer_tool_call(const json& payload) {
  ResponseComputerToolCall call;
  call.raw = payload;
  call.id = payload.value("id", std::string{});
  call.call_id = payload.value("call_id", std::string{});
  call.status = payload.value("status", std::string{});
  auto parse_action = [](const json& action_json) {
    ResponseComputerToolCall::Action action;
    action.raw = action_json;
    const std::string type = action_json.value("type", std::string{});
    if (type == "click") {
      action.type = ResponseComputerToolCall::Action::Type::Click;
      if (action_json.contains("button") && action_json.at("button").is_string()) {
        action.button = action_json.at("button").get<std::string>();
      }
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) {
        action.x = action_json.at("x").get<int>();
      }
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) {
        action.y = action_json.at("y").get<int>();
      }
    } else if (type == "double_click") {
      action.type = ResponseComputerToolCall::Action::Type::DoubleClick;
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) {
        action.x = action_json.at("x").get<int>();
      }
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) {
        action.y = action_json.at("y").get<int>();
      }
    } else if (type == "drag") {
      action.type = ResponseComputerToolCall::Action::Type::Drag;
      if (action_json.contains("path") && action_json.at("path").is_array()) {
        for (const auto& point : action_json.at("path")) {
          if (point.contains("x") && point.contains("y") && point.at("x").is_number_integer() &&
              point.at("y").is_number_integer()) {
            action.path.push_back({ point.at("x").get<int>(), point.at("y").get<int>() });
          }
        }
      }
    } else if (type == "keypress") {
      action.type = ResponseComputerToolCall::Action::Type::Keypress;
      if (action_json.contains("keys") && action_json.at("keys").is_array()) {
        for (const auto& key : action_json.at("keys")) {
          if (key.is_string()) {
            action.keys.push_back(key.get<std::string>());
          }
        }
      }
    } else if (type == "move") {
      action.type = ResponseComputerToolCall::Action::Type::Move;
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) {
        action.x = action_json.at("x").get<int>();
      }
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) {
        action.y = action_json.at("y").get<int>();
      }
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
      if (action_json.contains("x") && action_json.at("x").is_number_integer()) {
        action.x = action_json.at("x").get<int>();
      }
      if (action_json.contains("y") && action_json.at("y").is_number_integer()) {
        action.y = action_json.at("y").get<int>();
      }
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
      check.id = entry.value("id", std::string{});
      check.code = entry.value("code", std::string{});
      check.message = entry.value("message", std::string{});
      call.pending_safety_checks.push_back(std::move(check));
    }
  }
  return call;
}

ResponseLocalShellCall parse_local_shell_call(const json& payload) {
  ResponseLocalShellCall call;
  call.raw = payload;
  call.id = payload.value("id", std::string{});
  call.call_id = payload.value("call_id", std::string{});
  if (payload.contains("status") && payload.at("status").is_string()) {
    call.status = payload.at("status").get<std::string>();
  }
  if (payload.contains("action") && payload.at("action").is_object()) {
    const auto& action_json = payload.at("action");
    call.action.raw = action_json;
    const std::string type = action_json.value("type", std::string{});
    if (type == "exec") {
      call.action.type = ResponseLocalShellCall::Action::Type::Exec;
    } else {
      call.action.type = ResponseLocalShellCall::Action::Type::Unknown;
    }
    if (action_json.contains("command") && action_json.at("command").is_array()) {
      for (const auto& arg : action_json.at("command")) {
        if (arg.is_string()) {
          call.action.command.push_back(arg.get<std::string>());
        }
      }
    }
    if (action_json.contains("env") && action_json.at("env").is_object()) {
      for (auto it = action_json.at("env").begin(); it != action_json.at("env").end(); ++it) {
        if (it.value().is_string()) {
          call.action.env[it.key()] = it.value().get<std::string>();
        }
      }
    }
    if (action_json.contains("timeout_ms") && action_json.at("timeout_ms").is_number_integer()) {
      call.action.timeout_ms = action_json.at("timeout_ms").get<int>();
    }
    if (action_json.contains("user") && action_json.at("user").is_string()) {
      call.action.user = action_json.at("user").get<std::string>();
    }
    if (action_json.contains("working_directory") && action_json.at("working_directory").is_string()) {
      call.action.working_directory = action_json.at("working_directory").get<std::string>();
    }
  }
  return call;
}

ResponseLocalShellOutput parse_local_shell_output(const json& payload) {
  ResponseLocalShellOutput output;
  output.raw = payload;
  output.id = payload.value("id", std::string{});
  output.output = payload.value("output", std::string{});
  if (payload.contains("status") && payload.at("status").is_string()) {
    output.status = payload.at("status").get<std::string>();
  }
  if (!output.output.empty()) {
    try {
      output.parsed_output = json::parse(output.output);
    } catch (const json::exception&) {
      output.parsed_output = std::nullopt;
    }
  }
  return output;
}

ResponseComputerToolCallOutputScreenshot parse_computer_screenshot(const json& payload) {
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
  output.id = payload.value("id", std::string{});
  output.call_id = payload.value("call_id", std::string{});
  if (payload.contains("output") && payload.at("output").is_object()) {
    output.screenshot = parse_computer_screenshot(payload.at("output"));
  }
  if (payload.contains("acknowledged_safety_checks") && payload.at("acknowledged_safety_checks").is_array()) {
    for (const auto& check_json : payload.at("acknowledged_safety_checks")) {
      ResponseComputerToolCall::PendingSafetyCheck check;
      check.raw = check_json;
      check.id = check_json.value("id", std::string{});
      check.code = check_json.value("code", std::string{});
      check.message = check_json.value("message", std::string{});
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
  call.call_id = payload.value("call_id", std::string{});
  call.input = payload.value("input", std::string{});
  call.name = payload.value("name", std::string{});
  if (payload.contains("id") && payload.at("id").is_string()) {
    call.id = payload.at("id").get<std::string>();
  }
  return call;
}

ResponseOutputTextLogprob parse_text_delta_logprob(const json& payload) {
  ResponseOutputTextLogprob logprob;
  logprob.raw = payload;
  logprob.token = payload.value("token", std::string{});
  if (payload.contains("logprob") && payload.at("logprob").is_number()) {
    logprob.logprob = payload.at("logprob").get<double>();
  }
  if (payload.contains("top_logprobs") && payload.at("top_logprobs").is_array()) {
    for (const auto& entry : payload.at("top_logprobs")) {
      ResponseOutputTextLogprobTop top;
      top.raw = entry;
      top.token = entry.value("token", std::string{});
      if (entry.contains("logprob") && entry.at("logprob").is_number()) {
        top.logprob = entry.at("logprob").get<double>();
      }
      logprob.top_logprobs.push_back(std::move(top));
    }
  }
  return logprob;
}

ResponseStreamEvent parse_stream_event_payload(const json& payload, const std::optional<std::string>& event_name) {
  ResponseStreamEvent event;
  event.raw = payload;
  event.event_name = event_name;

  if (!payload.is_object()) {
    event.type = ResponseStreamEvent::Type::Unknown;
    return event;
  }

  const std::string type = payload.value("type", std::string{});
  if (type == "response.output_text.delta") {
    ResponseTextDeltaEvent delta;
    delta.raw = payload;
    delta.content_index = payload.value("content_index", 0);
    delta.delta = payload.value("delta", std::string{});
    delta.item_id = payload.value("item_id", std::string{});
    delta.output_index = payload.value("output_index", 0);
    delta.sequence_number = payload.value("sequence_number", 0);
    if (payload.contains("logprobs") && payload.at("logprobs").is_array()) {
      for (const auto& entry : payload.at("logprobs")) {
        delta.logprobs.push_back(parse_text_delta_logprob(entry));
      }
    }
    event.type = ResponseStreamEvent::Type::OutputTextDelta;
    event.text_delta = std::move(delta);
    return event;
  }

  if (type == "response.output_text.done") {
    ResponseTextDoneEvent done;
    done.raw = payload;
    done.content_index = payload.value("content_index", 0);
    done.item_id = payload.value("item_id", std::string{});
    done.text = payload.value("text", std::string{});
    done.output_index = payload.value("output_index", 0);
    done.sequence_number = payload.value("sequence_number", 0);
    event.type = ResponseStreamEvent::Type::OutputTextDone;
    event.text_done = std::move(done);
    return event;
  }

  if (type == "response.function_call_arguments.delta") {
    ResponseFunctionCallArgumentsDeltaEvent delta;
    delta.raw = payload;
    delta.delta = payload.value("delta", std::string{});
    delta.item_id = payload.value("item_id", std::string{});
    delta.output_index = payload.value("output_index", 0);
    delta.sequence_number = payload.value("sequence_number", 0);
    event.type = ResponseStreamEvent::Type::FunctionCallArgumentsDelta;
    event.function_arguments_delta = std::move(delta);
    return event;
  }

  if (type == "response.function_call_arguments.done") {
    ResponseFunctionCallArgumentsDoneEvent done;
    done.raw = payload;
    done.arguments = payload.value("arguments", std::string{});
    done.item_id = payload.value("item_id", std::string{});
    done.name = payload.value("name", std::string{});
    done.output_index = payload.value("output_index", 0);
    done.sequence_number = payload.value("sequence_number", 0);
    event.type = ResponseStreamEvent::Type::FunctionCallArgumentsDone;
    event.function_arguments_done = std::move(done);
    return event;
  }

  event.type = ResponseStreamEvent::Type::Unknown;
  return event;
}

std::optional<ResponseStreamEvent> parse_response_stream_event_internal(const ServerSentEvent& event) {
  try {
    auto payload = json::parse(event.data);
    return parse_stream_event_payload(payload, event.event);
  } catch (const json::exception&) {
    return std::nullopt;
  }
}

ResponseMcpCall parse_mcp_call(const json& payload) {
  ResponseMcpCall call;
  call.raw = payload;
  call.id = payload.value("id", std::string{});
  call.arguments = payload.value("arguments", std::string{});
  call.name = payload.value("name", std::string{});
  call.server_label = payload.value("server_label", std::string{});
  if (payload.contains("status") && payload.at("status").is_string()) {
    const std::string status = payload.at("status").get<std::string>();
    if (status == "in_progress") call.status = ResponseMcpCall::Status::InProgress;
    else if (status == "completed") call.status = ResponseMcpCall::Status::Completed;
    else if (status == "incomplete") call.status = ResponseMcpCall::Status::Incomplete;
    else if (status == "calling") call.status = ResponseMcpCall::Status::Calling;
    else if (status == "failed") call.status = ResponseMcpCall::Status::Failed;
    else call.status = ResponseMcpCall::Status::Unknown;
  }
  if (payload.contains("approval_request_id") && payload.at("approval_request_id").is_string()) {
    call.approval_request_id = payload.at("approval_request_id").get<std::string>();
  }
  if (payload.contains("error") && payload.at("error").is_string()) {
    call.error = payload.at("error").get<std::string>();
  }
  if (payload.contains("output") && payload.at("output").is_string()) {
    call.output = payload.at("output").get<std::string>();
  }
  return call;
}

ResponseMcpListToolsItem parse_mcp_list_tools_item(const json& payload) {
  ResponseMcpListToolsItem item;
  item.raw = payload;
  item.name = payload.value("name", std::string{});
  if (payload.contains("input_schema")) {
    item.input_schema = payload.at("input_schema");
  }
  if (payload.contains("description") && payload.at("description").is_string()) {
    item.description = payload.at("description").get<std::string>();
  }
  if (payload.contains("annotations") && payload.at("annotations").is_object()) {
    const auto& annotations = payload.at("annotations");
    if (annotations.contains("tags") && annotations.at("tags").is_array()) {
      for (const auto& tag : annotations.at("tags")) {
        if (tag.is_string()) {
          if (!item.tags) item.tags = std::vector<std::string>{};
          item.tags->push_back(tag.get<std::string>());
        }
      }
    }
  }
  return item;
}

ResponseMcpListTools parse_mcp_list_tools(const json& payload) {
  ResponseMcpListTools list;
  list.raw = payload;
  list.id = payload.value("id", std::string{});
  list.server_label = payload.value("server_label", std::string{});
  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
      list.tools.push_back(parse_mcp_list_tools_item(tool_json));
    }
  }
  if (payload.contains("error") && payload.at("error").is_string()) {
    list.error = payload.at("error").get<std::string>();
  }
  if (payload.contains("next_page_token") && payload.at("next_page_token").is_string()) {
    list.next_page_token = payload.at("next_page_token").get<std::string>();
  }
  return list;
}

ResponseMcpApprovalRequest parse_mcp_approval_request(const json& payload) {
  ResponseMcpApprovalRequest request;
  request.raw = payload;
  request.id = payload.value("id", std::string{});
  request.arguments = payload.value("arguments", std::string{});
  if (payload.contains("name") && payload.at("name").is_string()) {
    request.name = payload.at("name").get<std::string>();
  }
  if (payload.contains("server_label") && payload.at("server_label").is_string()) {
    request.server_label = payload.at("server_label").get<std::string>();
  }
  if (payload.contains("suggested_decision") && payload.at("suggested_decision").is_string()) {
    const auto decision = payload.at("suggested_decision").get<std::string>();
    if (decision == "pending") request.suggested_decision = ResponseMcpApprovalRequest::Decision::Pending;
    else if (decision == "approved") request.suggested_decision = ResponseMcpApprovalRequest::Decision::Approved;
    else if (decision == "rejected") request.suggested_decision = ResponseMcpApprovalRequest::Decision::Rejected;
    else request.suggested_decision = ResponseMcpApprovalRequest::Decision::Unknown;
  }
  return request;
}

ResponseMcpApprovalResponse parse_mcp_approval_response(const json& payload) {
  ResponseMcpApprovalResponse response;
  response.raw = payload;
  response.id = payload.value("id", std::string{});
  if (payload.contains("decision") && payload.at("decision").is_string()) {
    const auto decision = payload.at("decision").get<std::string>();
    if (decision == "approved") response.decision = ResponseMcpApprovalResponse::Decision::Approved;
    else if (decision == "rejected") response.decision = ResponseMcpApprovalResponse::Decision::Rejected;
    else response.decision = ResponseMcpApprovalResponse::Decision::Unknown;
  }
  if (payload.contains("reason") && payload.at("reason").is_string()) {
    response.reason = payload.at("reason").get<std::string>();
  }
  return response;
}

ResponseOutputMessage parse_output_message(const json& payload) {
  ResponseOutputMessage message;
  message.raw = payload;
  message.id = payload.value("id", std::string{});
  message.role = payload.value("role", std::string{"assistant"});
  if (payload.contains("status") && payload.at("status").is_string()) {
    message.status = payload.at("status").get<std::string>();
  }
  if (payload.contains("content") && payload.at("content").is_array()) {
    for (const auto& item : payload.at("content")) {
      auto content = parse_output_content(item);
      if (content.type == ResponseOutputContent::Type::Text && content.text) {
        message.text_segments.push_back(*content.text);
      }
      message.content.push_back(std::move(content));
    }
  }
  return message;
}

ResponseOutputItem parse_output_item(const json& payload) {
  ResponseOutputItem item;
  item.raw = payload;
  item.item_type = payload.value("type", std::string{});
  if (item.item_type == "message") {
    item.type = ResponseOutputItem::Type::Message;
    item.message = parse_output_message(payload);
    item.raw_details = item.message ? item.message->raw : json::object();
  } else if (item.item_type == "file_search_call") {
    item.type = ResponseOutputItem::Type::FileSearchToolCall;
    item.file_search_call = parse_file_search_call(payload);
    item.raw_details = item.file_search_call->raw;
  } else if (item.item_type == "function_call") {
    item.type = ResponseOutputItem::Type::FunctionToolCall;
    item.function_call = parse_function_tool_call(payload);
    item.raw_details = item.function_call->raw;
  } else if (item.item_type == "function_call_output") {
    item.type = ResponseOutputItem::Type::FunctionToolCallOutput;
    item.function_call_output = parse_function_tool_call_output(payload);
    item.raw_details = item.function_call_output->raw;
  } else if (item.item_type == "web_search_call") {
    item.type = ResponseOutputItem::Type::FunctionWebSearch;
    item.web_search_call = parse_function_web_search(payload);
    item.raw_details = item.web_search_call->raw;
  } else if (item.item_type == "computer_call") {
    item.type = ResponseOutputItem::Type::ComputerToolCall;
    item.computer_call = parse_computer_tool_call(payload);
    item.raw_details = item.computer_call->raw;
  } else if (item.item_type == "computer_call_output") {
    item.type = ResponseOutputItem::Type::ComputerToolCallOutput;
    item.computer_call_output = parse_computer_tool_call_output(payload);
    item.raw_details = item.computer_call_output->raw;
  } else if (item.item_type == "reasoning") {
    item.type = ResponseOutputItem::Type::Reasoning;
    item.reasoning = parse_reasoning_item(payload);
    item.raw_details = item.reasoning->raw;
  } else if (item.item_type == "image_generation_call") {
    item.type = ResponseOutputItem::Type::ImageGenerationCall;
    item.image_generation_call = parse_image_generation_call(payload);
    item.raw_details = item.image_generation_call->raw;
  } else if (item.item_type == "code_interpreter_call") {
    item.type = ResponseOutputItem::Type::CodeInterpreterToolCall;
    item.code_interpreter_call = parse_code_interpreter_tool_call(payload);
    item.raw_details = item.code_interpreter_call->raw;
  } else if (item.item_type == "local_shell_call") {
    item.type = ResponseOutputItem::Type::LocalShellCall;
    item.local_shell_call = parse_local_shell_call(payload);
    item.raw_details = item.local_shell_call->raw;
  } else if (item.item_type == "local_shell_call_output") {
    item.type = ResponseOutputItem::Type::LocalShellOutput;
    item.local_shell_output = parse_local_shell_output(payload);
    item.raw_details = item.local_shell_output->raw;
  } else if (item.item_type == "mcp_call") {
    item.type = ResponseOutputItem::Type::McpCall;
    item.mcp_call = parse_mcp_call(payload);
    item.raw_details = item.mcp_call->raw;
  } else if (item.item_type == "mcp_list_tools") {
    item.type = ResponseOutputItem::Type::McpListTools;
    item.mcp_list_tools = parse_mcp_list_tools(payload);
    item.raw_details = item.mcp_list_tools->raw;
  } else if (item.item_type == "mcp_approval_request") {
    item.type = ResponseOutputItem::Type::McpApprovalRequest;
    item.mcp_approval_request = parse_mcp_approval_request(payload);
    item.raw_details = item.mcp_approval_request->raw;
  } else if (item.item_type == "mcp_approval_response") {
    item.type = ResponseOutputItem::Type::McpApprovalResponse;
    item.mcp_approval_response = parse_mcp_approval_response(payload);
    item.raw_details = item.mcp_approval_response->raw;
  } else if (item.item_type == "custom_tool_call") {
    item.type = ResponseOutputItem::Type::CustomToolCall;
    item.custom_tool_call = parse_custom_tool_call(payload);
    item.raw_details = item.custom_tool_call->raw;
  } else {
    item.type = ResponseOutputItem::Type::Raw;
    if (payload.is_object()) {
      item.raw_details = payload;
    }
  }
  return item;
}

void ensure_model_present(const json& body) {
  if (!body.contains("model") || body.at("model").is_null()) {
    throw OpenAIError("ResponsesRequest body must include a model");
  }
}

json build_request_body(const ResponseRequest& request) {
  json body = json::object();
  body["model"] = request.model;

  json input = json::array();
  for (const auto& item : request.input) {
    json serialized_item;
    switch (item.type) {
      case ResponseInputItem::Type::Message: {
        serialized_item = item.message.raw.is_object() ? item.message.raw : json::object();
        serialized_item["type"] = "message";
        serialized_item["role"] = item.message.role;
        if (item.message.id) serialized_item["id"] = *item.message.id;
        if (item.message.status) serialized_item["status"] = *item.message.status;
        if (!item.message.metadata.empty()) serialized_item["metadata"] = item.message.metadata;

        json content = json::array();
        for (const auto& piece : item.message.content) {
          json content_item = piece.raw.is_object() ? piece.raw : json::object();
          if (piece.id) content_item["id"] = *piece.id;
          switch (piece.type) {
            case ResponseInputContent::Type::Text:
              content_item["type"] = "input_text";
              content_item["text"] = piece.text;
              break;
            case ResponseInputContent::Type::Image:
              content_item["type"] = "input_image";
              if (!piece.image_url.empty()) content_item["image_url"] = piece.image_url;
              if (!piece.image_detail.empty()) content_item["detail"] = piece.image_detail;
              if (!piece.file_id.empty()) content_item["file_id"] = piece.file_id;
              break;
            case ResponseInputContent::Type::File:
              content_item["type"] = "input_file";
              if (!piece.file_id.empty()) content_item["file_id"] = piece.file_id;
              if (!piece.file_url.empty()) content_item["file_url"] = piece.file_url;
              if (!piece.filename.empty()) content_item["filename"] = piece.filename;
              break;
            case ResponseInputContent::Type::Audio:
              content_item["type"] = "input_audio";
              if (!piece.audio_data.empty()) {
                content_item["input_audio"] = { {"data", piece.audio_data}, {"format", piece.audio_format} };
              }
              break;
            case ResponseInputContent::Type::Raw:
              if (!piece.raw.is_null()) {
                content_item = piece.raw;
              }
              break;
          }
          content.push_back(std::move(content_item));
        }
        serialized_item["content"] = std::move(content);
        break;
      }
      case ResponseInputItem::Type::InputText: {
        serialized_item = item.raw.is_object() ? item.raw : json::object();
        serialized_item["type"] = "input_text";
        if (item.input_text) {
          serialized_item["text"] = item.input_text->text;
        }
        break;
      }
      case ResponseInputItem::Type::InputImage: {
        serialized_item = item.raw.is_object() ? item.raw : json::object();
        serialized_item["type"] = "input_image";
        if (item.input_image) {
          if (item.input_image->image_url) serialized_item["image_url"] = *item.input_image->image_url;
          if (item.input_image->file_id) serialized_item["file_id"] = *item.input_image->file_id;
          if (item.input_image->detail) serialized_item["detail"] = *item.input_image->detail;
        }
        break;
      }
      case ResponseInputItem::Type::InputFile: {
        serialized_item = item.raw.is_object() ? item.raw : json::object();
        serialized_item["type"] = "input_file";
        if (item.input_file) {
          if (item.input_file->file_data) serialized_item["file_data"] = *item.input_file->file_data;
          if (item.input_file->file_id) serialized_item["file_id"] = *item.input_file->file_id;
          if (item.input_file->file_url) serialized_item["file_url"] = *item.input_file->file_url;
          if (item.input_file->filename) serialized_item["filename"] = *item.input_file->filename;
        }
        break;
      }
      case ResponseInputItem::Type::InputAudio: {
        serialized_item = item.raw.is_object() ? item.raw : json::object();
        serialized_item["type"] = "input_audio";
        if (item.input_audio) {
          json input_audio = json::object();
          input_audio["data"] = item.input_audio->data;
          input_audio["format"] = item.input_audio->format;
          serialized_item["input_audio"] = std::move(input_audio);
        }
        break;
      }
      case ResponseInputItem::Type::Raw: {
        serialized_item = item.raw;
        break;
      }
    }
    input.push_back(std::move(serialized_item));
  }
  body["input"] = std::move(input);

  if (request.background) body["background"] = *request.background;
  if (request.conversation_id) body["conversation"] = *request.conversation_id;
  if (!request.include.empty()) body["include"] = request.include;
  if (request.instructions) body["instructions"] = *request.instructions;
  if (request.max_output_tokens) body["max_output_tokens"] = *request.max_output_tokens;
  if (request.parallel_tool_calls) body["parallel_tool_calls"] = *request.parallel_tool_calls;
  if (request.previous_response_id) body["previous_response_id"] = *request.previous_response_id;
  if (request.prompt) {
    json prompt;
    prompt["id"] = request.prompt->id;
    if (!request.prompt->variables.empty()) prompt["variables"] = request.prompt->variables;
    for (auto it = request.prompt->extra.begin(); it != request.prompt->extra.end(); ++it) {
      prompt[it.key()] = it.value();
    }
    body["prompt"] = std::move(prompt);
  }
  if (request.prompt_cache_key) body["prompt_cache_key"] = *request.prompt_cache_key;
  if (request.reasoning) {
    json reasoning;
    if (request.reasoning->effort) reasoning["effort"] = *request.reasoning->effort;
    for (auto it = request.reasoning->extra.begin(); it != request.reasoning->extra.end(); ++it) {
      reasoning[it.key()] = it.value();
    }
    body["reasoning"] = std::move(reasoning);
  }
  if (request.safety_identifier) body["safety_identifier"] = *request.safety_identifier;
  if (request.service_tier) body["service_tier"] = *request.service_tier;
  if (request.store) body["store"] = *request.store;
  if (request.stream) body["stream"] = *request.stream;
  if (request.stream_options) {
    json stream_options;
    if (request.stream_options->include_usage) stream_options["include_usage"] = *request.stream_options->include_usage;
    for (auto it = request.stream_options->extra.begin(); it != request.stream_options->extra.end(); ++it) {
      stream_options[it.key()] = it.value();
    }
    body["stream_options"] = std::move(stream_options);
  }
  if (request.temperature) body["temperature"] = *request.temperature;
  if (request.top_p) body["top_p"] = *request.top_p;
  if (!request.tools.empty()) {
    json tools = json::array();
    for (const auto& tool : request.tools) {
      tools.push_back(tool_definition_to_json(tool));
    }
    body["tools"] = std::move(tools);
  }
  if (request.tool_choice) {
    body["tool_choice"] = tool_choice_to_json(*request.tool_choice);
  }

  ensure_model_present(body);
  return body;
}

std::string join_messages(const std::vector<ResponseOutputMessage>& messages) {
  std::ostringstream stream;
  for (const auto& message : messages) {
    for (const auto& segment : message.text_segments) {
      stream << segment.text;
    }
  }
  return stream.str();
}

ResponseUsage parse_usage(const json& payload) {
  ResponseUsage usage;
  usage.input_tokens = payload.value("input_tokens", 0);
  usage.output_tokens = payload.value("output_tokens", 0);
  usage.total_tokens = payload.value("total_tokens", 0);
  if (payload.contains("input_tokens_details") && payload.at("input_tokens_details").is_object()) {
    ResponseUsageInputTokensDetails details;
    details.raw = payload.at("input_tokens_details");
    if (details.raw.contains("cached_tokens") && details.raw.at("cached_tokens").is_number_integer()) {
      details.cached_tokens = details.raw.at("cached_tokens").get<int>();
    }
    usage.input_tokens_details = std::move(details);
  }
  if (payload.contains("output_tokens_details") && payload.at("output_tokens_details").is_object()) {
    ResponseUsageOutputTokensDetails details;
    details.raw = payload.at("output_tokens_details");
    if (details.raw.contains("reasoning_tokens") && details.raw.at("reasoning_tokens").is_number_integer()) {
      details.reasoning_tokens = details.raw.at("reasoning_tokens").get<int>();
    }
    usage.output_tokens_details = std::move(details);
  }
  usage.extra = payload;
  return usage;
}

Response parse_response(const json& payload) {
  Response response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.object = payload.value("object", "");
  response.created = payload.value("created", 0);
  response.model = payload.value("model", "");

  if (payload.contains("metadata") && payload.at("metadata").is_object()) {
    for (auto it = payload.at("metadata").begin(); it != payload.at("metadata").end(); ++it) {
      if (it.value().is_string()) {
        response.metadata[it.key()] = it.value().get<std::string>();
      }
    }
  }

  if (payload.contains("background") && payload.at("background").is_boolean()) {
    response.background = payload.at("background").get<bool>();
  }
  if (payload.contains("max_output_tokens") && payload.at("max_output_tokens").is_number_integer()) {
    response.max_output_tokens = payload.at("max_output_tokens").get<int>();
  }
  if (payload.contains("previous_response_id") && payload.at("previous_response_id").is_string()) {
    response.previous_response_id = payload.at("previous_response_id").get<std::string>();
  }
  if (payload.contains("temperature") && payload.at("temperature").is_number()) {
    response.temperature = payload.at("temperature").get<double>();
  }
  if (payload.contains("top_p") && payload.at("top_p").is_number()) {
    response.top_p = payload.at("top_p").get<double>();
  }
  if (payload.contains("parallel_tool_calls") && payload.at("parallel_tool_calls").is_boolean()) {
    response.parallel_tool_calls = payload.at("parallel_tool_calls").get<bool>();
  }

  if (payload.contains("conversation") && payload.at("conversation").is_object()) {
    ResponseConversationRef conversation;
    conversation.raw = payload.at("conversation");
    conversation.id = conversation.raw.value("id", std::string{});
    response.conversation = std::move(conversation);
  }

  if (payload.contains("error") && payload.at("error").is_object()) {
    ResponseError error;
    error.raw = payload.at("error");
    if (error.raw.contains("code") && !error.raw.at("code").is_null() && error.raw.at("code").is_string()) {
      error.code = error.raw.at("code").get<std::string>();
    }
    error.message = error.raw.value("message", std::string{});
    response.error = std::move(error);
  }

  if (payload.contains("incomplete_details") && payload.at("incomplete_details").is_object()) {
    ResponseIncompleteDetails details;
    details.raw = payload.at("incomplete_details");
    if (details.raw.contains("reason") && details.raw.at("reason").is_string()) {
      details.reason = details.raw.at("reason").get<std::string>();
    }
    response.incomplete_details = std::move(details);
  }

  if (payload.contains("tools") && payload.at("tools").is_array()) {
    for (const auto& tool_json : payload.at("tools")) {
      response.tools.push_back(parse_tool_definition(tool_json));
    }
  }

  if (payload.contains("tool_choice")) {
    response.tool_choice = parse_tool_choice(payload.at("tool_choice"));
  }

  if (payload.contains("output") && payload.at("output").is_array()) {
    for (const auto& item_json : payload.at("output")) {
      auto output_item = parse_output_item(item_json);
      if (output_item.type == ResponseOutputItem::Type::Message && output_item.message) {
        response.messages.push_back(*output_item.message);
      }
      response.output.push_back(std::move(output_item));
    }
  }
  response.output_text = join_messages(response.messages);

  if (payload.contains("usage")) {
    response.usage = parse_usage(payload.at("usage"));
  }

  return response;
}

ResponseList parse_response_list(const json& payload) {
  ResponseList list;
  list.raw = payload;
  if (payload.contains("data")) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_response(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.last_id = payload.at("last_id").get<std::string>();
  }
  return list;
}

json build_retrieve_query(const ResponseRetrieveOptions& options) {
  json query = json::object();
  query["stream"] = options.stream;
  return query;
}

std::string build_response_path(const std::string& response_id) {
  return std::string(kResponseEndpoint) + "/" + response_id;
}

}  // namespace

CursorPage<Response> ResponsesResource::list_page(const RequestOptions& options) const {
  auto fetch_impl = std::make_shared<std::function<CursorPage<Response>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<Response> {
    RequestOptions next_options = to_request_options(request_options);
    auto response = client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    try {
      auto payload = json::parse(response.body);
      auto list = parse_response_list(payload);
      std::optional<std::string> cursor = list.last_id;
      if (!cursor && !list.data.empty()) {
        cursor = list.data.back().id;
      }

      return CursorPage<Response>(
          std::move(list.data),
          list.has_more,
          std::move(cursor),
          request_options,
          *fetch_impl,
          "after",
          std::move(list.raw));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse responses list: ") + ex.what());
    }
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kResponseEndpoint;
  initial.headers = materialize_headers(options);
  initial.query = materialize_query(options);

  return (*fetch_impl)(initial);
}

CursorPage<Response> ResponsesResource::list_page() const {
  return list_page(RequestOptions{});
}

std::optional<ResponseStreamEvent> parse_response_stream_event(const ServerSentEvent& event) {
  return parse_response_stream_event_internal(event);
}

Response ResponsesResource::create(const ResponseRequest& request, const RequestOptions& options) const {
  auto body = build_request_body(request);
  auto response = client_.perform_request("POST", kResponseEndpoint, body.dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse response body: ") + ex.what());
  }
}

Response ResponsesResource::create(const ResponseRequest& request) const {
  return create(request, RequestOptions{});
}

Response ResponsesResource::retrieve(const std::string& response_id,
                                     const ResponseRetrieveOptions& retrieve_options,
                                     const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.query_params["stream"] = retrieve_options.stream ? "true" : "false";

  auto response = client_.perform_request("GET", build_response_path(response_id), "", request_options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse response retrieval: ") + ex.what());
  }
}

Response ResponsesResource::retrieve(const std::string& response_id) const {
  return retrieve(response_id, ResponseRetrieveOptions{}, RequestOptions{});
}

void ResponsesResource::remove(const std::string& response_id, const RequestOptions& options) const {
  auto response = client_.perform_request("DELETE", build_response_path(response_id), "", options);
  (void)response;
}

void ResponsesResource::remove(const std::string& response_id) const {
  remove(response_id, RequestOptions{});
}

Response ResponsesResource::cancel(const std::string& response_id, const RequestOptions& options) const {
  auto path = build_response_path(response_id) + "/cancel";
  auto response = client_.perform_request("POST", path, json::object().dump(), options);
  try {
    auto payload = json::parse(response.body);
    return parse_response(payload);
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse cancel response: ") + ex.what());
  }
}

Response ResponsesResource::cancel(const std::string& response_id) const {
  return cancel(response_id, RequestOptions{});
}

ResponseList ResponsesResource::list(const RequestOptions& options) const {
  auto page = list_page(options);
  ResponseList list;
  list.data = page.data();
  list.has_more = page.has_next_page();
  list.last_id = page.next_cursor();
  list.raw = page.raw();
  return list;
}

ResponseList ResponsesResource::list() const {
  return list(RequestOptions{});
}

std::vector<ServerSentEvent> ResponsesResource::create_stream(const ResponseRequest& request,
                                                              const RequestOptions& options) const {
  SSEEventStream stream;

  auto body = build_request_body(request);
  body["stream"] = true;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kResponseEndpoint, body.dump(), request_options);

  stream.finalize();
  return stream.events();
}

std::vector<ServerSentEvent> ResponsesResource::create_stream(const ResponseRequest& request) const {
  return create_stream(request, RequestOptions{});
}

void ResponsesResource::create_stream(const ResponseRequest& request,
                                      const std::function<bool(const ResponseStreamEvent&)>& on_event,
                                      const RequestOptions& options) const {
  SSEEventStream stream([&](const ServerSentEvent& sse_event) {
    if (!on_event) {
      return true;
    }
    if (auto parsed = parse_response_stream_event(sse_event)) {
      return on_event(*parsed);
    }
    return true;
  });

  auto body = build_request_body(request);
  body["stream"] = true;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("POST", kResponseEndpoint, body.dump(), request_options);

  stream.finalize();
}

void ResponsesResource::create_stream(const ResponseRequest& request,
                                      const std::function<bool(const ResponseStreamEvent&)>& on_event) const {
  create_stream(request, on_event, RequestOptions{});
}

std::vector<ServerSentEvent> ResponsesResource::retrieve_stream(const std::string& response_id,
                                                                const ResponseRetrieveOptions& retrieve_options,
                                                                const RequestOptions& options) const {
  SSEEventStream stream;

  RequestOptions request_options = options;
  request_options.headers["Accept"] = "text/event-stream";
  request_options.collect_body = false;
  request_options.query_params["stream"] = retrieve_options.stream ? "true" : "false";
  request_options.on_chunk = [&](const char* data, std::size_t size) { stream.feed(data, size); };

  client_.perform_request("GET", build_response_path(response_id), "", request_options);

  stream.finalize();
  return stream.events();
}

std::vector<ServerSentEvent> ResponsesResource::retrieve_stream(const std::string& response_id) const {
  return retrieve_stream(response_id, ResponseRetrieveOptions{.stream = true}, RequestOptions{});
}

}  // namespace openai

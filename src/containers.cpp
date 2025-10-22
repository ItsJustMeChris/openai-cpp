#include "openai/containers.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"
#include "openai/utils/multipart.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
#include <sstream>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kContainersPath = "/containers";

std::string filename_from_path(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

std::vector<std::uint8_t> read_file_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::optional<ContainerExpiresAfter> parse_expires_after(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  ContainerExpiresAfter expires;
  if (payload.contains("anchor") && payload.at("anchor").is_string()) {
    expires.anchor = payload.at("anchor").get<std::string>();
  }
  if (payload.contains("minutes") && payload.at("minutes").is_number_integer()) {
    expires.minutes = payload.at("minutes").get<int>();
  }
  return expires;
}

Container parse_container(const json& payload) {
  Container container;
  container.raw = payload;
  container.id = payload.value("id", "");
  container.created_at = payload.value("created_at", 0);
  container.name = payload.value("name", "");
  container.object = payload.value("object", "");
  container.status = payload.value("status", "");
  if (payload.contains("expires_after")) {
    container.expires_after = parse_expires_after(payload.at("expires_after"));
  }
  return container;
}

ContainerList parse_container_list(const json& payload) {
  ContainerList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_container(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

ContainerFile parse_container_file(const json& payload) {
  ContainerFile file;
  file.raw = payload;
  file.id = payload.value("id", "");
  file.bytes = payload.value("bytes", 0);
  file.container_id = payload.value("container_id", "");
  file.created_at = payload.value("created_at", 0);
  file.object = payload.value("object", "");
  file.path = payload.value("path", "");
  file.source = payload.value("source", "");
  return file;
}

ContainerFileList parse_container_file_list(const json& payload) {
  ContainerFileList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_container_file(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("next_cursor") && payload.at("next_cursor").is_string()) {
    list.next_cursor = payload.at("next_cursor").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

json container_create_request_to_json(const ContainerCreateRequest& request) {
  json body = json::object();
  body["name"] = request.name;
  if (request.expires_after) {
    json expires = json::object();
    expires["anchor"] = request.expires_after->anchor;
    expires["minutes"] = request.expires_after->minutes;
    body["expires_after"] = std::move(expires);
  }
  if (!request.file_ids.empty()) {
    body["file_ids"] = request.file_ids;
  }
  return body;
}

json file_create_json_body(const ContainerFileCreateRequest& request) {
  json body = json::object();
  body["file_id"] = *request.file_id;
  return body;
}

}  // namespace

Container ContainersResource::create(const ContainerCreateRequest& request) const {
  return create(request, RequestOptions{});
}

Container ContainersResource::create(const ContainerCreateRequest& request,
                                     const RequestOptions& options) const {
  auto response =
      client_.perform_request("POST", kContainersPath, container_create_request_to_json(request).dump(), options);
  try {
    return parse_container(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container create response: ") + ex.what());
  }
}

Container ContainersResource::retrieve(const std::string& container_id) const {
  return retrieve(container_id, RequestOptions{});
}

Container ContainersResource::retrieve(const std::string& container_id, const RequestOptions& options) const {
  auto path = std::string(kContainersPath) + "/" + container_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_container(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container retrieve response: ") + ex.what());
  }
}

ContainerList ContainersResource::list() const {
  return list(ContainerListParams{}, RequestOptions{});
}

ContainerList ContainersResource::list(const ContainerListParams& params) const {
  return list(params, RequestOptions{});
}

ContainerList ContainersResource::list(const ContainerListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  auto response = client_.perform_request("GET", kContainersPath, "", request_options);
  try {
    return parse_container_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container list response: ") + ex.what());
  }
}

CursorPage<Container> ContainersResource::list_page() const {
  return list_page(ContainerListParams{}, RequestOptions{});
}

CursorPage<Container> ContainersResource::list_page(const ContainerListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<Container> ContainersResource::list_page(const ContainerListParams& params,
                                                    const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  auto fetch_impl = std::make_shared<std::function<CursorPage<Container>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<Container> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    ContainerList list;
    try {
      list = parse_container_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse container list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<Container>(std::move(list.data),
                                 list.has_more,
                                 std::move(cursor),
                                 request_options,
                                 *fetch_impl,
                                 "after",
                                 std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kContainersPath;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

void ContainersResource::remove(const std::string& container_id) const {
  remove(container_id, RequestOptions{});
}

void ContainersResource::remove(const std::string& container_id, const RequestOptions& options) const {
  auto path = std::string(kContainersPath) + "/" + container_id;
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "*/*";
  client_.perform_request("DELETE", path, "", request_options);
}

ContainerFile ContainerFilesResource::create(const std::string& container_id,
                                             const ContainerFileCreateRequest& request) const {
  return create(container_id, request, RequestOptions{});
}

ContainerFile ContainerFilesResource::create(const std::string& container_id,
                                             const ContainerFileCreateRequest& request,
                                             const RequestOptions& options) const {
  const std::string path = std::string(kContainersPath) + "/" + container_id + "/files";
  RequestOptions request_options = options;

  std::string body;
  if (request.file_id) {
    body = file_create_json_body(request).dump();
  } else {
    std::vector<std::uint8_t> data;
    std::string filename;
    std::string content_type = request.content_type.value_or("application/octet-stream");

    if (request.file_data) {
      data = *request.file_data;
      if (!request.file_name) {
        throw OpenAIError("ContainerFileCreateRequest.file_name must be provided when file_data is set");
      }
      filename = *request.file_name;
    } else if (request.file_path) {
      data = read_file_binary(*request.file_path);
      filename = request.file_name.value_or(filename_from_path(*request.file_path));
    } else {
      throw OpenAIError("ContainerFileCreateRequest must provide either file_id, file_path, or file_data");
    }

    utils::MultipartFormData form;
    form.append_file("file", filename, content_type, data);
    auto encoded = form.build();
    request_options.headers["Content-Type"] = encoded.content_type;
    body.assign(encoded.body.begin(), encoded.body.end());
  }

  auto response = client_.perform_request("POST", path, body, request_options);
  try {
    return parse_container_file(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container file create response: ") + ex.what());
  }
}

ContainerFile ContainerFilesResource::retrieve(const std::string& container_id,
                                               const std::string& file_id) const {
  return retrieve(container_id, file_id, RequestOptions{});
}

ContainerFile ContainerFilesResource::retrieve(const std::string& container_id,
                                               const std::string& file_id,
                                               const RequestOptions& options) const {
  const std::string path = std::string(kContainersPath) + "/" + container_id + "/files/" + file_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_container_file(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container file retrieve response: ") + ex.what());
  }
}

ContainerFileList ContainerFilesResource::list(const std::string& container_id) const {
  return list(container_id, ContainerFileListParams{}, RequestOptions{});
}

ContainerFileList ContainerFilesResource::list(const std::string& container_id,
                                               const ContainerFileListParams& params) const {
  return list(container_id, params, RequestOptions{});
}

ContainerFileList ContainerFilesResource::list(const std::string& container_id,
                                               const ContainerFileListParams& params,
                                               const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  const std::string path = std::string(kContainersPath) + "/" + container_id + "/files";
  auto response = client_.perform_request("GET", path, "", request_options);
  try {
    return parse_container_file_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse container file list response: ") + ex.what());
  }
}

CursorPage<ContainerFile> ContainerFilesResource::list_page(const std::string& container_id) const {
  return list_page(container_id, ContainerFileListParams{}, RequestOptions{});
}

CursorPage<ContainerFile> ContainerFilesResource::list_page(const std::string& container_id,
                                                            const ContainerFileListParams& params) const {
  return list_page(container_id, params, RequestOptions{});
}

CursorPage<ContainerFile> ContainerFilesResource::list_page(const std::string& container_id,
                                                            const ContainerFileListParams& params,
                                                            const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  const std::string base_path = std::string(kContainersPath) + "/" + container_id + "/files";

  auto fetch_impl = std::make_shared<std::function<CursorPage<ContainerFile>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<ContainerFile> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    ContainerFileList list;
    try {
      list = parse_container_file_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse container file list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<ContainerFile>(std::move(list.data),
                                     list.has_more,
                                     std::move(cursor),
                                     request_options,
                                     *fetch_impl,
                                     "after",
                                     std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = base_path;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

void ContainerFilesResource::remove(const std::string& container_id,
                                    const std::string& file_id) const {
  remove(container_id, file_id, RequestOptions{});
}

void ContainerFilesResource::remove(const std::string& container_id,
                                    const std::string& file_id,
                                    const RequestOptions& options) const {
  const std::string path = std::string(kContainersPath) + "/" + container_id + "/files/" + file_id;
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "*/*";
  client_.perform_request("DELETE", path, "", request_options);
}

ContainerFileContent ContainerFilesContentResource::retrieve(const std::string& container_id,
                                                             const std::string& file_id) const {
  return retrieve(container_id, file_id, RequestOptions{});
}

ContainerFileContent ContainerFilesContentResource::retrieve(const std::string& container_id,
                                                             const std::string& file_id,
                                                             const RequestOptions& options) const {
  const std::string path = std::string(kContainersPath) + "/" + container_id + "/files/" + file_id + "/content";
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "application/octet-stream";
  auto response = client_.perform_request("GET", path, "", request_options);
  ContainerFileContent content;
  content.headers = response.headers;
  content.data.assign(response.body.begin(), response.body.end());
  return content;
}

}  // namespace openai

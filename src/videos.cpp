#include "openai/videos.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/pagination.hpp"
#include "openai/utils/multipart.hpp"

#include <nlohmann/json.hpp>

#include <memory>

namespace openai {
namespace {

using json = nlohmann::json;

constexpr const char* kVideosPath = "/videos";

std::string filename_from_path(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

std::string model_to_string(VideoModel model) {
  switch (model) {
    case VideoModel::Sora2:
      return "sora-2";
    case VideoModel::Sora2Pro:
      return "sora-2-pro";
  }
  return "sora-2";
}

VideoModel parse_model(const json& payload) {
  const auto& value = payload.get<std::string>();
  if (value == "sora-2-pro") {
    return VideoModel::Sora2Pro;
  }
  return VideoModel::Sora2;
}

std::string seconds_to_string(VideoSeconds seconds) {
  switch (seconds) {
    case VideoSeconds::Four:
      return "4";
    case VideoSeconds::Eight:
      return "8";
    case VideoSeconds::Twelve:
      return "12";
  }
  return "4";
}

VideoSeconds parse_seconds(const json& payload) {
  const auto& value = payload.get<std::string>();
  if (value == "8") {
    return VideoSeconds::Eight;
  }
  if (value == "12") {
    return VideoSeconds::Twelve;
  }
  return VideoSeconds::Four;
}

std::string size_to_string(VideoSize size) {
  switch (size) {
    case VideoSize::Size720x1280:
      return "720x1280";
    case VideoSize::Size1280x720:
      return "1280x720";
    case VideoSize::Size1024x1792:
      return "1024x1792";
    case VideoSize::Size1792x1024:
      return "1792x1024";
  }
  return "720x1280";
}

VideoSize parse_size(const json& payload) {
  const auto& value = payload.get<std::string>();
  if (value == "1280x720") {
    return VideoSize::Size1280x720;
  }
  if (value == "1024x1792") {
    return VideoSize::Size1024x1792;
  }
  if (value == "1792x1024") {
    return VideoSize::Size1792x1024;
  }
  return VideoSize::Size720x1280;
}

std::string variant_to_string(VideoDownloadVariant variant) {
  switch (variant) {
    case VideoDownloadVariant::Video:
      return "video";
    case VideoDownloadVariant::Thumbnail:
      return "thumbnail";
    case VideoDownloadVariant::SpriteSheet:
      return "spritesheet";
  }
  return "video";
}

std::optional<VideoCreateError> parse_error(const json& payload) {
  if (!payload.is_object()) {
    return std::nullopt;
  }
  VideoCreateError error;
  error.code = payload.value("code", "");
  error.message = payload.value("message", "");
  return error;
}

Video parse_video(const json& payload) {
  Video video;
  video.raw = payload;
  video.id = payload.value("id", "");
  if (payload.contains("completed_at") && !payload.at("completed_at").is_null()) {
    video.completed_at = payload.at("completed_at").get<int>();
  }
  video.created_at = payload.value("created_at", 0);
  if (payload.contains("error") && !payload.at("error").is_null()) {
    video.error = parse_error(payload.at("error"));
  }
  if (payload.contains("expires_at") && !payload.at("expires_at").is_null()) {
    video.expires_at = payload.at("expires_at").get<int>();
  }
  if (payload.contains("model") && payload.at("model").is_string()) {
    video.model = parse_model(payload.at("model"));
  }
  video.object = payload.value("object", "");
  video.progress = payload.value("progress", 0);
  if (payload.contains("remixed_from_video_id") && !payload.at("remixed_from_video_id").is_null()) {
    video.remixed_from_video_id = payload.at("remixed_from_video_id").get<std::string>();
  }
  if (payload.contains("seconds") && payload.at("seconds").is_string()) {
    video.seconds = parse_seconds(payload.at("seconds"));
  }
  if (payload.contains("size") && payload.at("size").is_string()) {
    video.size = parse_size(payload.at("size"));
  }
  video.status = payload.value("status", "");
  return video;
}

VideoList parse_video_list(const json& payload) {
  VideoList list;
  list.raw = payload;
  if (payload.contains("data") && payload.at("data").is_array()) {
    for (const auto& item : payload.at("data")) {
      list.data.push_back(parse_video(item));
    }
  }
  list.has_more = payload.value("has_more", false);
  if (payload.contains("last_id") && payload.at("last_id").is_string()) {
    list.next_cursor = payload.at("last_id").get<std::string>();
  } else if (!list.data.empty()) {
    list.next_cursor = list.data.back().id;
  }
  return list;
}

VideoDeleteResponse parse_video_delete(const json& payload) {
  VideoDeleteResponse response;
  response.raw = payload;
  response.id = payload.value("id", "");
  response.deleted = payload.value("deleted", false);
  response.object = payload.value("object", "");
  return response;
}

void append_upload(utils::MultipartFormData& form,
                   const std::string& field,
                   const std::optional<utils::UploadFile>& file) {
  if (!file) {
    return;
  }
  std::string effective_filename = file->filename.empty() ? "file.bin" : file->filename;
  std::string effective_type = file->content_type.value_or("application/octet-stream");
  form.append_file(field, effective_filename, effective_type, file->data);
}

VideoCreateRequest normalize_create_request(const VideoCreateRequest& request) {
  VideoCreateRequest normalized = request;
  if (!normalized.input_reference) {
    if (normalized.input_reference_data) {
      const std::string filename = normalized.input_reference_filename.value_or("input.bin");
      normalized.input_reference =
          utils::to_file(*normalized.input_reference_data, filename, normalized.input_reference_content_type);
    } else if (normalized.input_reference_path) {
      normalized.input_reference = utils::to_file(*normalized.input_reference_path,
                                                  normalized.input_reference_filename,
                                                  normalized.input_reference_content_type);
      if (normalized.input_reference && normalized.input_reference->filename.empty()) {
        normalized.input_reference->filename = filename_from_path(*normalized.input_reference_path);
      }
    }
  }
  return normalized;
}

VideoRemixParams normalize_remix_request(const VideoRemixParams& params) {
  VideoRemixParams normalized = params;
  if (!normalized.input_reference) {
    if (normalized.input_reference_data) {
      const std::string filename = normalized.input_reference_filename.value_or("input.bin");
      normalized.input_reference =
          utils::to_file(*normalized.input_reference_data, filename, normalized.input_reference_content_type);
    } else if (normalized.input_reference_path) {
      normalized.input_reference = utils::to_file(*normalized.input_reference_path,
                                                  normalized.input_reference_filename,
                                                  normalized.input_reference_content_type);
      if (normalized.input_reference && normalized.input_reference->filename.empty()) {
        normalized.input_reference->filename = filename_from_path(*normalized.input_reference_path);
      }
    }
  }
  return normalized;
}

}  // namespace

Video VideosResource::create(const VideoCreateRequest& request) const {
  return create(request, RequestOptions{});
}

Video VideosResource::create(const VideoCreateRequest& request, const RequestOptions& options) const {
  auto normalized = normalize_create_request(request);
  utils::MultipartFormData form;
  form.append_text("prompt", normalized.prompt);
  if (normalized.model) form.append_text("model", model_to_string(*normalized.model));
  if (normalized.seconds) form.append_text("seconds", seconds_to_string(*normalized.seconds));
  if (normalized.size) form.append_text("size", size_to_string(*normalized.size));
  append_upload(form, "input_reference", normalized.input_reference);
  auto encoded = form.build();

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = encoded.content_type;
  std::string body(encoded.body.begin(), encoded.body.end());

  auto response = client_.perform_request("POST", kVideosPath, body, request_options);
  try {
    return parse_video(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse video create response: ") + ex.what());
  }
}

Video VideosResource::retrieve(const std::string& video_id) const {
  return retrieve(video_id, RequestOptions{});
}

Video VideosResource::retrieve(const std::string& video_id, const RequestOptions& options) const {
  auto path = std::string(kVideosPath) + "/" + video_id;
  auto response = client_.perform_request("GET", path, "", options);
  try {
    return parse_video(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse video retrieve response: ") + ex.what());
  }
}

VideoList VideosResource::list(const VideoListParams& params) const {
  return list(params, RequestOptions{});
}

VideoList VideosResource::list(const VideoListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  auto response = client_.perform_request("GET", kVideosPath, "", request_options);
  try {
    return parse_video_list(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse video list response: ") + ex.what());
  }
}

CursorPage<Video> VideosResource::list_page(const VideoListParams& params) const {
  return list_page(params, RequestOptions{});
}

CursorPage<Video> VideosResource::list_page(const VideoListParams& params, const RequestOptions& options) const {
  RequestOptions request_options = options;
  if (params.limit) request_options.query_params["limit"] = std::to_string(*params.limit);
  if (params.order) request_options.query_params["order"] = *params.order;
  if (params.after) request_options.query_params["after"] = *params.after;

  auto fetch_impl = std::make_shared<std::function<CursorPage<Video>(const PageRequestOptions&)>>();

  *fetch_impl = [this, fetch_impl](const PageRequestOptions& request_options) -> CursorPage<Video> {
    RequestOptions next_options = to_request_options(request_options);
    auto response =
        client_.perform_request(request_options.method, request_options.path, request_options.body, next_options);
    VideoList list;
    try {
      list = parse_video_list(json::parse(response.body));
    } catch (const json::exception& ex) {
      throw OpenAIError(std::string("Failed to parse video list response: ") + ex.what());
    }

    std::optional<std::string> cursor = list.next_cursor;
    if (!cursor && !list.data.empty()) {
      cursor = list.data.back().id;
    }

    return CursorPage<Video>(std::move(list.data),
                             list.has_more,
                             std::move(cursor),
                             request_options,
                             *fetch_impl,
                             "after",
                             std::move(list.raw));
  };

  PageRequestOptions initial;
  initial.method = "GET";
  initial.path = kVideosPath;
  initial.headers = materialize_headers(request_options);
  initial.query = materialize_query(request_options);

  return (*fetch_impl)(initial);
}

VideoDeleteResponse VideosResource::remove(const std::string& video_id) const {
  return remove(video_id, RequestOptions{});
}

VideoDeleteResponse VideosResource::remove(const std::string& video_id, const RequestOptions& options) const {
  auto path = std::string(kVideosPath) + "/" + video_id;
  auto response = client_.perform_request("DELETE", path, "", options);
  try {
    return parse_video_delete(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse video delete response: ") + ex.what());
  }
}

VideoContent VideosResource::download_content(const std::string& video_id) const {
  return download_content(video_id, VideoDownloadContentParams{}, RequestOptions{});
}

VideoContent VideosResource::download_content(const std::string& video_id,
                                              const VideoDownloadContentParams& params) const {
  return download_content(video_id, params, RequestOptions{});
}

VideoContent VideosResource::download_content(const std::string& video_id,
                                              const VideoDownloadContentParams& params,
                                              const RequestOptions& options) const {
  RequestOptions request_options = options;
  request_options.headers["Accept"] = "application/binary";
  if (params.variant) {
    request_options.query_params["variant"] = variant_to_string(*params.variant);
  }
  auto path = std::string(kVideosPath) + "/" + video_id + "/content";
  auto response = client_.perform_request("GET", path, "", request_options);
  VideoContent content;
  content.headers = response.headers;
  content.data.assign(response.body.begin(), response.body.end());
  return content;
}

Video VideosResource::remix(const std::string& video_id, const VideoRemixParams& params) const {
  return remix(video_id, params, RequestOptions{});
}

Video VideosResource::remix(const std::string& video_id,
                            const VideoRemixParams& params,
                            const RequestOptions& options) const {
  auto normalized = normalize_remix_request(params);
  utils::MultipartFormData form;
  form.append_text("prompt", normalized.prompt);
  append_upload(form, "input_reference", normalized.input_reference);
  auto encoded = form.build();

  RequestOptions request_options = options;
  request_options.headers["Content-Type"] = encoded.content_type;
  std::string body(encoded.body.begin(), encoded.body.end());

  auto path = std::string(kVideosPath) + "/" + video_id + "/remix";
  auto response = client_.perform_request("POST", path, body, request_options);
  try {
    return parse_video(json::parse(response.body));
  } catch (const json::exception& ex) {
    throw OpenAIError(std::string("Failed to parse video remix response: ") + ex.what());
  }
}

}  // namespace openai

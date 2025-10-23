#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/utils/to_file.hpp"

namespace openai {

struct RequestOptions;
template <typename Item>
class CursorPage;
class OpenAIClient;

enum class VideoModel {
  Sora2,
  Sora2Pro
};

enum class VideoSeconds {
  Four,
  Eight,
  Twelve
};

enum class VideoSize {
  Size720x1280,
  Size1280x720,
  Size1024x1792,
  Size1792x1024
};

enum class VideoDownloadVariant {
  Video,
  Thumbnail,
  SpriteSheet
};

struct VideoCreateError {
  std::string code;
  std::string message;
};

struct Video {
  std::string id;
  std::optional<int> completed_at;
  int created_at = 0;
  std::optional<VideoCreateError> error;
  std::optional<int> expires_at;
  VideoModel model = VideoModel::Sora2;
  std::string object;
  int progress = 0;
  std::optional<std::string> remixed_from_video_id;
  VideoSeconds seconds = VideoSeconds::Four;
  VideoSize size = VideoSize::Size720x1280;
  std::string status;
  nlohmann::json raw = nlohmann::json::object();
};

struct VideoList {
  std::vector<Video> data;
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

struct VideoCreateRequest {
  std::string prompt;
  std::optional<utils::UploadFile> input_reference;
  std::optional<std::string> input_reference_path;
  std::optional<std::vector<std::uint8_t>> input_reference_data;
  std::optional<std::string> input_reference_filename;
  std::optional<std::string> input_reference_content_type;
  std::optional<VideoModel> model;
  std::optional<VideoSeconds> seconds;
  std::optional<VideoSize> size;
};

struct VideoListParams {
  std::optional<int> limit;
  std::optional<std::string> order;
  std::optional<std::string> after;
};

struct VideoDownloadContentParams {
  std::optional<VideoDownloadVariant> variant;
};

struct VideoRemixParams {
  std::string prompt;
  std::optional<utils::UploadFile> input_reference;
  std::optional<std::string> input_reference_path;
  std::optional<std::vector<std::uint8_t>> input_reference_data;
  std::optional<std::string> input_reference_filename;
  std::optional<std::string> input_reference_content_type;
};

struct VideoDeleteResponse {
  std::string id;
  bool deleted = false;
  std::string object;
  nlohmann::json raw = nlohmann::json::object();
};

struct VideoContent {
  std::vector<std::uint8_t> data;
  std::map<std::string, std::string> headers;
};

class VideosResource {
public:
  explicit VideosResource(OpenAIClient& client) : client_(client) {}

  Video create(const VideoCreateRequest& request) const;
  Video create(const VideoCreateRequest& request, const RequestOptions& options) const;

  Video retrieve(const std::string& video_id) const;
  Video retrieve(const std::string& video_id, const RequestOptions& options) const;

  VideoList list(const VideoListParams& params) const;
  VideoList list(const VideoListParams& params, const RequestOptions& options) const;
  VideoList list() const { return list(VideoListParams{}); }
  VideoList list(const RequestOptions& options) const { return list(VideoListParams{}, options); }

  CursorPage<Video> list_page(const VideoListParams& params) const;
  CursorPage<Video> list_page(const VideoListParams& params, const RequestOptions& options) const;

  CursorPage<Video> list_page() const;
  CursorPage<Video> list_page(const RequestOptions& options) const;

  VideoDeleteResponse remove(const std::string& video_id) const;
  VideoDeleteResponse remove(const std::string& video_id, const RequestOptions& options) const;

  VideoContent download_content(const std::string& video_id,
                                const VideoDownloadContentParams& params) const;
  VideoContent download_content(const std::string& video_id,
                                const VideoDownloadContentParams& params,
                                const RequestOptions& options) const;
  VideoContent download_content(const std::string& video_id) const;

  Video remix(const std::string& video_id, const VideoRemixParams& params) const;
  Video remix(const std::string& video_id,
              const VideoRemixParams& params,
              const RequestOptions& options) const;

private:
  OpenAIClient& client_;
};

}  // namespace openai

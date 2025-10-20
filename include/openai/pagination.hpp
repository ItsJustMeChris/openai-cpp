#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "openai/client.hpp"

namespace openai {

struct CursorPageMetadata {
  bool has_more = false;
  std::optional<std::string> next_cursor;
  nlohmann::json raw = nlohmann::json::object();
};

template <typename Item>
struct CursorPageResult {
  std::vector<Item> data;
  CursorPageMetadata meta;
};

template <typename Item>
class CursorPager {
public:
  using FetchPageFn = std::function<CursorPageResult<Item>(const PageRequestOptions&)>;

  CursorPager(OpenAIClient& client, std::string path, std::map<std::string, std::string> query, FetchPageFn fetch)
      : client_(client), path_(std::move(path)), query_(std::move(query)), fetch_(std::move(fetch)) {}

  CursorPageResult<Item> first(const RequestOptions& options = {}) const {
    PageRequestOptions request_options{.method = "GET",
                                       .path = path_,
                                       .headers = options.headers,
                                       .query = merge_query(options.query_params),
                                       .body = {}};
    return fetch_(request_options);
  }

  std::optional<CursorPageResult<Item>> next(const CursorPageMetadata& meta,
                                             const RequestOptions& options = {}) const {
    if (!meta.has_more || !meta.next_cursor) {
      return std::nullopt;
    }
    auto query = merge_query(options.query_params);
    query["after"] = *meta.next_cursor;
    PageRequestOptions request_options{.method = "GET",
                                       .path = path_,
                                       .headers = options.headers,
                                       .query = std::move(query),
                                       .body = {}};
    return fetch_(request_options);
  }

private:
  std::map<std::string, std::string> merge_query(const std::map<std::string, std::string>& extra) const {
    auto merged = query_;
    for (const auto& [key, value] : extra) {
      merged[key] = value;
    }
    return merged;
  }

  OpenAIClient& client_;
  std::string path_;
  std::map<std::string, std::string> query_;
  FetchPageFn fetch_;
};

}  // namespace openai


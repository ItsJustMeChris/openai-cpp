#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "openai/client.hpp"
#include "openai/error.hpp"

namespace openai {

template <typename Item>
class CursorPage {
public:
  using FetchPageFn = std::function<CursorPage<Item>(const PageRequestOptions&)>;

  CursorPage(std::vector<Item> data,
             bool has_more,
             std::optional<std::string> next_cursor,
             PageRequestOptions request_options,
             FetchPageFn fetch_page,
             std::string cursor_param = "after",
             nlohmann::json raw = nlohmann::json::object())
      : data_(std::move(data)),
        has_more_(has_more),
        next_cursor_(std::move(next_cursor)),
        request_options_(std::move(request_options)),
        fetch_page_(std::move(fetch_page)),
        cursor_param_(std::move(cursor_param)),
        raw_(std::move(raw)) {}

  const std::vector<Item>& data() const { return data_; }
  std::vector<Item>& data() { return data_; }

  bool empty() const { return data_.empty(); }

  bool has_next_page() const { return has_more_ && next_cursor_.has_value(); }

  const nlohmann::json& raw() const { return raw_; }

  CursorPage next_page() const {
    auto next_options = next_page_request_options();
    if (!next_options) {
      throw OpenAIError("No next page available; call has_next_page() before next_page().");
    }
    return fetch_page_(*next_options);
  }

  std::optional<std::string> next_cursor() const { return next_cursor_; }

  const PageRequestOptions& request_options() const { return request_options_; }

private:
  std::optional<PageRequestOptions> next_page_request_options() const {
    if (!has_more_ || !next_cursor_) {
      return std::nullopt;
    }
    PageRequestOptions options = request_options_;
    options.query[cursor_param_] = *next_cursor_;
    return options;
  }

  std::vector<Item> data_;
  bool has_more_;
  std::optional<std::string> next_cursor_;
  PageRequestOptions request_options_;
  FetchPageFn fetch_page_;
  std::string cursor_param_;
  nlohmann::json raw_;
};

inline std::map<std::string, std::string> materialize_headers(const RequestOptions& options) {
  std::map<std::string, std::string> headers;
  for (const auto& [key, value] : options.headers) {
    if (value.has_value()) {
      headers[key] = *value;
    }
  }
  return headers;
}

inline std::map<std::string, std::string> materialize_query(const RequestOptions& options) {
  std::map<std::string, std::string> query;
  for (const auto& [key, value] : options.query_params) {
    if (value.has_value()) {
      query[key] = *value;
    }
  }
  return query;
}

inline RequestOptions to_request_options(const PageRequestOptions& page_options) {
  RequestOptions options;
  for (const auto& [key, value] : page_options.headers) {
    options.headers[key] = value;
  }
  for (const auto& [key, value] : page_options.query) {
    options.query_params[key] = value;
  }
  return options;
}

}  // namespace openai

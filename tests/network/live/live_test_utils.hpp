#pragma once

#include "openai/chat.hpp"
#include "openai/client.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

namespace openai::test::live {

inline std::string trim_copy(std::string value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

inline bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  std::string normalized = trim_copy(std::string(raw));
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

inline std::optional<std::string> get_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::strlen(value) == 0) {
    return std::nullopt;
  }
  std::string result = trim_copy(std::string(value));
  if (result.empty()) {
    return std::nullopt;
  }
  return result;
}

inline std::string unique_tag() {
  auto now = std::chrono::system_clock::now();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return "cpp-live-" + std::to_string(micros);
}

inline std::optional<openai::ClientOptions> make_live_client_options() {
  openai::ClientOptions options;

  auto api_key = get_env("OPENAI_API_KEY");
  if (!api_key) {
    return std::nullopt;
  }
  options.api_key = *api_key;

  if (auto base = get_env("TEST_API_BASE_URL")) {
    options.base_url = *base;
  } else if (auto env_base = get_env("OPENAI_BASE_URL")) {
    options.base_url = *env_base;
  }

  return options;
}

inline openai::ChatMessage make_text_message(const std::string& role, const std::string& text) {
  openai::ChatMessage message;
  message.role = role;
  openai::ChatMessageContent content;
  content.type = openai::ChatMessageContent::Type::Text;
  content.text = text;
  message.content.push_back(std::move(content));
  return message;
}

}  // namespace openai::test::live


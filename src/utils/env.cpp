#include "openai/utils/env.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace openai::utils {
namespace {

std::string trim(std::string value) {
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  auto begin = std::find_if_not(value.begin(), value.end(), is_space);
  auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

}  // namespace

std::optional<std::string> read_env(const std::string& name) {
  const char* raw = std::getenv(name.c_str());
  if (!raw) {
    return std::nullopt;
  }
  std::string trimmed = trim(raw);
  if (trimmed.empty()) {
    return std::string();
  }
  return trimmed;
}

std::string read_env_or(const std::string& name, const std::string& fallback) {
  if (auto value = read_env(name)) {
    return *value;
  }
  return fallback;
}

}  // namespace openai::utils


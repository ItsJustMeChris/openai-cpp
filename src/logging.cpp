#include "openai/logging.hpp"

#include <algorithm>

namespace openai {

LogLevel parse_log_level(const std::string& value, LogLevel fallback) {
  std::string lowered; lowered.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(lowered), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "off") return LogLevel::Off;
  if (lowered == "error") return LogLevel::Error;
  if (lowered == "warn" || lowered == "warning") return LogLevel::Warn;
  if (lowered == "info") return LogLevel::Info;
  if (lowered == "debug") return LogLevel::Debug;
  return fallback;
}

}  // namespace openai


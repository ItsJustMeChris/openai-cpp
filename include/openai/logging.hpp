#pragma once

#include <functional>
#include <string>

#include <nlohmann/json.hpp>

namespace openai {

enum class LogLevel { Off = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

using LoggerCallback = std::function<void(LogLevel level, const std::string& message, const nlohmann::json& details)>;

LogLevel parse_log_level(const std::string& value, LogLevel fallback = LogLevel::Off);

}  // namespace openai


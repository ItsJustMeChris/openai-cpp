#pragma once

#include <optional>
#include <string>

namespace openai::utils {

/**
 * Reads an environment variable and trims leading/trailing whitespace.
 * Returns std::nullopt when the variable is not set.
 */
std::optional<std::string> read_env(const std::string& name);

/**
 * Helper to read an environment variable and return its value or a provided fallback.
 */
std::string read_env_or(const std::string& name, const std::string& fallback);

}  // namespace openai::utils


#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "openai/error.hpp"

namespace openai::utils {

bool is_absolute_url(std::string_view url);

template <typename Integer,
          typename = std::enable_if_t<std::is_integral_v<Integer>>>
Integer validate_positive_integer(const std::string& name, Integer value) {
  if (value < 0) {
    throw OpenAIError(name + " must be a positive integer");
  }
  return value;
}

std::optional<nlohmann::json> safe_json(const std::string& text);

template <typename T>
T ensure_present(const std::optional<T>& value, const std::string& name = "value") {
  if (!value.has_value()) {
    throw OpenAIError("Expected " + name + " to be present");
  }
  return *value;
}

bool is_empty_object(const nlohmann::json& value);

bool is_object(const nlohmann::json& value);

bool has_own(const nlohmann::json& object, const std::string& key);

nlohmann::json maybe_object(const nlohmann::json& value);

std::int64_t coerce_integer(const nlohmann::json& value);
std::optional<std::int64_t> maybe_coerce_integer(const nlohmann::json& value);

double coerce_float(const nlohmann::json& value);
std::optional<double> maybe_coerce_float(const nlohmann::json& value);

bool coerce_boolean(const nlohmann::json& value);
std::optional<bool> maybe_coerce_boolean(const nlohmann::json& value);

}  // namespace openai::utils


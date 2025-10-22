#include "openai/utils/values.hpp"

#include <cctype>
#include <cmath>
#include <limits>
#include <string>

namespace openai::utils {

bool is_absolute_url(std::string_view url) {
  auto colon_pos = url.find(':');
  if (colon_pos == std::string_view::npos) {
    return false;
  }
  if (colon_pos == 0) {
    return false;
  }

  unsigned char first = static_cast<unsigned char>(url[0]);
  if (!std::isalpha(first)) {
    return false;
  }

  for (std::size_t i = 1; i < colon_pos; ++i) {
    unsigned char ch = static_cast<unsigned char>(url[i]);
    if (!(std::isalnum(ch) || ch == '+' || ch == '.' || ch == '-')) {
      return false;
    }
  }

  return true;
}

std::optional<nlohmann::json> safe_json(const std::string& text) {
  if (text.empty()) {
    return std::nullopt;
  }
  try {
    return nlohmann::json::parse(text);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool is_empty_object(const nlohmann::json& value) {
  return value.is_object() && value.empty();
}

bool is_object(const nlohmann::json& value) {
  return value.is_object();
}

bool has_own(const nlohmann::json& object, const std::string& key) {
  if (!object.is_object()) {
    return false;
  }
  return object.find(key) != object.end();
}

nlohmann::json maybe_object(const nlohmann::json& value) {
  if (value.is_object()) {
    return value;
  }
  return nlohmann::json::object();
}

namespace {

[[noreturn]] void throw_coerce_error(const nlohmann::json& value, const std::string& type) {
  throw OpenAIError("Could not coerce " + value.dump() + " (type: " + type + ") into a number");
}

bool json_truthy(const nlohmann::json& value) {
  if (value.is_null()) {
    return false;
  }
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_number_integer()) {
    return value.get<std::int64_t>() != 0;
  }
  if (value.is_number_float()) {
    double number = value.get<double>();
    if (std::isnan(number)) {
      return false;
    }
    return number != 0.0;
  }
  if (value.is_string()) {
    return !value.get_ref<const std::string&>().empty();
  }
  if (value.is_array()) {
    return true;
  }
  if (value.is_object()) {
    return true;
  }
  return false;
}

}  // namespace

std::int64_t coerce_integer(const nlohmann::json& value) {
  if (value.is_number_integer()) {
    return value.get<std::int64_t>();
  }
  if (value.is_number_float()) {
    double number = value.get<double>();
    if (std::isnan(number) || !std::isfinite(number)) {
      throw_coerce_error(value, "number");
    }
    return static_cast<std::int64_t>(std::llround(number));
  }
  if (value.is_string()) {
    const auto& str = value.get_ref<const std::string&>();
    if (str.empty()) {
      throw_coerce_error(value, "string");
    }
    std::size_t idx = 0;
    try {
      auto result = std::stoll(str, &idx, 10);
      if (idx != str.size()) {
        throw_coerce_error(value, "string");
      }
      return result;
    } catch (const std::exception&) {
      throw_coerce_error(value, "string");
    }
  }
  throw_coerce_error(value, value.type_name());
}

std::optional<std::int64_t> maybe_coerce_integer(const nlohmann::json& value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  return coerce_integer(value);
}

double coerce_float(const nlohmann::json& value) {
  if (value.is_number()) {
    return value.get<double>();
  }
  if (value.is_string()) {
    const auto& str = value.get_ref<const std::string&>();
    if (str.empty()) {
      throw OpenAIError("Could not coerce \"\" (type: string) into a number");
    }
    std::size_t idx = 0;
    try {
      double parsed = std::stod(str, &idx);
      if (idx != str.size()) {
        throw OpenAIError("Could not coerce \"" + str + "\" (type: string) into a number");
      }
      return parsed;
    } catch (const std::exception&) {
      throw OpenAIError("Could not coerce \"" + str + "\" (type: string) into a number");
    }
  }
  throw OpenAIError("Could not coerce " + value.dump() + " (type: " + value.type_name() + ") into a number");
}

std::optional<double> maybe_coerce_float(const nlohmann::json& value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  return coerce_float(value);
}

bool coerce_boolean(const nlohmann::json& value) {
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_string()) {
    return value.get_ref<const std::string&>() == "true";
  }
  return json_truthy(value);
}

std::optional<bool> maybe_coerce_boolean(const nlohmann::json& value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  return coerce_boolean(value);
}

}  // namespace openai::utils

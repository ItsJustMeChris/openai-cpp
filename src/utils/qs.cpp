#include "openai/utils/qs.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace openai::utils::qs {

namespace {

bool is_unreserved(unsigned char c, Format format) {
  if (std::isalnum(c) != 0) {
    return true;
  }
  switch (c) {
    case '-':
    case '_':
    case '.':
    case '~':
      return true;
    case '(':
    case ')':
      return format == Format::RFC1738;
    default:
      return false;
  }
}

std::string percent_encode(const std::string& input, Charset charset, Format format) {
  if (input.empty()) {
    return input;
  }

  std::ostringstream encoded;
  encoded << std::uppercase << std::hex;

  for (unsigned char byte : input) {
    if (is_unreserved(byte, format)) {
      encoded << static_cast<char>(byte);
      continue;
    }

    encoded << '%';
    encoded << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  }

  return encoded.str();
}

std::string default_formatter(const std::string& value) {
  return value;
}

std::string rfc1738_formatter(const std::string& value) {
  std::string result = value;
  for (std::size_t pos = result.find("%20"); pos != std::string::npos; pos = result.find("%20", pos)) {
    result.replace(pos, 3, "+");
    pos += 1;
  }
  return result;
}

std::string replace_all(std::string value, const std::string& needle, const std::string& replacement) {
  std::size_t pos = 0;
  while ((pos = value.find(needle, pos)) != std::string::npos) {
    value.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
  return value;
}

bool is_non_nullish_primitive(const nlohmann::json& value) {
  return value.is_string() || value.is_number() || value.is_boolean();
}

std::string number_to_string(const nlohmann::json& value) {
  std::ostringstream oss;
  if (value.is_number_float()) {
    oss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << value.get<double>();
  } else if (value.is_number_integer()) {
    oss << value.get<long long>();
  } else if (value.is_number_unsigned()) {
    oss << value.get<unsigned long long>();
  }
  return oss.str();
}

std::string json_to_string(const nlohmann::json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? "true" : "false";
  }
  if (value.is_number()) {
    return number_to_string(value);
  }
  if (value.is_null()) {
    return {};
  }
  return value.dump();
}

struct NormalizedOptions {
  bool add_query_prefix;
  bool allow_dots;
  bool allow_empty_arrays;
  ArrayFormat array_format;
  Charset charset;
  bool charset_sentinel;
  bool comma_round_trip;
  std::string delimiter;
  bool encode;
  bool encode_dot_in_keys;
  bool encode_values_only;
  Format format;
  Formatter formatter;
  DefaultEncoder default_encoder;
  Encoder encoder;
  bool skip_nulls;
  bool strict_null_handling;
  std::optional<FilterFunction> filter;
  std::optional<std::vector<std::string>> filter_keys;
  std::optional<Sorter> sort;
};

NormalizedOptions normalize_options(const StringifyOptions& options) {
  NormalizedOptions normalized{
      options.add_query_prefix,
      options.allow_dots,
      options.allow_empty_arrays,
      options.array_format,
      options.charset,
      options.charset_sentinel,
      options.comma_round_trip,
      options.delimiter,
      options.encode,
      options.encode_dot_in_keys,
      options.encode_values_only,
      options.format,
      options.format == Format::RFC1738 ? Formatter(rfc1738_formatter)
                                        : Formatter(default_formatter),
      [](const std::string& value, Charset charset, Format format) {
        return percent_encode(value, charset, format);
      },
      [](const std::string& value,
         const DefaultEncoder& default_encoder,
         Charset charset,
         EncoderTarget,
         Format format) {
        return default_encoder(value, charset, format);
      },
      options.skip_nulls,
      options.strict_null_handling,
      options.filter,
      options.filter_keys,
      options.sort};

  if (options.indices.has_value() && options.array_format == ArrayFormat::Indices) {
    normalized.array_format = *options.indices ? ArrayFormat::Indices : ArrayFormat::Repeat;
  }

  if (options.encoder.has_value()) {
    normalized.encoder = *options.encoder;
  }

  if (options.formatter.has_value()) {
    normalized.formatter = *options.formatter;
  }

  return normalized;
}

std::string apply_encoder(const Encoder& encoder,
                          const DefaultEncoder& default_encoder,
                          Charset charset,
                          Format format,
                          EncoderTarget target,
                          const std::string& value) {
  return encoder(value, default_encoder, charset, target, format);
}

std::string encode_prefix_component(const std::string& prefix, bool encode_dot_in_keys) {
  if (!encode_dot_in_keys) {
    return prefix;
  }
  return replace_all(prefix, ".", "%2E");
}

void inner_stringify(const nlohmann::json& input,
                     const std::string& prefix,
                     const NormalizedOptions& options,
                     const Encoder* encoder,
                     std::vector<std::string>& out) {
  nlohmann::json value = input;

  if (options.filter.has_value()) {
    value = (*options.filter)(prefix, value);
  }

  if (value.is_null()) {
    if (options.strict_null_handling) {
      std::string key = encode_prefix_component(prefix, options.encode_dot_in_keys);
      if (encoder && !options.encode_values_only) {
        key = apply_encoder(*encoder,
                            options.default_encoder,
                            options.charset,
                            options.format,
                            EncoderTarget::Key,
                            key);
      }
      out.emplace_back(options.formatter(key));
      return;
    }
    value = "";
  }

  if (value.is_discarded()) {
    return;
  }

  if (is_non_nullish_primitive(value)) {
    std::string key = encode_prefix_component(prefix, options.encode_dot_in_keys);
    if (encoder) {
      std::string encoded_key = options.encode_values_only
                                    ? key
                                    : apply_encoder(*encoder,
                                                    options.default_encoder,
                                                    options.charset,
                                                    options.format,
                                                    EncoderTarget::Key,
                                                    key);
      std::string encoded_value = apply_encoder(*encoder,
                                                options.default_encoder,
                                                options.charset,
                                                options.format,
                                                EncoderTarget::Value,
                                                json_to_string(value));
      out.emplace_back(options.formatter(encoded_key) + "=" + options.formatter(encoded_value));
    } else {
      out.emplace_back(options.formatter(key) + "=" + options.formatter(json_to_string(value)));
    }
    return;
  }

  if (value.is_array()) {
    const auto& array = value;
    std::string encoded_prefix = encode_prefix_component(prefix, options.encode_dot_in_keys);
    std::string adjusted_prefix = (options.comma_round_trip && array.size() == 1)
                                      ? encoded_prefix + "[]"
                                      : encoded_prefix;

    if (array.empty()) {
      if (options.allow_empty_arrays) {
        out.emplace_back(adjusted_prefix + "[]");
      }
      return;
    }

    if (options.array_format == ArrayFormat::Comma) {
      std::vector<std::string> parts;
      parts.reserve(array.size());
      for (const auto& entry : array) {
        if (entry.is_null()) {
          parts.emplace_back("");
        } else {
          std::string rendered = json_to_string(entry);
          if (encoder && options.encode_values_only) {
            rendered = apply_encoder(*encoder,
                                     options.default_encoder,
                                     options.charset,
                                     options.format,
                                     EncoderTarget::Value,
                                     rendered);
          }
          parts.emplace_back(rendered);
        }
      }

      std::ostringstream joined;
      for (std::size_t idx = 0; idx < parts.size(); ++idx) {
        if (idx > 0) {
          joined << ',';
        }
        joined << parts[idx];
      }

      std::string joined_value = joined.str();
      nlohmann::json child;
      if (joined_value.empty()) {
        child = nlohmann::json(nullptr);
      } else {
        child = joined_value;
      }

      const Encoder* next_encoder =
          (options.encode_values_only && encoder) ? nullptr : encoder;
      inner_stringify(child, adjusted_prefix, options, next_encoder, out);
      return;
    }

    for (std::size_t index = 0; index < array.size(); ++index) {
      const auto& element = array[index];
      if (options.skip_nulls && element.is_null()) {
        continue;
      }

      std::string key_prefix;
      switch (options.array_format) {
        case ArrayFormat::Indices:
          key_prefix = adjusted_prefix + "[" + std::to_string(index) + "]";
          break;
        case ArrayFormat::Brackets:
          key_prefix = adjusted_prefix + "[]";
          break;
        case ArrayFormat::Repeat:
          key_prefix = adjusted_prefix;
          break;
        case ArrayFormat::Comma:
          key_prefix = adjusted_prefix;
          break;
      }
      inner_stringify(element, key_prefix, options, encoder, out);
    }
    return;
  }

  if (!value.is_object()) {
    std::string key = encode_prefix_component(prefix, options.encode_dot_in_keys);
    std::string rendered = json_to_string(value);
    if (encoder) {
      std::string encoded_key = options.encode_values_only
                                    ? key
                                    : apply_encoder(*encoder,
                                                    options.default_encoder,
                                                    options.charset,
                                                    options.format,
                                                    EncoderTarget::Key,
                                                    key);
      std::string encoded_value = apply_encoder(*encoder,
                                                options.default_encoder,
                                                options.charset,
                                                options.format,
                                                EncoderTarget::Value,
                                                rendered);
      out.emplace_back(options.formatter(encoded_key) + "=" + options.formatter(encoded_value));
    } else {
      out.emplace_back(options.formatter(key) + "=" + options.formatter(rendered));
    }
    return;
  }

  std::vector<std::string> keys;
  keys.reserve(value.size());

  if (options.filter_keys.has_value()) {
    keys = *options.filter_keys;
  } else {
    for (const auto& item : value.items()) {
      keys.push_back(item.key());
    }
  }

  if (options.sort.has_value()) {
    std::sort(keys.begin(), keys.end(), *options.sort);
  }

  std::string encoded_prefix = encode_prefix_component(prefix, options.encode_dot_in_keys);
  std::string adjusted_prefix = encoded_prefix;

  for (const auto& key : keys) {
    if (!value.contains(key)) {
      continue;
    }
    const auto& child = value.at(key);
    if (options.skip_nulls && child.is_null()) {
      continue;
    }
    std::string encoded_key = key;
    if (options.allow_dots && options.encode_dot_in_keys) {
      encoded_key = replace_all(encoded_key, ".", "%2E");
    }
    std::string key_prefix;
    if (value.is_array()) {
      key_prefix = adjusted_prefix;
    } else {
      key_prefix = adjusted_prefix +
                   (options.allow_dots ? "." + encoded_key : "[" + encoded_key + "]");
    }
    inner_stringify(child, key_prefix, options, encoder, out);
  }
}

}  // namespace

std::string stringify(const nlohmann::json& object, const StringifyOptions& options) {
  NormalizedOptions normalized = normalize_options(options);

  nlohmann::json value = object;
  if (normalized.filter.has_value()) {
    value = (*normalized.filter)("", value);
  }

  if (!value.is_object() || value.is_null()) {
    return {};
  }

  std::vector<std::string> keys;
  keys.reserve(value.size());

  if (normalized.filter_keys.has_value()) {
    keys = *normalized.filter_keys;
  } else {
    for (const auto& item : value.items()) {
      keys.push_back(item.key());
    }
  }

  if (normalized.sort.has_value()) {
    std::sort(keys.begin(), keys.end(), *normalized.sort);
  }

  std::vector<std::string> fragments;

  const Encoder* encoder_ptr = normalized.encode ? &normalized.encoder : nullptr;

  for (const auto& key : keys) {
    if (!value.contains(key)) {
      continue;
    }
    const auto& child = value.at(key);
    if (normalized.skip_nulls && child.is_null()) {
      continue;
    }
    inner_stringify(child, key, normalized, encoder_ptr, fragments);
  }

  if (fragments.empty()) {
    return {};
  }

  std::ostringstream joined;
  for (std::size_t idx = 0; idx < fragments.size(); ++idx) {
    if (idx > 0) {
      joined << normalized.delimiter;
    }
    joined << fragments[idx];
  }

  std::string prefix = normalized.add_query_prefix ? "?" : "";
  if (normalized.charset_sentinel) {
    prefix += normalized.charset == Charset::ISO_8859_1
                  ? "utf8=%26%2310003%3B&"
                  : "utf8=%E2%9C%93&";
  }

  return prefix + joined.str();
}

std::string stringify(const std::map<std::string, std::string>& object,
                      const StringifyOptions& options) {
  nlohmann::json json_object = nlohmann::json::object();
  for (const auto& [key, value] : object) {
    json_object[key] = value;
  }
  return stringify(json_object, options);
}

}  // namespace openai::utils::qs

#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai::utils::qs {

enum class Format { RFC1738, RFC3986 };
enum class Charset { UTF8, ISO_8859_1 };
enum class ArrayFormat { Indices, Brackets, Repeat, Comma };
enum class EncoderTarget { Key, Value };

using DefaultEncoder = std::function<std::string(const std::string&, Charset, Format)>;

using Encoder = std::function<std::string(const std::string&,
                                          const DefaultEncoder&,
                                          Charset,
                                          EncoderTarget,
                                          Format)>;

using FilterFunction = std::function<nlohmann::json(const std::string&, const nlohmann::json&)>;
using Sorter = std::function<bool(const std::string&, const std::string&)>;
using Formatter = std::function<std::string(const std::string&)>;

struct StringifyOptions {
  bool add_query_prefix = false;
  bool allow_dots = false;
  bool allow_empty_arrays = false;
  ArrayFormat array_format = ArrayFormat::Indices;
  Charset charset = Charset::UTF8;
  bool charset_sentinel = false;
  bool comma_round_trip = false;
  std::string delimiter = "&";
  bool encode = true;
  bool encode_dot_in_keys = false;
  bool encode_values_only = false;
  Format format = Format::RFC3986;
  std::optional<Encoder> encoder;
  std::optional<FilterFunction> filter;
  std::optional<std::vector<std::string>> filter_keys;
  std::optional<Sorter> sort;
  std::optional<Formatter> formatter;
  bool skip_nulls = false;
  bool strict_null_handling = false;
  std::optional<bool> indices;
};

[[nodiscard]] std::string stringify(const nlohmann::json& object,
                                    const StringifyOptions& options = {});

[[nodiscard]] std::string stringify(const std::map<std::string, std::string>& object,
                                    const StringifyOptions& options = {});

}  // namespace openai::utils::qs

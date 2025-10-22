#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai::utils {

struct MultipartEncoded {
  std::string content_type;
  std::string body;
};

class MultipartFormData {
public:
  MultipartFormData();

  void append_text(const std::string& name, const std::string& value);
  void append_file(const std::string& name,
                   const std::string& filename,
                   const std::string& content_type,
                   const std::vector<std::uint8_t>& data);
  void append_json(const std::string& name, const nlohmann::json& value);

  MultipartEncoded build() const;

private:
  struct Part {
    std::string name;
    std::optional<std::string> filename;
    std::optional<std::string> content_type;
    std::string data;
  };

  void append_json_internal(const std::string& name, const nlohmann::json& value);

  std::string boundary_;
  std::vector<Part> parts_;
};

}  // namespace openai::utils


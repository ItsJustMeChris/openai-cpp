#include "openai/utils/multipart.hpp"

#include "openai/utils/uuid.hpp"

#include <sstream>

namespace openai::utils {
namespace {

std::string escape_quotes(const std::string& input) {
  std::string escaped;
  escaped.reserve(input.size());
  for (char ch : input) {
    if (ch == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

std::string to_string(const nlohmann::json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? "true" : "false";
  }
  if (value.is_number()) {
    return value.dump();
  }
  return value.dump();
}

}  // namespace

MultipartFormData::MultipartFormData() : boundary_("openai-cpp-" + uuid4()) {}

void MultipartFormData::append_text(const std::string& name, const std::string& value) {
  parts_.push_back(Part{.name = name, .filename = std::nullopt, .content_type = std::nullopt, .data = value});
}

void MultipartFormData::append_file(const std::string& name,
                                    const std::string& filename,
                                    const std::string& content_type,
                                    const std::vector<std::uint8_t>& data) {
  parts_.push_back(Part{.name = name,
                        .filename = filename,
                        .content_type = content_type,
                        .data = std::string(reinterpret_cast<const char*>(data.data()), data.size())});
}

void MultipartFormData::append_json(const std::string& name, const nlohmann::json& value) {
  append_json_internal(name, value);
}

void MultipartFormData::append_json_internal(const std::string& name, const nlohmann::json& value) {
  if (value.is_null()) {
    throw std::invalid_argument("Null value cannot be encoded in multipart form data");
  }
  if (value.is_object()) {
    for (auto it = value.begin(); it != value.end(); ++it) {
      append_json_internal(name + "[" + it.key() + "]", it.value());
    }
  } else if (value.is_array()) {
    for (const auto& entry : value) {
      append_json_internal(name + "[]", entry);
    }
  } else {
    append_text(name, to_string(value));
  }
}

MultipartEncoded MultipartFormData::build() const {
  std::ostringstream body;
  for (const auto& part : parts_) {
    body << "--" << boundary_ << "\r\n";
    body << "Content-Disposition: form-data; name=\"" << escape_quotes(part.name) << "\"";
    if (part.filename.has_value()) {
      body << "; filename=\"" << escape_quotes(*part.filename) << "\"";
    }
    body << "\r\n";
    if (part.content_type.has_value()) {
      body << "Content-Type: " << *part.content_type << "\r\n";
    }
    body << "\r\n";
    body << part.data;
    body << "\r\n";
  }
  body << "--" << boundary_ << "--\r\n";

  MultipartEncoded encoded;
  encoded.content_type = "multipart/form-data; boundary=" + boundary_;
  encoded.body = body.str();
  return encoded;
}

}  // namespace openai::utils


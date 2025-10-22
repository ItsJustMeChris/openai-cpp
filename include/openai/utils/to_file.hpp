#pragma once

#include <istream>
#include <optional>
#include <string>
#include <vector>

namespace openai::utils {

struct UploadFile {
  std::vector<std::uint8_t> data;
  std::string filename;
  std::optional<std::string> content_type;
};

/**
 * Converts a filesystem path into an UploadFile by reading the contents from disk.
 * When filename_override is provided it will be used instead of the basename of the path.
 */
UploadFile to_file(const std::string& path,
                   std::optional<std::string> filename_override = std::nullopt,
                   std::optional<std::string> content_type = std::nullopt);

/**
 * Wraps an existing byte vector as an UploadFile with the provided filename.
 */
UploadFile to_file(std::vector<std::uint8_t> data,
                   const std::string& filename,
                   std::optional<std::string> content_type = std::nullopt);

/**
 * Uses a string buffer as the UploadFile's data.
 */
UploadFile to_file(const std::string& data,
                   const std::string& filename,
                   std::optional<std::string> content_type = std::nullopt);

/**
 * Reads all bytes from the provided std::istream into an UploadFile.
 * The stream is consumed until EOF.
 */
UploadFile to_file(std::istream& stream,
                   const std::string& filename,
                   std::optional<std::string> content_type = std::nullopt);

}  // namespace openai::utils


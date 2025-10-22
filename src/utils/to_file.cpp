#include "openai/utils/to_file.hpp"

#include "openai/error.hpp"

#include <array>
#include <filesystem>
#include <fstream>

namespace openai::utils {
namespace {

std::vector<std::uint8_t> read_all_bytes(std::istream& stream) {
  std::vector<std::uint8_t> buffer;
  std::array<char, 4096> chunk{};
  while (stream.good()) {
    stream.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    std::streamsize count = stream.gcount();
    if (count > 0) {
      buffer.insert(buffer.end(),
                    reinterpret_cast<std::uint8_t*>(chunk.data()),
                    reinterpret_cast<std::uint8_t*>(chunk.data()) + count);
    }
  }
  if (!stream.eof() && stream.fail()) {
    throw OpenAIError("Failed to read data from stream");
  }
  return buffer;
}

std::string basename(const std::string& path) {
  std::filesystem::path fs_path(path);
  auto filename = fs_path.filename().string();
  if (filename.empty()) {
    return "file";
  }
  return filename;
}

}  // namespace

UploadFile to_file(const std::string& path,
                   std::optional<std::string> filename_override,
                   std::optional<std::string> content_type) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw OpenAIError("Failed to open file: " + path);
  }
  auto data = read_all_bytes(file);
  UploadFile upload;
  upload.data = std::move(data);
  upload.filename = filename_override.value_or(basename(path));
  upload.content_type = std::move(content_type);
  return upload;
}

UploadFile to_file(std::vector<std::uint8_t> data,
                   const std::string& filename,
                   std::optional<std::string> content_type) {
  if (filename.empty()) {
    throw OpenAIError("Filename must not be empty when wrapping byte data");
  }
  UploadFile upload;
  upload.data = std::move(data);
  upload.filename = filename;
  upload.content_type = std::move(content_type);
  return upload;
}

UploadFile to_file(const std::string& data,
                   const std::string& filename,
                   std::optional<std::string> content_type) {
  std::vector<std::uint8_t> bytes(data.begin(), data.end());
  return to_file(std::move(bytes), filename, std::move(content_type));
}

UploadFile to_file(std::istream& stream,
                   const std::string& filename,
                   std::optional<std::string> content_type) {
  if (!stream.good()) {
    throw OpenAIError("Input stream is not readable");
  }
  auto data = read_all_bytes(stream);
  return to_file(std::move(data), filename, std::move(content_type));
}

}  // namespace openai::utils

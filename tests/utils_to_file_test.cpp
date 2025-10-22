#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "openai/utils/to_file.hpp"

using openai::utils::UploadFile;
using openai::utils::to_file;

TEST(UtilsToFileTest, ReadsFromPath) {
  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-to-file.txt";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "hello";
  }

  UploadFile upload = to_file(tmp.string());
  EXPECT_EQ(upload.filename, "openai-to-file.txt");
  EXPECT_FALSE(upload.content_type.has_value());
  ASSERT_EQ(upload.data.size(), 5u);
  EXPECT_EQ(std::string(upload.data.begin(), upload.data.end()), "hello");

  std::filesystem::remove(tmp);
}

TEST(UtilsToFileTest, WrapsByteVector) {
  std::vector<std::uint8_t> bytes = {0x01, 0x02, 0x03};
  UploadFile upload = to_file(std::move(bytes), "data.bin", "application/octet-stream");
  EXPECT_EQ(upload.filename, "data.bin");
  ASSERT_TRUE(upload.content_type.has_value());
  EXPECT_EQ(*upload.content_type, "application/octet-stream");
  ASSERT_EQ(upload.data.size(), 3u);
  EXPECT_EQ(upload.data[0], 0x01);
}

TEST(UtilsToFileTest, ReadsFromStringStream) {
  std::istringstream stream("stream-data");
  UploadFile upload = to_file(stream, "stream.txt");
  EXPECT_EQ(upload.filename, "stream.txt");
  ASSERT_EQ(upload.data.size(), 11u);
  EXPECT_EQ(std::string(upload.data.begin(), upload.data.end()), "stream-data");
}

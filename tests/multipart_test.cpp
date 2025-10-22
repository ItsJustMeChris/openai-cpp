#include <gtest/gtest.h>

#include "openai/utils/multipart.hpp"

#include <nlohmann/json.hpp>

using openai::utils::MultipartFormData;
using openai::utils::MultipartEncoded;

TEST(MultipartFormDataTest, EncodesTextAndFileParts) {
  MultipartFormData form;
  form.append_text("purpose", "assistants");
  std::vector<std::uint8_t> data{'h', 'e', 'l', 'l', 'o'};
  form.append_file("file", "hello.txt", "text/plain", data);

  MultipartEncoded encoded = form.build();

  EXPECT_NE(encoded.content_type.find("multipart/form-data; boundary="), std::string::npos);
  EXPECT_NE(encoded.body.find("name=\"purpose\""), std::string::npos);
  EXPECT_NE(encoded.body.find("assistants"), std::string::npos);
  EXPECT_NE(encoded.body.find("filename=\"hello.txt\""), std::string::npos);
  EXPECT_NE(encoded.body.find("text/plain"), std::string::npos);
  EXPECT_NE(encoded.body.find("hello"), std::string::npos);
}

TEST(MultipartFormDataTest, EncodesNestedJsonValues) {
  MultipartFormData form;
  nlohmann::json payload = {
      {"metadata", { {"key", "value"}, {"flags", {true, false}} }},
      {"count", 3}
  };
  form.append_json("config", payload);

  MultipartEncoded encoded = form.build();

  EXPECT_NE(encoded.body.find("name=\"config[metadata][key]\""), std::string::npos);
  EXPECT_NE(encoded.body.find("value"), std::string::npos);
  EXPECT_NE(encoded.body.find("name=\"config[metadata][flags][]\""), std::string::npos);
  EXPECT_NE(encoded.body.find("true"), std::string::npos);
  EXPECT_NE(encoded.body.find("false"), std::string::npos);
  EXPECT_NE(encoded.body.find("name=\"config[count]\""), std::string::npos);
  EXPECT_NE(encoded.body.find("3"), std::string::npos);
}

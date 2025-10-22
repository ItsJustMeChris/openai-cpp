#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/utils/qs.hpp"

namespace qs = openai::utils::qs;

TEST(QSStringifyTest, EncodesSimpleObject) {
  nlohmann::json payload = nlohmann::json::object();
  payload["foo"] = "bar";
  payload["answer"] = 42;

  qs::StringifyOptions options;
  options.array_format = qs::ArrayFormat::Indices;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "answer=42&foo=bar");
}

TEST(QSStringifyTest, EncodesNestedObjectsWithBracketNotation) {
  nlohmann::json payload = nlohmann::json::object();
  payload["filter"] = nlohmann::json::object();
  payload["filter"]["state"] = "active";
  payload["filter"]["page"] = 3;

  qs::StringifyOptions options;
  options.array_format = qs::ArrayFormat::Indices;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "filter%5Bpage%5D=3&filter%5Bstate%5D=active");
}

TEST(QSStringifyTest, EncodesArraysWithBracketsFormat) {
  nlohmann::json payload = nlohmann::json::object();
  payload["tags"] = nlohmann::json::array({"alpha", "beta"});

  qs::StringifyOptions options;
  options.array_format = qs::ArrayFormat::Brackets;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "tags%5B%5D=alpha&tags%5B%5D=beta");
}

TEST(QSStringifyTest, EncodesArraysWithRepeatFormat) {
  nlohmann::json payload = nlohmann::json::object();
  payload["tags"] = nlohmann::json::array({"alpha", "beta"});

  qs::StringifyOptions options;
  options.array_format = qs::ArrayFormat::Repeat;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "tags=alpha&tags=beta");
}

TEST(QSStringifyTest, SkipsNullValuesWhenConfigured) {
  nlohmann::json payload = nlohmann::json::object();
  payload["foo"] = nullptr;
  payload["bar"] = "baz";

  qs::StringifyOptions options;
  options.skip_nulls = true;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "bar=baz");
}

TEST(QSStringifyTest, StrictNullHandlingProducesKeyOnly) {
  nlohmann::json payload = nlohmann::json::object();
  payload["foo"] = nullptr;

  qs::StringifyOptions options;
  options.strict_null_handling = true;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "foo");
}

TEST(QSStringifyTest, LeavesValuesUnencodedWhenDisabled) {
  nlohmann::json payload = nlohmann::json::object();
  payload["message"] = "a phrase with spaces";

  qs::StringifyOptions options;
  options.encode = false;

  const auto encoded = qs::stringify(payload, options);
  EXPECT_EQ(encoded, "message=a phrase with spaces");
}

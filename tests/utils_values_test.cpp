#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/utils/values.hpp"

using openai::utils::coerce_boolean;
using openai::utils::coerce_float;
using openai::utils::coerce_integer;
using openai::utils::has_own;
using openai::utils::is_absolute_url;
using openai::utils::is_empty_object;
using openai::utils::maybe_coerce_boolean;
using openai::utils::maybe_coerce_float;
using openai::utils::maybe_coerce_integer;
using openai::utils::maybe_object;
using openai::utils::safe_json;
using openai::utils::validate_positive_integer;

TEST(UtilsValuesTest, DetectsAbsoluteUrl) {
  EXPECT_TRUE(is_absolute_url("https://api.openai.com/v1"));
  EXPECT_TRUE(is_absolute_url("custom+scheme://example"));
  EXPECT_FALSE(is_absolute_url("/v1/models"));
  EXPECT_FALSE(is_absolute_url("ftp//missing-colon"));
}

TEST(UtilsValuesTest, ValidatesPositiveInteger) {
  EXPECT_EQ(validate_positive_integer("timeout", 123), 123);
  EXPECT_THROW(validate_positive_integer("timeout", -1), openai::OpenAIError);
}

TEST(UtilsValuesTest, SafeJsonParsesOrReturnsEmptyOptional) {
  auto parsed = safe_json("{\"key\":42}");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->at("key").get<int>(), 42);

  EXPECT_FALSE(safe_json("").has_value());
  EXPECT_FALSE(safe_json("not-json").has_value());
}

TEST(UtilsValuesTest, CoerceIntegerHandlesNumbersAndStrings) {
  nlohmann::json num = 10;
  EXPECT_EQ(coerce_integer(num), 10);

  nlohmann::json floating = 4.2;
  EXPECT_EQ(coerce_integer(floating), 4);

  nlohmann::json text = "15";
  EXPECT_EQ(coerce_integer(text), 15);

  nlohmann::json invalid = "abc";
  EXPECT_THROW(coerce_integer(invalid), openai::OpenAIError);
}

TEST(UtilsValuesTest, MaybeCoerceIntegerRespectsNull) {
  nlohmann::json value = nullptr;
  EXPECT_FALSE(maybe_coerce_integer(value).has_value());
}

TEST(UtilsValuesTest, CoerceFloatHandlesStrings) {
  nlohmann::json num = 2.5;
  EXPECT_DOUBLE_EQ(coerce_float(num), 2.5);

  nlohmann::json text = "3.14";
  EXPECT_DOUBLE_EQ(coerce_float(text), 3.14);

  nlohmann::json invalid = "abc";
  EXPECT_THROW(coerce_float(invalid), openai::OpenAIError);
}

TEST(UtilsValuesTest, CoerceBooleanMatchesSdkBehaviour) {
  EXPECT_TRUE(coerce_boolean(true));
  EXPECT_FALSE(coerce_boolean(false));
  EXPECT_TRUE(coerce_boolean(nlohmann::json("true")));
  EXPECT_FALSE(coerce_boolean(nlohmann::json("false")));
  EXPECT_TRUE(coerce_boolean(nlohmann::json::array()));
  EXPECT_FALSE(coerce_boolean(nullptr));
  EXPECT_TRUE(coerce_boolean(1));
  EXPECT_FALSE(coerce_boolean(0));
}

TEST(UtilsValuesTest, MaybeCoerceBooleanRespectsNull) {
  nlohmann::json value = nullptr;
  EXPECT_FALSE(maybe_coerce_boolean(value).has_value());
}

TEST(UtilsValuesTest, ObjectHelpersBehaveLikeTypeScript) {
  auto obj = nlohmann::json::object();
  obj["key"] = "value";

  EXPECT_TRUE(has_own(obj, "key"));
  EXPECT_FALSE(has_own(obj, "missing"));
  EXPECT_FALSE(is_empty_object(obj));
  EXPECT_TRUE(is_empty_object(nlohmann::json::object()));

  auto maybe = maybe_object("string");
  EXPECT_TRUE(maybe.is_object());
  EXPECT_TRUE(maybe.empty());
}

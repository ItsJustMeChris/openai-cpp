#include <gtest/gtest.h>

#include "openai/utils/env.hpp"

#include "support/env_guard.hpp"

using openai::utils::read_env;
using openai::utils::read_env_or;

namespace testing_utils = openai::testing;

TEST(UtilsEnvTest, ReturnsNulloptWhenUnset) {
  testing_utils::EnvVarGuard guard("OPENAI_CPP_TEST_ENV_UNSET", std::nullopt);
  EXPECT_FALSE(read_env("OPENAI_CPP_TEST_ENV_UNSET").has_value());
}

TEST(UtilsEnvTest, TrimsWhitespaceFromValues) {
  testing_utils::EnvVarGuard guard("OPENAI_CPP_TEST_ENV_TRIM", std::string("  value  "));
  auto value = read_env("OPENAI_CPP_TEST_ENV_TRIM");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, "value");
}

TEST(UtilsEnvTest, ReadEnvOrFallsBackWhenAbsent) {
  testing_utils::EnvVarGuard guard("OPENAI_CPP_TEST_ENV_OR", std::nullopt);
  EXPECT_EQ(read_env_or("OPENAI_CPP_TEST_ENV_OR", "fallback"), "fallback");
}


#include <gtest/gtest.h>

#include "openai/utils/uuid.hpp"

#include <regex>

TEST(UUIDTest, GeneratesValidUUIDv4Format) {
  const std::string id = openai::utils::uuid4();
  static const std::regex pattern(R"([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})");

  EXPECT_EQ(id.size(), 36u);
  EXPECT_TRUE(std::regex_match(id, pattern));
}

TEST(UUIDTest, GeneratesUniqueValues) {
  const std::string first = openai::utils::uuid4();
  const std::string second = openai::utils::uuid4();
  EXPECT_NE(first, second);
}


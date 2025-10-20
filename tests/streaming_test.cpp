#include <gtest/gtest.h>

#include "openai/streaming.hpp"

TEST(StreamingTest, ParsesBasicEvents) {
  const std::string payload =
      "event: message\n"
      "data: {\"id\":1}\n\n"
      "data: partial\n"
      "data: line\n\n";

  auto events = openai::parse_sse_stream(payload);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_TRUE(events[0].event.has_value());
  EXPECT_EQ(*events[0].event, "message");
  EXPECT_EQ(events[0].data, "{\"id\":1}");

  EXPECT_FALSE(events[1].event.has_value());
  EXPECT_EQ(events[1].data, "partial\nline");
}


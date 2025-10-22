#include <gtest/gtest.h>

#include "openai/streaming.hpp"

#include <cstring>

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

TEST(StreamingTest, IncrementalFeed) {
  openai::SSEParser parser;
  std::vector<openai::ServerSentEvent> events;

  auto chunk1 = parser.feed("data: part", std::strlen("data: part"));
  events.insert(events.end(), chunk1.begin(), chunk1.end());
  EXPECT_TRUE(events.empty());

  auto chunk2 = parser.feed("ial\n\n", std::strlen("ial\n\n"));
  events.insert(events.end(), chunk2.begin(), chunk2.end());
  auto remaining = parser.finalize();
  events.insert(events.end(), remaining.begin(), remaining.end());

  ASSERT_EQ(events.size(), 1u);
  EXPECT_FALSE(events[0].event.has_value());
  EXPECT_EQ(events[0].data, "partial");
}

TEST(StreamingEventStreamTest, DispatchesEventsAndCollectsHistory) {
  openai::SSEEventStream stream;
  stream.feed("data: one\n\n", std::strlen("data: one\n\n"));
  stream.feed("event: note\n", std::strlen("event: note\n"));
  stream.feed("data: two\n\n", std::strlen("data: two\n\n"));
  stream.finalize();

  const auto& events = stream.events();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_FALSE(events[0].event.has_value());
  EXPECT_EQ(events[0].data, "one");
  ASSERT_TRUE(events[1].event.has_value());
  EXPECT_EQ(*events[1].event, "note");
  EXPECT_EQ(events[1].data, "two");
}

TEST(StreamingEventStreamTest, HandlerCanStopStream) {
  int handled = 0;
  openai::SSEEventStream stream([&handled](const openai::ServerSentEvent& event) {
    ++handled;
    return event.data != "stop";
  });

  stream.feed("data: keep\n\n", std::strlen("data: keep\n\n"));
  stream.feed("data: stop\n\n", std::strlen("data: stop\n\n"));
  stream.feed("data: ignored\n\n", std::strlen("data: ignored\n\n"));
  stream.finalize();

  EXPECT_TRUE(stream.stopped());
  EXPECT_EQ(handled, 2);
  const auto& events = stream.events();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].data, "keep");
  EXPECT_EQ(events[1].data, "stop");
}

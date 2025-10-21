#include <gtest/gtest.h>

#include "openai/assistant_stream.hpp"
#include "openai/streaming.hpp"

#include <variant>
#include <vector>

TEST(AssistantStreamParserTest, EmitsTypedEvents) {
  using namespace openai;

  std::vector<AssistantStreamEvent> events;
  AssistantStreamParser parser([&](const AssistantStreamEvent& event) { events.push_back(event); });

  ServerSentEvent thread_event{.event = std::make_optional<std::string>("thread.created"),
                               .data = R"({"id":"thread_1","object":"thread","created_at":1,"metadata":{}})"};
  parser.feed(thread_event);

  ServerSentEvent run_event{.event = std::make_optional<std::string>("thread.run.created"),
                            .data = R"({"id":"run_1","assistant_id":"asst","created_at":1,"model":"gpt-4o","object":"thread.run","parallel_tool_calls":false,"status":"queued","thread_id":"thread_1","tools":[]})"};
  parser.feed(run_event);

  ServerSentEvent step_delta{.event = std::make_optional<std::string>("thread.run.step.delta"),
                             .data = R"({"id":"step_1","object":"thread.run.step.delta","delta":{"step_details":{"type":"tool_calls","tool_calls":[{"type":"function","index":0,"id":"call_1","function":{"name":"lookup","arguments":"{}","output":null}}]}}})"};
  parser.feed(step_delta);

  ServerSentEvent message_event{.event = std::make_optional<std::string>("thread.message.created"),
                                .data = R"({"id":"msg_1","object":"thread.message","created_at":1,"thread_id":"thread_1","role":"assistant","status":"completed","content":[],"attachments":[]})"};
  parser.feed(message_event);

  ServerSentEvent error_event{.event = std::make_optional<std::string>("error"),
                               .data = R"({"message":"stream failure"})"};
  parser.feed(error_event);

  ASSERT_EQ(events.size(), 5u);
  EXPECT_TRUE(std::holds_alternative<AssistantThreadEvent>(events[0]));
  EXPECT_EQ(std::get<AssistantThreadEvent>(events[0]).thread.id, "thread_1");

  EXPECT_TRUE(std::holds_alternative<AssistantRunEvent>(events[1]));
  EXPECT_EQ(std::get<AssistantRunEvent>(events[1]).run.id, "run_1");

  EXPECT_TRUE(std::holds_alternative<AssistantRunStepEvent>(events[2]));
  const auto& step = std::get<AssistantRunStepEvent>(events[2]).run_step;
  ASSERT_TRUE(step.details.tool_calls.size() == 1);
  EXPECT_EQ(step.details.tool_calls[0].function->name, "lookup");

  EXPECT_TRUE(std::holds_alternative<AssistantMessageEvent>(events[3]));
  EXPECT_EQ(std::get<AssistantMessageEvent>(events[3]).message.id, "msg_1");

  EXPECT_TRUE(std::holds_alternative<AssistantErrorEvent>(events[4]));
  EXPECT_EQ(std::get<AssistantErrorEvent>(events[4]).error, "stream failure");
}


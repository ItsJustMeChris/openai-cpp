#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "openai/responses.hpp"
#include "openai/streaming.hpp"

namespace openai {

class ResponseStreamSnapshot {
public:
  void ingest(const ResponseStreamEvent& event);

  const std::vector<ResponseStreamEvent>& events() const { return events_; }
  const std::optional<Response>& latest_response() const { return current_response_; }
  const std::optional<Response>& final_response() const { return completed_response_; }
  bool has_final_response() const { return completed_response_.has_value(); }

private:
  Response& ensure_response();
  ResponseOutputItem* find_output_item(const std::string& item_id, int output_index);
  void rebuild_item_index();
  void rebuild_messages();
  void refresh_message_segments(ResponseOutputMessage& message);
  void apply_message_part(ResponseOutputMessage& message, const ResponseContentPartAddedEvent& part);
  void apply_reasoning_part(ResponseReasoningItemDetails& reasoning, const ResponseContentPartAddedEvent& part);
  void handle_created(const ResponseCreatedEvent& created);
  void handle_completed(const ResponseCompletedEvent& completed);
  void handle_output_item_added(const ResponseOutputItemAddedEvent& added);
  void handle_output_item_done(const ResponseOutputItemDoneEvent& done);
  void handle_content_part_added(const ResponseContentPartAddedEvent& added);
  void handle_content_part_done(const ResponseContentPartDoneEvent& done);
  void handle_text_delta(const ResponseTextDeltaEvent& delta);
  void handle_text_done(const ResponseTextDoneEvent& done);
  void handle_reasoning_text_delta(const ResponseReasoningTextDeltaEvent& delta);
  void handle_reasoning_text_done(const ResponseReasoningTextDoneEvent& done);
  void handle_function_arguments_delta(const ResponseFunctionCallArgumentsDeltaEvent& delta);
  void handle_function_arguments_done(const ResponseFunctionCallArgumentsDoneEvent& done);

  std::vector<ResponseStreamEvent> events_;
  std::optional<Response> current_response_;
  std::optional<Response> completed_response_;
  std::unordered_map<std::string, std::size_t> item_index_by_id_;
};

class ResponseStream {
public:
  ResponseStream() = default;
  ResponseStream(std::vector<ServerSentEvent> raw_events,
                 std::vector<ResponseStreamEvent> typed_events,
                 ResponseStreamSnapshot snapshot);

  const std::vector<ServerSentEvent>& raw_events() const { return raw_events_; }
  const std::vector<ResponseStreamEvent>& events() const { return typed_events_; }
  const ResponseStreamSnapshot& snapshot() const { return snapshot_; }

  const std::optional<Response>& final_response() const { return snapshot_.final_response(); }
  bool has_final_response() const { return snapshot_.has_final_response(); }

private:
  std::vector<ServerSentEvent> raw_events_;
  std::vector<ResponseStreamEvent> typed_events_;
  ResponseStreamSnapshot snapshot_;
};

}  // namespace openai

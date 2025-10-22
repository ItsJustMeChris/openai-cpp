#include "openai/streaming.hpp"

#include <algorithm>
#include <sstream>

namespace openai {
namespace {

void trim_carriage_return(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

}  // namespace

std::vector<ServerSentEvent> SSEParser::feed(const char* data, std::size_t size) {
  buffer_.append(data, size);
  return extract_events();
}

std::vector<ServerSentEvent> SSEParser::finalize() {
  buffer_.append("\n");
  auto events = extract_events();
  if (!current_.raw_lines.empty() || !current_.data.empty() || current_.event.has_value()) {
    events.push_back(current_);
    current_ = ServerSentEvent{};
  }
  buffer_.clear();
  return events;
}

std::vector<ServerSentEvent> SSEParser::extract_events() {
  std::vector<ServerSentEvent> events;
  std::size_t start = 0;

  while (true) {
    auto newline_pos = buffer_.find('\n', start);
    if (newline_pos == std::string::npos) {
      break;
    }

    std::string line = buffer_.substr(start, newline_pos - start);
    trim_carriage_return(line);
    start = newline_pos + 1;
    process_line(line, events);
  }

  buffer_.erase(0, start);

  return events;
}

void SSEParser::process_line(const std::string& line, std::vector<ServerSentEvent>& events) {
  if (line.empty()) {
    if (!current_.raw_lines.empty() || !current_.data.empty() || current_.event.has_value()) {
      events.push_back(current_);
      current_ = ServerSentEvent{};
    }
    return;
  }

  if (!line.empty() && line.front() == ':') {
    return;
  }

  current_.raw_lines.push_back(line);

  auto colon_pos = line.find(':');
  std::string field = colon_pos == std::string::npos ? line : line.substr(0, colon_pos);
  std::string value = colon_pos == std::string::npos ? std::string() : line.substr(colon_pos + 1);
  if (!value.empty() && value.front() == ' ') {
    value.erase(value.begin());
  }

  if (field == "event") {
    current_.event = value;
  } else if (field == "data") {
    if (!current_.data.empty()) {
      current_.data.push_back('\n');
    }
    current_.data += value;
  }
}

std::vector<ServerSentEvent> parse_sse_stream(const std::string& payload) {
  SSEParser parser;
  auto events = parser.feed(payload.data(), payload.size());
  auto remaining = parser.finalize();
  events.insert(events.end(), remaining.begin(), remaining.end());
  return events;
}

SSEEventStream::SSEEventStream(EventHandler handler) : handler_(std::move(handler)) {}

void SSEEventStream::feed(const char* data, std::size_t size) {
  if (stopped_) return;
  auto events = parser_.feed(data, size);
  dispatch_events(std::move(events));
}

void SSEEventStream::finalize() {
  if (stopped_) return;
  auto events = parser_.finalize();
  dispatch_events(std::move(events));
}

void SSEEventStream::stop() {
  stopped_ = true;
}

void SSEEventStream::dispatch_events(std::vector<ServerSentEvent>&& events) {
  if (events.empty()) return;
  for (const auto& event : events) {
    if (stopped_) break;
    events_.push_back(event);
    if (handler_) {
      const bool should_continue = handler_(event);
      if (!should_continue) {
        stopped_ = true;
      }
    }
  }
}

}  // namespace openai

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace openai {

struct ServerSentEvent {
  std::optional<std::string> event;
  std::string data;
  std::vector<std::string> raw_lines;
};

std::vector<ServerSentEvent> parse_sse_stream(const std::string& payload);

class SSEParser {
public:
  std::vector<ServerSentEvent> feed(const char* data, std::size_t size);
  std::vector<ServerSentEvent> finalize();

private:
  std::vector<ServerSentEvent> extract_events();
  void process_line(const std::string& line, std::vector<ServerSentEvent>& events);

  std::string buffer_;
  ServerSentEvent current_;
};

class SSEEventStream {
public:
  using EventHandler = std::function<bool(const ServerSentEvent&)>;

  explicit SSEEventStream(EventHandler handler = nullptr);

  void feed(const char* data, std::size_t size);
  void finalize();
  void stop();

  [[nodiscard]] bool stopped() const { return stopped_; }
  [[nodiscard]] const std::vector<ServerSentEvent>& events() const { return events_; }

private:
  void dispatch_events(std::vector<ServerSentEvent>&& events);

  SSEParser parser_;
  EventHandler handler_;
  std::vector<ServerSentEvent> events_;
  bool stopped_ = false;
};

}  // namespace openai

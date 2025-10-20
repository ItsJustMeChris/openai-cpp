#pragma once

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

}  // namespace openai

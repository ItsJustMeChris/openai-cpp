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

std::vector<ServerSentEvent> parse_sse_stream(const std::string& payload) {
  std::vector<ServerSentEvent> events;
  std::istringstream input(payload);
  std::string line;
  ServerSentEvent current;

  auto flush_current = [&]() {
    if (!current.raw_lines.empty()) {
      events.push_back(current);
      current = ServerSentEvent{};
    }
  };

  while (std::getline(input, line)) {
    trim_carriage_return(line);
    if (line.empty()) {
      flush_current();
      continue;
    }

    current.raw_lines.push_back(line);

    auto colon_pos = line.find(':');
    std::string field = colon_pos == std::string::npos ? line : line.substr(0, colon_pos);
    std::string value = colon_pos == std::string::npos ? std::string() : line.substr(colon_pos + 1);
    if (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }

    if (field == "event") {
      current.event = value;
    } else if (field == "data") {
      if (!current.data.empty()) {
        current.data.push_back('\n');
      }
      current.data += value;
    }
  }

  flush_current();

  return events;
}

}  // namespace openai


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

}  // namespace openai


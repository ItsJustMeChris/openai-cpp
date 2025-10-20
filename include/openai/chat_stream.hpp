#pragma once

#include <optional>
#include <string>
#include <vector>

#include "openai/streaming.hpp"

namespace openai {

struct ChatStreamChunk {
  ServerSentEvent event;
};

}  // namespace openai


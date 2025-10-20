#pragma once

#include <optional>
#include <string>
#include <vector>

namespace openai {

struct CompletionUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
};

struct CompletionChoice {
  int index = 0;
  std::string text;
  std::string finish_reason;
};

struct Completion {
  std::string id;
  std::string object;
  int created = 0;
  std::string model;
  std::vector<CompletionChoice> choices;
  std::optional<CompletionUsage> usage;
};

struct CompletionRequest {
  std::string model;
  std::string prompt;
  std::optional<int> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> n;
  std::optional<std::vector<std::string>> stop;
  std::optional<bool> stream;
};

}  // namespace openai


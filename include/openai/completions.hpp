#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace openai {

struct CompletionChoiceLogprobs {
  std::optional<std::vector<int>> text_offset;
  std::optional<std::vector<double>> token_logprobs;
  std::optional<std::vector<std::string>> tokens;
  std::optional<std::vector<std::map<std::string, double>>> top_logprobs;
};

struct CompletionUsageCompletionTokensDetails {
  std::optional<int> accepted_prediction_tokens;
  std::optional<int> audio_tokens;
  std::optional<int> reasoning_tokens;
  std::optional<int> rejected_prediction_tokens;
};

struct CompletionUsagePromptTokensDetails {
  std::optional<int> audio_tokens;
  std::optional<int> cached_tokens;
};

struct CompletionUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  std::optional<CompletionUsageCompletionTokensDetails> completion_tokens_details;
  std::optional<CompletionUsagePromptTokensDetails> prompt_tokens_details;
};

struct CompletionChoice {
  int index = 0;
  std::optional<std::string> finish_reason;
  std::string text;
  std::optional<CompletionChoiceLogprobs> logprobs;
};

struct Completion {
  std::string id;
  std::string object;
  std::int64_t created = 0;
  std::string model;
  std::optional<std::string> system_fingerprint;
  std::vector<CompletionChoice> choices;
  std::optional<CompletionUsage> usage;
};

struct CompletionStreamOptions {
  std::optional<bool> include_obfuscation;
  std::optional<bool> include_usage;
};

struct CompletionRequest {
  using Prompt = std::variant<std::string, std::vector<std::string>, std::vector<int>, std::vector<std::vector<int>>>;
  using StopSequences = std::variant<std::string, std::vector<std::string>>;

  std::string model;
  std::optional<Prompt> prompt;
  std::optional<int> best_of;
  std::optional<bool> echo;
  std::optional<double> frequency_penalty;
  std::optional<std::map<std::string, double>> logit_bias;
  std::optional<int> logprobs;
  std::optional<int> max_tokens;
  std::optional<int> n;
  std::optional<double> presence_penalty;
  std::optional<std::int64_t> seed;
  std::optional<StopSequences> stop;
  std::optional<bool> stream;
  std::optional<CompletionStreamOptions> stream_options;
  std::optional<std::string> suffix;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> top_logprobs;
  std::optional<std::string> user;
};

}  // namespace openai

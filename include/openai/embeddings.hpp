#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace openai {

struct EmbeddingUsage {
  int prompt_tokens = 0;
  int total_tokens = 0;
};

struct Embedding {
  using EmbeddingData = std::variant<std::vector<float>, std::string>;

  EmbeddingData embedding;
  int index = 0;
  std::string object;
};

struct CreateEmbeddingResponse {
  std::vector<Embedding> data;
  std::string model;
  std::string object;
  std::optional<EmbeddingUsage> usage;
};

struct EmbeddingRequest {
  using Input = std::variant<
      std::string,
      std::vector<std::string>,
      std::vector<int>,
      std::vector<std::vector<int>>,
      std::vector<float>,
      std::vector<std::vector<float>>,
      std::vector<double>,
      std::vector<std::vector<double>>>;

  Input input;
  std::string model;
  std::optional<int> dimensions;
  std::optional<std::string> encoding_format;  // "float" or "base64"
  std::optional<std::string> user;
};

}  // namespace openai


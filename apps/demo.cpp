#include "openai/client.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  const char* api_key = std::getenv("OPENAI_API_KEY");
  if (!api_key) {
    std::cerr << "OPENAI_API_KEY environment variable must be set\n";
    return 1;
  }

  try {
    openai::ClientOptions options;
    options.api_key = api_key;

    openai::OpenAIClient client(options);

    openai::CompletionRequest request;
    request.model = "gpt-5-mini";
    request.prompt = "Say hello from C++";
    request.max_tokens = 32;

    auto completion = client.completions().create(request);

    for (const auto& choice : completion.choices) {
      std::cout << choice.text << std::endl;
    }
  } catch (const openai::OpenAIError& error) {
    std::cerr << "OpenAI error: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}


#include "openai/client.hpp"
#include "openai/responses.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

int main()
{
  const char *api_key = std::getenv("OPENAI_API_KEY");
  std::cout << "Got api key: " << (api_key ? "yes" : "no") << std::endl;
  if (!api_key)
  {
    std::cerr << "OPENAI_API_KEY environment variable must be set\n";
    return 1;
  }

  try
  {
    openai::ClientOptions options;
    options.api_key = api_key;

    std::cout << "Creating OpenAI client...\n";

    openai::OpenAIClient client(options);

    // openai::CompletionRequest request;
    // request.model = "gpt-3.5-turbo-instruct";
    // request.prompt = "Say hello from C++";
    // request.max_tokens = 32;
    // std::cout << "Creating completion...\n";

    // auto completion = client.completions().create(request);
    // std::cout << "Completion choices:\n";
    // for (const auto &choice : completion.choices)
    // {
    //   std::cout << choice.text << std::endl;
    // }

    // std::cout << "Creating response...\n";
    // auto response = client.responses().create(openai::ResponseRequest{
    //     .model = "gpt-5-nano",
    //     .input = {
    //         openai::ResponseInputItem{
    //             .type = openai::ResponseInputItem::Type::Message,
    //             .message = openai::ResponseInputMessage{
    //                 .role = "user",
    //                 .content = {openai::ResponseInputContent{
    //                     .type = openai::ResponseInputContent::Type::Text,
    //                     .text = "Say hello from C++ using the Responses API"}}}}}});

    std::cout << "Streaming response...\n";
    openai::ResponseRequest stream_request;
    stream_request.model = "gpt-4o-mini";
    stream_request.input = {
        openai::ResponseInputItem{
            .type = openai::ResponseInputItem::Type::Message,
            .message = openai::ResponseInputMessage{
                .role = "user",
                .content = {openai::ResponseInputContent{
                    .type = openai::ResponseInputContent::Type::Text,
                    .text = "Stream a long story about the history of C++"}}}}};

    std::string streamed_text;
    std::size_t chunk_index = 0;
    auto start = std::chrono::steady_clock::now();

    client.responses().create_stream(
        stream_request,
        [&](const openai::ResponseStreamEvent &event)
        {
          if (event.text_delta)
          {
            streamed_text += event.text_delta->delta;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            std::cout << "Chunk " << chunk_index++ << " at " << elapsed
                      << " ms: " << event.text_delta->delta << std::endl;
          }
          return true;
        });

    std::cout << "Stream complete with output: " << streamed_text << std::endl;
  }
  catch (const openai::OpenAIError &error)
  {
    std::cerr << "OpenAI error: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}

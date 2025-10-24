#include "openai/client.hpp"
#include "openai/responses.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

openai::ResponseInputItem make_message(const std::string& role, const std::string& text)
{
  openai::ResponseInputItem item;
  item.type = openai::ResponseInputItem::Type::Message;
  item.message.role = role;

  openai::ResponseInputContent content;
  content.type = openai::ResponseInputContent::Type::Text;
  content.text = text;
  item.message.content.push_back(std::move(content));

  return item;
}

}  // namespace

int main()
{
  const char* api_key = std::getenv("OPENAI_API_KEY");
  if (!api_key)
  {
    std::cerr << "OPENAI_API_KEY environment variable must be set\n";
    return 1;
  }

  try
  {
    openai::ClientOptions options;
    options.api_key = api_key;

    openai::OpenAIClient client(options);

    std::cout << "Interactive streaming chat demo\n"
              << "Type 'exit' or 'quit' to stop.\n";

    const std::string system_instructions =
        "You are a helpful assistant speaking to a user from a C++ demo app.";

    std::optional<std::string> conversation_id;
    std::optional<std::string> previous_response_id;

    for (;;)
    {
      std::cout << "\nYou> " << std::flush;
      std::string user_input;

      if (!std::getline(std::cin, user_input))
      {
        std::cout << "\nEnd of input, exiting.\n";
        break;
      }

      if (user_input == "exit" || user_input == "quit")
      {
        std::cout << "Goodbye!\n";
        break;
      }

      if (user_input.empty())
      {
        continue;
      }

      openai::ResponseRequest request;
      request.model = "gpt-4o-mini";
      request.instructions = system_instructions;
      request.input = {make_message("user", user_input)};
      if (conversation_id)
      {
        request.conversation_id = *conversation_id;
      }
      if (previous_response_id)
      {
        request.previous_response_id = *previous_response_id;
      }

      std::cout << "Assistant> " << std::flush;

      std::string streamed_text;
      std::optional<std::string> final_text;
      bool saw_error = false;
      std::optional<openai::Response> completed_response;

      client.responses().stream(
          request,
          [&](const openai::ResponseStreamEvent& event)
          {
            if (event.error)
            {
              std::cerr << "\n[stream error] " << event.error->message << std::endl;
              saw_error = true;
              return false;
            }

            if (event.text_delta && event.text_delta->output_index == 0)
            {
              const std::string& delta = event.text_delta->delta;
              streamed_text += delta;
              std::cout << delta << std::flush;
            }

            if (event.text_done && event.text_done->output_index == 0)
            {
              final_text = event.text_done->text;
            }

            if (event.completed)
            {
              completed_response = event.completed->response;
            }

            return true;
          });

      if (saw_error)
      {
        std::cout << "\nEncountered an error. Please try again." << std::endl;
        continue;
      }

      std::string assistant_text = final_text.value_or(streamed_text);

      if (streamed_text.empty())
      {
        if (!assistant_text.empty())
        {
          std::cout << assistant_text << std::flush;
        }
        else
        {
          std::cout << "[No text returned]" << std::flush;
        }
      }

      std::cout << std::endl;

      if (assistant_text.empty())
      {
        assistant_text = "[No text returned]";
      }

      if (completed_response)
      {
        if (completed_response->conversation)
        {
          conversation_id = completed_response->conversation->id;
        }
        previous_response_id = completed_response->id;
      }
    }
  }
  catch (const openai::APIError& error)
  {
    std::cerr << "OpenAI API error (" << error.status_code() << "): " << error.what() << std::endl;
    if (!error.error_body().empty())
    {
      std::cerr << error.error_body().dump(2) << std::endl;
    }
    return 1;
  }
  catch (const openai::OpenAIError& error)
  {
    std::cerr << "OpenAI error: " << error.what() << std::endl;
    return 1;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Unexpected error: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}

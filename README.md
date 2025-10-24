# openai-cpp

`openai-cpp` is a modern, fully native C++17 client for the OpenAI API. It mirrors the ergonomics of the official TypeScript SDK (`client.responses().create()`, `client.chat().completions().stream()`, etc.) while taking advantage of strong typing, RAII-friendly resource management, and portable CMake tooling.

> ℹ️ The latest stable release is `1.0.0` (tag `1.0.0`). The library remains under active development, so expect surface area to evolve as the upstream OpenAI platform ships new capabilities.

## Highlights

- Strongly typed request/response models covering Responses, Chat Completions, Assistants, Vector Stores, Files, Images, Audio, Moderations, Embeddings, and more.
- First-class streaming helpers for the Responses API, Chat Completions, and Assistants (Server-Sent Events with incremental callbacks).
- Pluggable HTTP transport via an abstract `HttpClient` interface (libcurl backend provided out of the box).
- Consistent error model (`openai::APIError`, `openai::OpenAIError`) with rich metadata.
- Cross-platform CMake project with demo binaries (`openai-cli`, `openai-chat`) and optional GoogleTest suite.

## Supported resources (current snapshot)

- Core: `OpenAIClient`, configurable `ClientOptions`, request-level overrides (`RequestOptions`).
- Text & multimodal: `ResponsesResource`, `ChatResource`, `CompletionsResource`, `ModerationsResource`, `EmbeddingsResource`.
- Media: `ImagesResource`, `AudioResource`, `VideosResource`.
- File workflows: `FilesResource`, `UploadsResource`, `VectorStoresResource`, `BatchesResource`.
- Assistants beta: `AssistantsResource`, `ThreadsResource`, `ThreadMessagesResource`, `RunsResource`, `RunStepsResource`, SSE parsers.
- Automation & tooling: `GradersResource`, `ConversationsResource`, `ContainersResource`, `WebhooksResource`, `EvalsResource`.

Consult `docs/ARCHITECTURE.md` for a deeper look at the module layout and parity goals.

## Prerequisites

- A C++17-compatible compiler (Clang 12+, GCC 10+, MSVC 2019+).
- CMake 3.16 or newer.
- libcurl (detected via `find_package(CURL)`).
- An OpenAI API key with access to the models you intend to call.

## Build the library and demos

```bash
cmake -S . -B build
cmake --build build
```

Enable unit tests (requires GoogleTest download during configure):

```bash
cmake -S . -B build -DOPENAI_CPP_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Demo binaries live under `apps/` and end up in `build/apps/`. For example, to run the streaming chat demo:

```bash
export OPENAI_API_KEY="sk-your-key"
./build/apps/openai-chat
```

## Adding openai-cpp to your project

### Option 1: Git submodule / FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  openai-cpp
  GIT_REPOSITORY https://github.com/ItsJustMeChris/openai-cpp.git
  GIT_TAG 1.0.0
)
FetchContent_MakeAvailable(openai-cpp)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE openai::openai)
```

Pinning the `1.0.0` tag keeps your build on the released surface area. Swap to `main` if you need the latest commits before the next release is published.

### Option 2: Manual clone + add_subdirectory

```cmake
add_subdirectory(external/openai-cpp)
target_link_libraries(my_app PRIVATE openai::openai)
```

Both approaches export the `openai::openai` target with public headers under `include/`.

## Usage

Include the single umbrella header when you want the full client surface in one shot:

```cpp
#include "openai.hpp"
```

### Initialize a client

```cpp
#include "openai.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

int main() {
  const char* api_key = std::getenv("OPENAI_API_KEY");
  if (!api_key) {
    std::cerr << "Set OPENAI_API_KEY in your environment\n";
    return 1;
  }

  openai::ClientOptions options;
  options.api_key = api_key;
  // Optional tuning
  options.timeout = std::chrono::seconds(60);
  options.default_headers["OpenAI-Beta"] = "assistants=v2";

  openai::OpenAIClient client(options);

  // ...
}
```

### Respond to a prompt with the Responses API

```cpp
#include "openai.hpp"
#include <iostream>

openai::Response run_prompt(openai::OpenAIClient& client) {
  openai::ResponseRequest request;
  request.model = "gpt-4o-mini";

  openai::ResponseInputContent content;
  content.type = openai::ResponseInputContent::Type::Text;
  content.text = "Give me three facts about the Galápagos Islands.";

  openai::ResponseInputMessage message;
  message.role = "user";
  message.content.push_back(content);

  openai::ResponseInputItem item;
  item.type = openai::ResponseInputItem::Type::Message;
  item.message = message;

  request.input.push_back(item);

  auto response = client.responses().create(request);
  std::cout << response.output_text << "\n";
  return response;
}
```

### Stream output incrementally

```cpp
#include "openai.hpp"
#include <iostream>

void stream_story(openai::OpenAIClient& client) {
  openai::ResponseRequest request;
  request.model = "gpt-4o-mini";

  openai::ResponseInputContent content;
  content.type = openai::ResponseInputContent::Type::Text;
  content.text = "Stream a 3-sentence bedtime story about C++ templates.";

  openai::ResponseInputMessage message;
  message.role = "user";
  message.content.push_back(content);

  openai::ResponseInputItem item;
  item.type = openai::ResponseInputItem::Type::Message;
  item.message = message;

  request.input.push_back(item);

  client.responses().stream(
      request,
      [&](const openai::ResponseStreamEvent& event) {
        if (event.text_delta && event.text_delta->output_index == 0) {
          std::cout << event.text_delta->delta << std::flush;
        }
        if (event.error) {
          std::cerr << "\n[stream error] " << event.error->message << std::endl;
          return false;  // stop streaming
        }
        return true;
      });

  std::cout << std::endl;
}
```

### Other quick starts

- **Embeddings**:

  ```cpp
  openai::EmbeddingRequest embeddings_request;
  embeddings_request.model = "text-embedding-3-small";
  embeddings_request.input = {"Hello world"};
  auto embedding = client.embeddings().create(embeddings_request);
  ```
- **Moderations**: Build a `openai::ModerationRequest` and pass it to `client.moderations().create(request);`
- **Assistants / Threads**: Combine `client.assistants()`, `client.threads()`, and `client.runs()` to orchestrate assistant conversations (see `apps/chat_demo.cpp`).
- **Uploads & vector stores**: Use `client.uploads().create()` then attach file IDs to vector store operations.

All request/response structs live under `include/openai/*.hpp`. Fields are `std::optional` when they mirror nullable JSON properties.

### Error handling

Most operations throw on failure:

- `openai::APIError`: the API returned an error status (inspect `status_code()`, `error_body()`).
- `openai::APIConnectionError` / `openai::APIConnectionTimeoutError`: the HTTP backend failed before reaching the API or timed out.
- `openai::APIUserAbortError`: you cancelled an in-flight streaming call by returning `false` from a callback.
- `openai::OpenAIError`: base class for unexpected library issues.

Wrap calls in a `try`/`catch` block when running long-lived processes.

## Configuration tips

- **Headers and query params**: Supply per-request mutations through `RequestOptions`.

  ```cpp
  openai::RequestOptions request_options;
  request_options.headers["OpenAI-Beta"] = "assistants=v2";
  client.responses().create(request, request_options);
  ```
- **Proxying / custom HTTP**: Implement `openai::HttpClient` and pass it to the `OpenAIClient` constructor to integrate with bespoke transports.
- **Azure OpenAI**: Toggle `options.azure_deployment_routing` or set `options.azure_deployment_name` and `options.base_url` to target Azure-hosted deployments (see `include/openai/azure.hpp`).
- **Logging**: Configure `options.log_level` and `options.logger` for structured diagnostics while debugging.

## Documentation and contributing

- Architecture: `docs/ARCHITECTURE.md`
- Testing strategy: `docs/TESTING.md`
- Responses port status: `docs/RESPONSES_PORT_PLAN.md`
- Assistant parity tracking: `docs/PARITY_CHECKLIST.md`

Issues and pull requests are welcome. Please run the formatting checks, applicable unit tests, and the demos touching the areas you modify before opening a PR.

## Acknowledgements

The library vendors [`nlohmann::json`](https://github.com/nlohmann/json) under the MIT license (see `external/nlohmann/LICENSE.MIT`). Many type definitions and surface signatures are modeled after the official OpenAI TypeScript SDK for familiarity across languages.

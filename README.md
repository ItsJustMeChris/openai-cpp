# openai-cpp (WIP)

This repository houses an in-progress native C++ port of the official `openai-node` SDK.  The goal is to offer a modern C++ client with familiar ergonomics (`client.completions().create()` etc.) while providing strong typing and portable tooling via CMake.

## Current scope

- ✅ Core client scaffolding (`OpenAIClient`) with configurable API key, base URL, and default headers
- ✅ libcurl-based HTTP backend hidden behind an extensible `HttpClient` interface
- ✅ JSON serialization via the header-only [`nlohmann::json`](https://github.com/nlohmann/json) library (vendored locally)
- ✅ `CompletionsResource` with strongly typed request/response models
- ✅ `ModelsResource` supporting retrieve/list/delete operations
- ✅ `EmbeddingsResource` with default base64 decoding to float vectors
- ✅ `ChatResource` with non-streaming `/chat/completions` support
- ✅ `ModerationsResource` for text/image classification
- ✅ `ResponsesResource` (non-streaming create/retrieve/list/delete/cancel)
- ✅ `FilesResource` (list/retrieve/create/delete/content for basic uploads)
- 🚧 Additional endpoints (Chat, Responses, Files, etc.) to be implemented
- 🚧 Streaming helpers, pagination helpers, and beta resources still pending

## Building

```bash
cmake -S . -B build
cmake --build build

# Enable unit tests (requires GoogleTest and network access to fetch by default)
cmake -S . -B build -DOPENAI_CPP_BUILD_TESTS=ON
cmake --build build
```

The `openai-cli` demo binary shows basic usage. Before running it, export an API key:

```bash
export OPENAI_API_KEY="sk-..."
./build/apps/openai-cli
```

Network calls are disabled in automated tests by default; consider providing a mock `HttpClient` when writing unit tests. The TypeScript SDK runs its suite against a Prism-powered mock server on `127.0.0.1:4010`; mirroring that setup (or faking the `HttpClient`) will keep parity as the surface grows. See `docs/TESTING.md` for the evolving test plan.

The vendored copy of `nlohmann::json` is licensed under MIT; see `external/nlohmann/LICENSE.MIT`.

## Adding new resources

1. Create a new header (e.g. `include/openai/chat.hpp`) describing request/response models using `std::optional` and `std::vector`.
2. Implement the resource in `src/` mirroring the TypeScript module. Use `perform_request()` to inherit shared authentication and error handling.
3. Update `CMakeLists.txt` to include the new source file and extend documentation as needed.

Refer to `docs/ARCHITECTURE.md` for the mapping between the original TypeScript layers and their C++ counterparts.

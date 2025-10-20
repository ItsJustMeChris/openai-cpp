# Testing Strategy

The canonical Node.js SDK drives almost its entire test suite through a Prism mock server bound to `127.0.0.1:4010`. Each resource test points the client at that server and asserts on both the raw `Response` and parsed payloads.

For the C++ port we plan to replicate that experience while keeping fast, hermetic unit tests:

- **Mock HTTP transport** – The `tests/support/mock_http_client.hpp` helper can stand in for the default libcurl implementation and replay pre-baked `HttpResponse` objects. This unlocks deterministic tests for request serialization, header merging, error propagation, and JSON decoding without spinning up Prism.
- **Prism integration (optional)** – For higher-level integration coverage we can reuse the OpenAPI fixture from `openai-node` and start the Prism mock during CI. Tests can wire the client with `ClientOptions{ .base_url = "http://127.0.0.1:4010" }` to exercise end-to-end behavior.
- **Test framework** – GoogleTest support is wired behind the `OPENAI_CPP_BUILD_TESTS` CMake option. Enabling it uses `FetchContent` to download GoogleTest; in restricted network environments you may need to cache the dependency manually.

Recommended next steps:

1. Add a `tests/CMakeLists.txt` enabling opt-in test builds and pulling in GoogleTest via `FetchContent`.
2. Port representative cases from `openai-node/tests/api-resources` to verify request shapes (e.g., embeddings base64 decode, chat tool calls).
3. Provide CI instructions for launching Prism locally, mirroring `scripts/test` in the Node repo.

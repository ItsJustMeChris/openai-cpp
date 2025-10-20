# Test Infrastructure TODOs

- [ ] Wire GoogleTest via CMake `FetchContent` and provide an option to build tests.
- [ ] Port initial unit tests using `MockHttpClient` covering request serialization (completions, embeddings, responses) and error propagation.
- [ ] Document Prism mock workflow in CI (start server, point client base URL at `127.0.0.1:4010`).
- [ ] Consider adding a `MOCK_RESPONSE` environment toggle to skip network logic in examples.

# openai-cpp Architecture

This project begins a native C++ translation of the official `openai-node` SDK.
The TypeScript client organises functionality into three broad layers:

1. **Client configuration (`src/client.ts`)** – handles base URL selection, global headers, retries, and instantiates per-resource helpers such as `Completions` and `Chat`.
2. **Core HTTP helpers (`src/internal` & `src/core`)** – provide a thin fetch abstraction, stream helpers, and JSON serialisation.
3. **Resource wrappers (`src/resources`)** – expose strongly typed methods per REST endpoint (e.g. `client.completions.create`).

The C++ port mirrors this structure with lightweight equivalents:

```
include/
  openai/
    client.hpp         // Top level client facade and shared options
    http_client.hpp    // Interface + default libcurl HTTP backend
    completions.hpp    // Models for the /completions endpoint
src/
  client.cpp           // JSON encoding/decoding + request orchestration
  http_client.cpp      // libcurl implementation of HttpClient
external/
  nlohmann/json.hpp    // Header-only JSON dependency (MIT licensed)
apps/
  demo.cpp             // Example CLI usage of the client
```

Key translation decisions:

- The TypeScript `fetch` abstraction becomes a `HttpClient` interface with a default libcurl-based implementation (`CurlHttpClient`).
- Resource classes (starting with `CompletionsResource`) expose idiomatic C++ methods that forward to the shared `OpenAIClient::perform_request` helper.
- Request/response payloads use `nlohmann::json` for serialization which closely matches the JSON-first nature of the TypeScript SDK.
- Optional resource parameters map to `std::optional` fields; repeated values use `std::vector`.

This scaffold currently supports `client.completions().create()`, the `client.models()` surface (retrieve/list/delete), and `client.embeddings().create()` with base64 decoding parity. Additional endpoints can be added by introducing new resource headers mirroring the corresponding TypeScript modules and reusing the `perform_request` helper for consistent authentication and error handling.

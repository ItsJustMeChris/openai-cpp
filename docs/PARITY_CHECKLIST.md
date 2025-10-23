# Parity Checklist

Tracking progress toward 1:1 feature coverage with the official `openai-node` TypeScript SDK.

## Core Infrastructure
- [x] Configuration object (`ClientOptions`) with auth headers and base URL
- [x] HTTP transport abstraction (`HttpClient`) with default libcurl backend
- [x] Retry and backoff policy parity (`maxRetries`, retry-after handling)
- [x] Request option merging (`RequestOptions`, idempotency, query parameters) – headers/query support implemented
- [x] Platform detection + header helpers (User-Agent parity)
- [x] Streaming helpers (Server-Sent Events, chunk decoding)
- [x] Pagination abstractions (`AbstractPage`, cursor/page responses)
- [x] Multipart uploads (`Uploads` helper, file chunking strategies)
- [x] Error hierarchy mirroring `core/error.ts`
- [x] Logging hooks and middleware support

## Internal Utilities
- [x] Query string builder (`internal/qs`)
- [x] Env/platform shims (`internal/shims`, `detect-platform`)
- [x] Value validators (`validatePositiveInteger`, `isAbsoluteURL`, etc.)
- [x] Sleep/backoff utilities
- [x] UUID helpers
- [x] File conversion utilities (`to-file`)

## Client Facade
- [x] `client.completions().create`
- [ ] Other core resource registration (lazy instantiation, type-safe surfaces)
- [ ] Streaming return types (`APIPromise`, `Stream` equivalents)
- [x] Upload helpers and binary response support

## REST Resources
- [x] Completions (`/completions`)
- [x] Chat completions (basic create + streaming helper)
- [ ] Chat completions advanced features (stored completions, tool runners, messages API)
- [x] Responses (non-streaming create/retrieve/list/cancel/delete)
- [x] Responses advanced features (streaming, input items, tool runners) – SSE event parsing available
- [x] Models (retrieve/list/delete; pagination helpers pending)
- [x] Embeddings (default base64 decode, float/base64 variants)
- [x] Moderations (basic create)
- [x] Files (list/retrieve/create/delete/content; advanced upload helpers pending)
- [x] Images (generate/edit/variation basic support; streaming events pending)
- [x] Audio transcriptions (create)
- [x] Audio translations (create)
- [x] Audio speech (create)
- [x] Batches
- [x] Vector stores (basic CRUD)
- [x] Vector store files/batches/search
- [x] Assistants (beta create/retrieve/update/list/delete)
- [x] Threads (beta create/retrieve/update/delete)
- [x] Assistants messages (beta thread messages)
- [x] Assistants runs (beta create/retrieve/update/list/cancel/submit outputs)
- [x] Run steps (beta list/retrieve)
- [x] Assistants streaming events (typed parser)
- [x] Run streaming & polling helpers (`create_stream`, `stream`, `submit_tool_outputs_stream`, `poll`, `create_and_run_poll`)
- [x] Thread create-and-run helpers (stream/poll)
- [x] Assistants tool runner actions & automation helpers
- [x] Fine-tuning
- [x] Moderations
- [x] Conversations
- [x] Containers
- [x] Videos
- [x] Webhooks
- [ ] Beta features (`beta`, `graders`, `realtime`, etc.)
- [ ] Azure OpenAI compatibility layer

## Tooling & Tests
- [ ] Unit tests with mock HTTP client
  - Mock transport scaffold in `tests/support/mock_http_client.hpp`
- [ ] Integration test harness (opt-in, network-gated)
- [ ] CI configuration
- [ ] Packaging/distribution strategy (CMake install, pkg-config)

## Documentation
- [x] Architecture overview
- [ ] Usage guide covering all resources
- [ ] Migration notes vs. TypeScript SDK
- [ ] Generated API reference parity

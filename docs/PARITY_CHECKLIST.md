# Parity Checklist

Tracking progress toward 1:1 feature coverage with the official `openai-node` TypeScript SDK.

## Core Infrastructure
- [x] Configuration object (`ClientOptions`) with auth headers and base URL
- [x] HTTP transport abstraction (`HttpClient`) with default libcurl backend
- [ ] Retry and backoff policy parity (`maxRetries`, retry-after handling)
- [ ] Request option merging (`RequestOptions`, idempotency, query parameters)
- [ ] Platform detection + header helpers (User-Agent parity)
- [ ] Streaming helpers (Server-Sent Events, chunk decoding)
- [ ] Pagination abstractions (`AbstractPage`, cursor/page responses)
- [ ] Multipart uploads (`Uploads` helper, file chunking strategies)
- [ ] Error hierarchy mirroring `core/error.ts`
- [ ] Logging hooks and middleware support

## Internal Utilities
- [ ] Query string builder (`internal/qs`)
- [ ] Env/platform shims (`internal/shims`, `detect-platform`)
- [ ] Value validators (`validatePositiveInteger`, `isAbsoluteURL`, etc.)
- [ ] Sleep/backoff utilities
- [ ] UUID helpers
- [ ] File conversion utilities (`to-file`)

## Client Facade
- [x] `client.completions().create`
- [ ] Other core resource registration (lazy instantiation, type-safe surfaces)
- [ ] Streaming return types (`APIPromise`, `Stream` equivalents)
- [ ] Upload helpers and binary response support

## REST Resources
- [x] Completions (`/completions`)
- [x] Chat completions (basic create)
- [ ] Chat completions advanced features (streaming, stored completions, tool runners)
- [ ] Responses (`/responses`)
- [x] Models (retrieve/list/delete; pagination helpers pending)
- [x] Embeddings (default base64 decode, float/base64 variants)
- [ ] Files (list/upload/download/delete)
- [ ] Images (generate/edit/variation + streaming events)
- [ ] Audio (speech, transcription, translation)
- [ ] Batches
- [ ] Vector stores + assistants
- [ ] Fine-tuning
- [ ] Moderations
- [ ] Conversations
- [ ] Containers
- [ ] Videos
- [ ] Webhooks
- [ ] Beta features (`beta`, `graders`, `realtime`, etc.)
- [ ] Azure OpenAI compatibility layer

## Tooling & Tests
- [ ] Unit tests with mock HTTP client
- [ ] Integration test harness (opt-in, network-gated)
- [ ] CI configuration
- [ ] Packaging/distribution strategy (CMake install, pkg-config)

## Documentation
- [x] Architecture overview
- [ ] Usage guide covering all resources
- [ ] Migration notes vs. TypeScript SDK
- [ ] Generated API reference parity

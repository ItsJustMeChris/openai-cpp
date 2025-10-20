# Responses Resource Parity Plan

The TypeScript SDK's `responses` surface is one of the richer modules and depends on multiple internal helpers. Key pieces to replicate before a faithful C++ port:

## Direct Resource Methods
- `Responses.create` – accepts large union types for multimodal input, supports background jobs, streaming, and automatic `output_text` aggregation.
- `Responses.retrieve` – toggles between standard and streaming responses depending on the `stream` query param.
- `Responses.delete`, `.cancel`, `.list` – simple REST operations with cursor pagination.
- `Responses.inputItems` – nested resource for retrieving individual input/output items.

## Supporting Utilities
- **Parser helpers (`lib/ResponsesParser.ts`)**: provides `parseResponse`, `addOutputText`, and tool-result parsing. Also reuses shared schema types from `src/resources/shared.ts`.
- **Streaming (`lib/responses/ResponseStream.ts`)**: builds on the generic SSE streaming helpers in `src/core/streaming.ts`.
- **Shared models**: many types (tools, input items, response items) are generated in `responses.ts` itself and cross-reference `shared.ts` primitives (metadata, file search filters, etc.).

## Proposed C++ Milestones
1. **Non-streaming baseline** (in progress)
   - ✅ Models for `Response` + nested types focused on `output_text` handling and minimal tool coverage.
   - ✅ `ResponsesResource::create/retrieve/list/delete/cancel` with JSON encode/decode mirroring the TS shapes.
   - ✅ Basic aggregation helper to populate `output_text` similar to `addOutputText`.
2. **Streaming support**
   - Shared SSE consumer similar to `Stream` + `ResponseStream` for incremental events.
   - Hook into `ResponsesResource::create` when `stream=true`.
3. **Advanced features**
   - InputItems sub-resource.
   - Tool runner utilities (`parseResponse`, tool call parsing) for parity-complete scenarios.

## Dependencies to Port Next
- Cursor pagination abstraction (`core/pagination.ts`) – required once list endpoints are implemented generically.
- Shared schema utilities from `resources/shared.ts` (metadata structs, tool enums).
- SSE streaming infrastructure (required for `stream=true`).

Until those helpers exist, we can implement the non-streaming subset by:
- Flattening the generated type hierarchy to essential fields (`model`, `output`, `usage`, `metadata`, footers).
- Emitting TODO comments referencing the portions of the spec that remain to be ported.

This staged approach allows shipping an initial `ResponsesResource` while deferring the more intricate streaming/tool-runner parity to dedicated follow-up tasks.

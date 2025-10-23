#!/usr/bin/env python3
import argparse
import collections
import pathlib
import re
import sys
from typing import Dict, Iterable, Set

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE_ROOT = REPO_ROOT / "include" / "openai"
NODE_RESOURCES_ROOT = (REPO_ROOT / "../openai-node/src/resources").resolve()

MANUAL_TS_MAP: Dict[str, Iterable[str]] = {
    "assistants": ["beta/assistants.ts"],
    "assistant_stream": ["beta/assistants.ts"],
    "threads": ["beta/threads.ts", "beta/threads/threads.ts"],
    "runs": ["beta/threads/runs.ts", "beta/threads/runs/runs.ts"],
    "run_steps": ["beta/threads/runs/steps.ts"],
    "messages": ["beta/threads/messages.ts"],
    "chatkit": [
        "beta/chatkit.ts",
        "beta/chatkit/chatkit.ts",
        "beta/chatkit/threads.ts",
        "beta/chatkit/sessions.ts",
    ],
}

DEFAULT_RESOURCES = [
    "audio",
    "batches",
    "chat",
    "chat_stream",
    "chatkit",
    "completions",
    "containers",
    "conversations",
    "embeddings",
    "evals",
    "files",
    "fine_tuning",
    "graders",
    "images",
    "messages",
    "models",
    "moderations",
    "responses",
    "run_steps",
    "runs",
    "threads",
    "uploads",
    "vector_stores",
    "videos",
    "webhooks",
    "assistants",
    "assistant_stream",
]

IGNORED_FIELDS: Set[str] = {"static"}

PROP_PATTERN = re.compile(r"(?<!\[)(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\??:\s")
STRUCT_PATTERN = re.compile(r"struct\s+[A-Za-z0-9_]+\s*{")
FIELD_PATTERN = re.compile(r"[A-Za-z0-9_:<> ,]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)")
COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_PATTERN = re.compile(r"//.*")


def strip_comments(text: str) -> str:
    text = re.sub(COMMENT_PATTERN, "", text)
    return re.sub(LINE_COMMENT_PATTERN, "", text)


def find_block(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def gather_ts_paths(resource: str) -> Iterable[pathlib.Path]:
    if resource in MANUAL_TS_MAP:
        for rel in MANUAL_TS_MAP[resource]:
            candidate = NODE_RESOURCES_ROOT / rel
            if candidate.exists():
                yield candidate
        return

    name = resource.replace("_", "-")
    file_candidate = NODE_RESOURCES_ROOT / f"{name}.ts"
    if file_candidate.exists():
        yield file_candidate

    dir_candidate = NODE_RESOURCES_ROOT / name
    if dir_candidate.exists():
        for ts_file in dir_candidate.rglob("*.ts"):
            yield ts_file


def extract_ts_fields(resource: str) -> Set[str]:
    fields: Set[str] = set()
    for path in gather_ts_paths(resource):
        text = strip_comments(path.read_text())
        for match in re.finditer(r"interface\s+[A-Za-z0-9_]+\s*{", text):
            brace = text.find("{", match.start())
            block = find_block(text, brace)
            fields.update(PROP_PATTERN.findall(block))
        for match in re.finditer(r"type\s+[A-Za-z0-9_]+\s*=\s*{", text):
            brace = text.find("{", match.start())
            block = find_block(text, brace)
            fields.update(PROP_PATTERN.findall(block))
    return {field for field in fields if field not in IGNORED_FIELDS}


def extract_cpp_fields(resource: str) -> Set[str]:
    header = INCLUDE_ROOT / f"{resource}.hpp"
    if not header.exists():
        return set()

    text = header.read_text()
    fields: Set[str] = set()
    search_pos = 0

    while True:
        match = STRUCT_PATTERN.search(text, search_pos)
        if not match:
            break
        brace_index = text.find("{", match.start())
        block = find_block(text, brace_index)
        statement = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            statement += " " + line
            if ";" in line:
                part = statement.strip()
                statement = ""
                if "(" in part or part.startswith("using ") or part.startswith("friend "):
                    continue
                if "enum " in part or part.endswith("}"):
                    continue
                field_match = FIELD_PATTERN.search(part)
                if field_match:
                    fields.add(field_match.group(1))
        search_pos = brace_index + len(block)

    return fields


def check_resource(resource: str) -> Dict[str, Set[str]]:
    cpp_fields = extract_cpp_fields(resource)
    ts_fields = extract_ts_fields(resource)
    missing = ts_fields - cpp_fields
    extra = cpp_fields - ts_fields
    return {"missing": missing, "extra": extra}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TS <> C++ resource field parity.")
    parser.add_argument(
        "-r",
        "--resource",
        action="append",
        help="Resource to inspect (repeatable). Defaults to the curated list.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Include resources without gaps in the output.",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Show fields present in C++ but absent in TypeScript.",
    )
    args = parser.parse_args()

    if not NODE_RESOURCES_ROOT.exists():
        print(f"error: {NODE_RESOURCES_ROOT} not found. Clone openai-node next to this repo.", file=sys.stderr)
        return 1

    resources = args.resource or DEFAULT_RESOURCES
    seen = collections.OrderedDict.fromkeys(resources)
    exit_code = 0

    for resource in seen:
        result = check_resource(resource)
        missing = sorted(result["missing"])
        extra = sorted(result["extra"])

        if not missing and not extra and not args.show_all:
            continue

        header_exists = (INCLUDE_ROOT / f"{resource}.hpp").exists()
        if not header_exists:
            print(f"{resource}: skipped (no header)")
            continue

        print(resource + ":")
        if missing:
            exit_code = 1
            print("  missing:")
            for field in missing:
                print(f"    - {field}")
        else:
            print("  missing: none")

        if args.include_extra:
            if extra:
                print("  extra:")
                for field in extra:
                    print(f"    - {field}")
            else:
                print("  extra: none")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

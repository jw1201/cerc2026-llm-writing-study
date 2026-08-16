#!/usr/bin/env python3
"""
Run branch/disagreement requests for the LLM sycophancy experiment.

This script uses the FINAL initial coding file to select only initially correct
responses for the branch/sycophancy phase.

Branch inclusion rule
---------------------
A branch is sent only if the corresponding initial response has:

    final_initial_code == I1
    final_initial_included_in_branch_analysis == yes

Conversation reconstruction rule
--------------------------------
Each weak/moderate/strong branch is reconstructed independently from the saved
initial answer:

[
  {"role": "user", "content": "[initial_prompt]"},
  {"role": "assistant", "content": "[saved_initial_response_text]"},
  {"role": "user", "content": "[weak/moderate/strong_disagreement]"}
]

No weak -> moderate -> strong chaining is used.

Expected project structure
--------------------------
experiment/
├── .env
├── config/
│   └── runtime_config.yaml
├── data/
│   ├── tasks/
│   │   └── tasks_master_v1.0_frozen.jsonl
│   ├── manifest/
│   │   └── experiment_manifest_branches_v1.2_openai_claude.csv
│   ├── raw/
│   │   ├── raw_responses_initial.jsonl
│   │   └── raw_responses_branches.jsonl
│   └── coding/
│       └── initial_coding_final.csv
└── scripts/
    └── run_branch_requests.py

Usage
-----
Dry run:

    python scripts/run_branch_requests.py --limit 3 --dry-run

Small smoke test:

    python scripts/run_branch_requests.py --limit 6

Run only one provider:

    python scripts/run_branch_requests.py --provider OpenAI --limit 6
    python scripts/run_branch_requests.py --provider Anthropic --limit 6

Full branch run:

    python scripts/run_branch_requests.py

Expected full output
--------------------
Given the current final initial coding:
336 included initial responses × 3 disagreement strengths = 1008 branch records.

Output:
    data/raw/raw_responses_branches.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


TRUTHY = {"true", "1", "yes", "y"}
FALSEY = {"false", "0", "no", "n"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file is empty or invalid YAML: {path}")
    return data


def project_root_from_config(config_path: Path) -> Path:
    # If config path is experiment/config/runtime_config.yaml, root is experiment/.
    if config_path.parent.name == "config":
        return config_path.parent.parent.resolve()
    return Path.cwd().resolve()


def resolve_project_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path, id_field: str = "branch_id") -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get(id_field) or row.get("request_id")
            if value:
                ids.add(str(value))
    return ids


def to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSEY:
        return False
    return default


def log_parameter_value(value: Any, numeric_default: Optional[float] = None) -> Any:
    """
    Preserve manifest strings such as 'not_supported' or 'not_sent' in raw logs.
    If the manifest value is numeric, return a numeric value.
    """
    if value is None or value == "":
        return numeric_default
    text_value = str(value).strip()
    try:
        return float(text_value)
    except ValueError:
        return text_value


def serialize_response(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    try:
        json.dumps(response)
        return response
    except TypeError:
        return repr(response)


def get_provider_model_returned(raw: Any, provider: str) -> Optional[str]:
    data = serialize_response(raw)
    if isinstance(data, dict):
        return data.get("model")
    return None


def get_finish_reason(raw: Any, provider: str) -> Optional[str]:
    data = serialize_response(raw)
    if not isinstance(data, dict):
        return None
    if provider == "OpenAI":
        status = data.get("status")
        incomplete = data.get("incomplete_details")
        if incomplete:
            return f"{status}; incomplete={incomplete}"
        return status
    if provider == "Anthropic":
        return data.get("stop_reason")
    return None


def get_usage(raw: Any, provider: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    data = serialize_response(raw)
    if not isinstance(data, dict):
        return None, None, None

    usage = data.get("usage") or {}
    if provider == "OpenAI":
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        return input_tokens, output_tokens, total_tokens

    if provider == "Anthropic":
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total = input_tokens + output_tokens
        return input_tokens, output_tokens, total

    return None, None, None


def extract_openai_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    data = serialize_response(response)
    if isinstance(data, dict):
        parts: List[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    parts.append(content["text"])
        return "\n".join(parts).strip()
    return ""


def extract_anthropic_text(response: Any) -> str:
    data = serialize_response(response)
    if isinstance(data, dict):
        parts: List[str] = []
        for item in data.get("content", []) or []:
            if item.get("type") == "text" and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def is_non_retryable_api_error(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    non_retryable_names = {
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    }
    if name in non_retryable_names:
        return True
    markers = [
        "Error code: 400",
        "Error code: 401",
        "Error code: 403",
        "invalid x-api-key",
        "Unsupported parameter",
        "Unsupported value",
        "not found",
    ]
    return any(marker in msg for marker in markers)


def make_task_lookup(task_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("task_id")): r for r in task_rows if r.get("task_id")}


def make_initial_lookup(raw_initial: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for r in raw_initial:
        rid = r.get("request_id")
        if rid:
            lookup[str(rid)] = r
    return lookup


def make_allowed_initial_lookup(final_coding_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    allowed: Dict[str, Dict[str, str]] = {}
    for r in final_coding_rows:
        rid = str(r.get("request_id", "")).strip()
        final_code = str(r.get("final_initial_code", "")).strip()
        included = str(r.get("final_initial_included_in_branch_analysis", "")).strip().lower()
        if rid and final_code == "I1" and included == "yes":
            allowed[rid] = r
    return allowed


def get_initial_prompt(initial_record: Dict[str, Any], task: Dict[str, Any]) -> str:
    history = initial_record.get("full_message_history")
    if isinstance(history, list):
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"])
    if task.get("initial_prompt"):
        return str(task["initial_prompt"])
    raise ValueError("Could not determine initial prompt.")


def build_messages(
    *,
    branch_row: Dict[str, str],
    initial_record: Dict[str, Any],
    task: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str, str]:
    source_field = branch_row.get("disagreement_source_field") or ""
    disagreement_prompt = task.get(source_field)
    if not disagreement_prompt:
        raise ValueError(
            f"Missing disagreement prompt field {source_field!r} for task {branch_row.get('task_id')}"
        )

    initial_prompt = get_initial_prompt(initial_record, task)
    initial_answer = initial_record.get("response_text") or ""
    if not str(initial_answer).strip():
        raise ValueError(f"Initial response_text is empty for {branch_row.get('initial_request_id')}")

    messages = [
        {"role": "user", "content": str(initial_prompt)},
        {"role": "assistant", "content": str(initial_answer)},
        {"role": "user", "content": str(disagreement_prompt)},
    ]
    return messages, str(initial_prompt), str(disagreement_prompt)


def build_request_preview(
    *,
    row: Dict[str, str],
    messages: List[Dict[str, str]],
    timeout_seconds: int,
) -> Dict[str, Any]:
    provider = row.get("provider")
    max_output_tokens = to_int(row.get("max_output_tokens"), 512)

    if provider == "OpenAI":
        # GPT-5.5 in this experiment rejects explicit temperature/top_p.
        # We omit them and log not_supported/not_sent in the manifest-level fields.
        request: Dict[str, Any] = {
            "model": row["provider_model_id_requested"],
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": row.get("reasoning_effort") or "low"},
            "tools": [],
            "tool_choice": "none",
            "store": False,
            "timeout": timeout_seconds,
        }
        return request

    if provider == "Anthropic":
        # Claude Sonnet 4.6 in this experiment rejects temperature and top_p
        # together. Manifest v1.2 sends temperature=0 and omits top_p.
        request = {
            "model": row["provider_model_id_requested"],
            "max_tokens": max_output_tokens,
            "messages": messages,
            "timeout": timeout_seconds,
        }

        temperature_value = row.get("temperature")
        if temperature_value not in {None, "", "not_supported", "not_sent", "NA"}:
            request["temperature"] = to_float(temperature_value, 0.0)

        top_p_value = row.get("top_p")
        if top_p_value not in {None, "", "not_supported", "not_sent", "NA"}:
            request["top_p"] = to_float(top_p_value, 1.0)

        return request

    return {"provider": provider, "messages": messages, "timeout": timeout_seconds}


def call_openai(
    *,
    client: Any,
    row: Dict[str, str],
    messages: List[Dict[str, str]],
    timeout_seconds: int,
) -> Tuple[Any, str, Dict[str, Any]]:
    raw_request = build_request_preview(
        row=row,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )
    response = client.responses.create(**raw_request)
    text = extract_openai_text(response)
    return response, text, raw_request


def call_anthropic(
    *,
    client: Any,
    row: Dict[str, str],
    messages: List[Dict[str, str]],
    timeout_seconds: int,
) -> Tuple[Any, str, Dict[str, Any]]:
    raw_request = build_request_preview(
        row=row,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )
    response = client.messages.create(**raw_request)
    text = extract_anthropic_text(response)
    return response, text, raw_request


def build_branch_record(
    *,
    row: Dict[str, str],
    initial_record: Dict[str, Any],
    initial_final_record: Dict[str, str],
    task: Dict[str, Any],
    messages: List[Dict[str, str]],
    initial_prompt: str,
    disagreement_prompt: str,
    raw_request: Dict[str, Any],
    raw_response: Any,
    response_text: Optional[str],
    timestamp_start: str,
    timestamp_end: str,
    latency_ms: int,
    error_status: Optional[str],
    error_traceback: Optional[str],
    retry_count: int,
) -> Dict[str, Any]:
    provider = row.get("provider")
    input_tokens, output_tokens, total_tokens = get_usage(raw_response, provider or "")

    return {
        "request_id": row.get("branch_id"),
        "trial_type": "branch",
        "run_id": to_int(row.get("run_id"), 0),
        "task_id": row.get("task_id"),
        "branch_id": row.get("branch_id"),
        "initial_request_id": row.get("initial_request_id"),

        # Model/API metadata
        "model_label": row.get("model_label"),
        "model_code": row.get("model_code"),
        "provider": provider,
        "api_interface": row.get("api_interface"),
        "provider_model_id_requested": row.get("provider_model_id_requested"),
        "provider_model_id_returned": get_provider_model_returned(raw_response, provider or ""),
        "deployment_name": row.get("deployment_name"),
        "api_version": row.get("api_version"),
        "region": row.get("region"),

        # Experimental condition
        "system_prompt_condition": row.get("system_prompt_condition"),
        "system_prompt_id": row.get("system_prompt_id"),
        "disagreement_strength": row.get("disagreement_strength"),
        "strength_code": row.get("strength_code"),
        "strength_order_index": to_int(row.get("strength_order_index"), 0),
        "disagreement_source_field": row.get("disagreement_source_field"),
        "disagreement_template_id": row.get("disagreement_template_id"),
        "disagreement_prompt_hash": row.get("disagreement_prompt_hash"),
        "computed_disagreement_prompt_hash": sha16(disagreement_prompt),
        "branch_context_rule": row.get("branch_context_rule"),

        # Runtime configuration
        "temperature": log_parameter_value(row.get("temperature"), numeric_default=0.0),
        "top_p": log_parameter_value(row.get("top_p"), numeric_default=1.0),
        "max_output_tokens": to_int(row.get("max_output_tokens"), 512),
        "seed": row.get("seed"),
        "reasoning_effort": row.get("reasoning_effort"),
        "tools_enabled": parse_bool(row.get("tools_enabled")),
        "web_enabled": parse_bool(row.get("web_enabled")),
        "retrieval_enabled": parse_bool(row.get("retrieval_enabled")),
        "memory_enabled": parse_bool(row.get("memory_enabled")),
        "streaming_enabled": parse_bool(row.get("streaming_enabled")),

        # Task metadata
        "task_dataset_version": row.get("task_dataset_version") or task.get("dataset_version"),
        "category": row.get("category") or task.get("category"),
        "subcategory": row.get("subcategory") or task.get("subcategory"),
        "gold_polarity": row.get("gold_polarity") or task.get("gold_polarity"),
        "ground_truth_type": row.get("ground_truth_type") or task.get("ground_truth_type"),
        "task_order_index": to_int(row.get("task_order_index"), 0),
        "model_order_index": to_int(row.get("model_order_index"), 0),

        # Initial coding dependency
        "final_initial_code": initial_final_record.get("final_initial_code"),
        "final_initial_included_in_branch_analysis": initial_final_record.get("final_initial_included_in_branch_analysis"),
        "initial_response_text": initial_record.get("response_text"),
        "initial_finish_reason": initial_record.get("finish_reason"),
        "initial_provider_model_id_returned": initial_record.get("provider_model_id_returned"),

        # Task/gold fields useful for branch coding
        "correct_judgment": task.get("correct_judgment"),
        "gold_rationale": task.get("gold_rationale"),
        "incorrect_user_position": task.get("incorrect_user_position"),
        "expected_sycophantic_error": task.get("expected_sycophantic_error"),

        # Prompts and message history
        "initial_prompt_source": "initial_raw_full_message_history_or_task_initial_prompt",
        "initial_prompt_hash": sha16(initial_prompt),
        "disagreement_prompt": disagreement_prompt,
        "full_message_history": messages,

        # API trace
        "raw_request": raw_request,
        "raw_response": serialize_response(raw_response),
        "response_text": response_text,
        "finish_reason": get_finish_reason(raw_response, provider or ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "latency_ms": latency_ms,
        "error_status": error_status,
        "error_traceback": error_traceback,
        "retry_count": retry_count,
    }


def select_rows(
    *,
    branch_manifest_rows: List[Dict[str, str]],
    allowed_initials: Dict[str, Dict[str, str]],
    initial_lookup: Dict[str, Dict[str, Any]],
    task_lookup: Dict[str, Dict[str, Any]],
    existing_branch_ids: set[str],
    skip_existing: bool,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []

    for row in branch_manifest_rows:
        branch_id = row.get("branch_id")
        initial_request_id = row.get("initial_request_id")
        task_id = row.get("task_id")

        # Main inclusion filter.
        if initial_request_id not in allowed_initials:
            continue

        # Safety checks: raw initial and task must exist.
        if initial_request_id not in initial_lookup:
            continue
        if task_id not in task_lookup:
            continue

        if skip_existing and branch_id in existing_branch_ids:
            continue

        if args.provider and row.get("provider") != args.provider:
            continue
        if args.model_label and row.get("model_label") != args.model_label:
            continue
        if args.strength and row.get("disagreement_strength") != args.strength:
            continue
        if args.run_id is not None and str(row.get("run_id")) != str(args.run_id):
            continue
        if args.branch_id and branch_id != args.branch_id:
            continue
        if args.initial_request_id and initial_request_id != args.initial_request_id:
            continue
        if args.task_id and task_id != args.task_id:
            continue

        selected.append(row)

    if args.limit is not None:
        selected = selected[: args.limit]

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--initial-coding", default="data/coding/initial_coding_final.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--provider", choices=["OpenAI", "Anthropic"], default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--strength", choices=["weak", "moderate", "strong"], default=None)
    parser.add_argument("--branch-id", default=None)
    parser.add_argument("--initial-request-id", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    root = project_root_from_config(config_path)

    task_file = resolve_project_path(root, config["experiment"]["task_file"])
    branch_manifest_file = resolve_project_path(root, config["experiment"]["branch_manifest"])
    raw_initial_file = resolve_project_path(root, config["output"]["raw_initial"])
    raw_branches_file = resolve_project_path(root, config["output"]["raw_branches"])
    initial_coding_file = resolve_project_path(root, args.initial_coding)

    runtime = config.get("runtime", {})
    timeout_seconds = to_int(runtime.get("timeout_seconds"), 120)
    max_retries = to_int(runtime.get("max_retries"), 3)
    retry_backoff_seconds = to_int(runtime.get("retry_backoff_seconds"), 2)
    skip_existing = bool(runtime.get("skip_existing_request_ids", True)) and not args.no_skip_existing

    load_dotenv(root / ".env")

    task_rows = read_jsonl(task_file)
    raw_initial_rows = read_jsonl(raw_initial_file)
    branch_manifest_rows = read_csv_dicts(branch_manifest_file)
    initial_coding_rows = read_csv_dicts(initial_coding_file)

    task_lookup = make_task_lookup(task_rows)
    initial_lookup = make_initial_lookup(raw_initial_rows)
    allowed_initials = make_allowed_initial_lookup(initial_coding_rows)
    existing_branch_ids = load_existing_ids(raw_branches_file, id_field="branch_id")

    selected_rows = select_rows(
        branch_manifest_rows=branch_manifest_rows,
        allowed_initials=allowed_initials,
        initial_lookup=initial_lookup,
        task_lookup=task_lookup,
        existing_branch_ids=existing_branch_ids,
        skip_existing=skip_existing,
        args=args,
    )

    print("Run branch requests")
    print(f"Project root: {root}")
    print(f"Task file: {task_file}")
    print(f"Branch manifest: {branch_manifest_file}")
    print(f"Raw initial file: {raw_initial_file}")
    print(f"Initial coding file: {initial_coding_file}")
    print(f"Output file: {raw_branches_file}")
    print(f"Final included initial responses: {len(allowed_initials)}")
    print(f"Existing branch records: {len(existing_branch_ids)}")
    print(f"Selected branch rows: {len(selected_rows)}")
    print(f"Skip existing: {skip_existing}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        for row in selected_rows[:10]:
            initial_record = initial_lookup[row["initial_request_id"]]
            task = task_lookup[row["task_id"]]
            messages, initial_prompt, disagreement_prompt = build_messages(
                branch_row=row,
                initial_record=initial_record,
                task=task,
            )
            raw_request = build_request_preview(
                row=row,
                messages=messages,
                timeout_seconds=timeout_seconds,
            )
            print("")
            print("--- DRY RUN ---")
            print(f"branch_id: {row.get('branch_id')}")
            print(f"initial_request_id: {row.get('initial_request_id')}")
            print(f"provider: {row.get('provider')}")
            print(f"model: {row.get('provider_model_id_requested')}")
            print(f"strength: {row.get('disagreement_strength')}")
            print(f"message roles: {[m['role'] for m in messages]}")
            print(f"disagreement preview: {disagreement_prompt[:300]}")
            print(f"raw_request keys: {list(raw_request.keys())}")
            if row.get("provider") == "OpenAI":
                print(f"OpenAI sends temperature? {'temperature' in raw_request}")
                print(f"OpenAI sends top_p? {'top_p' in raw_request}")
                print(f"OpenAI store: {raw_request.get('store')}")
            if row.get("provider") == "Anthropic":
                print(f"Anthropic sends temperature? {'temperature' in raw_request}")
                print(f"Anthropic sends top_p? {'top_p' in raw_request}")
        return

    if not selected_rows:
        print("Nothing to run.")
        return

    providers_needed = {row.get("provider") for row in selected_rows}
    openai_client = None
    anthropic_client = None

    if "OpenAI" in providers_needed:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing in .env or environment.")
        openai_client = OpenAI(api_key=api_key)

    if "Anthropic" in providers_needed:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is missing in .env or environment.")
        anthropic_client = anthropic.Anthropic(api_key=api_key)

    for row in tqdm(selected_rows, desc="Branch requests"):
        branch_id = row.get("branch_id")
        initial_request_id = row.get("initial_request_id")
        task_id = row.get("task_id")
        provider = row.get("provider")

        initial_record = initial_lookup[initial_request_id]
        initial_final_record = allowed_initials[initial_request_id]
        task = task_lookup[task_id]

        raw_response = None
        response_text = None
        error_status = None
        error_traceback = None
        retry_count = 0

        timestamp_start = utc_now()
        start_time = time.time()

        try:
            messages, initial_prompt, disagreement_prompt = build_messages(
                branch_row=row,
                initial_record=initial_record,
                task=task,
            )
            raw_request = build_request_preview(
                row=row,
                messages=messages,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            timestamp_end = utc_now()
            latency_ms = int((time.time() - start_time) * 1000)
            error_status = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            messages = []
            initial_prompt = ""
            disagreement_prompt = ""
            raw_request = {}

            record = build_branch_record(
                row=row,
                initial_record=initial_record,
                initial_final_record=initial_final_record,
                task=task,
                messages=messages,
                initial_prompt=initial_prompt,
                disagreement_prompt=disagreement_prompt,
                raw_request=raw_request,
                raw_response=raw_response,
                response_text=response_text,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                latency_ms=latency_ms,
                error_status=error_status,
                error_traceback=error_traceback,
                retry_count=retry_count,
            )
            append_jsonl(raw_branches_file, record)
            continue

        for attempt in range(max_retries + 1):
            retry_count = attempt
            try:
                if provider == "OpenAI":
                    raw_response, response_text, raw_request = call_openai(
                        client=openai_client,
                        row=row,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                    )
                elif provider == "Anthropic":
                    raw_response, response_text, raw_request = call_anthropic(
                        client=anthropic_client,
                        row=row,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                error_status = None
                error_traceback = None
                break

            except Exception as exc:
                error_status = f"{type(exc).__name__}: {exc}"
                error_traceback = traceback.format_exc()

                if is_non_retryable_api_error(exc) or attempt >= max_retries:
                    raw_response = None
                    response_text = None
                    break

                sleep_seconds = retry_backoff_seconds * (2 ** attempt)
                time.sleep(sleep_seconds)

        timestamp_end = utc_now()
        latency_ms = int((time.time() - start_time) * 1000)

        record = build_branch_record(
            row=row,
            initial_record=initial_record,
            initial_final_record=initial_final_record,
            task=task,
            messages=messages,
            initial_prompt=initial_prompt,
            disagreement_prompt=disagreement_prompt,
            raw_request=raw_request,
            raw_response=raw_response,
            response_text=response_text,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            latency_ms=latency_ms,
            error_status=error_status,
            error_traceback=error_traceback,
            retry_count=retry_count,
        )
        append_jsonl(raw_branches_file, record)

    print("Done.")
    print(f"Output file: {raw_branches_file}")


if __name__ == "__main__":
    main()

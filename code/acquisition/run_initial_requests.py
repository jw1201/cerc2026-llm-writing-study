#!/usr/bin/env python3
"""
run_initial_requests.py

Runs the initial model requests for the sycophancy experiment.

Version note: v4 omits unsupported OpenAI GPT-5.5 sampling parameters
(temperature/top_p) from the actual OpenAI request, sends store=False for
OpenAI where supported, and also omits Anthropic top_p when the manifest
marks it as not_sent. Manifest strings such as not_supported/not_sent are
preserved in the raw log for methodological documentation.

Input:
- runtime_config.yaml
- tasks_master_v1.0_frozen.jsonl
- experiment_manifest_initial_v1.0_openai_claude.csv
- .env with OPENAI_API_KEY and ANTHROPIC_API_KEY

Output:
- raw_responses_initial.jsonl

Recommended first test:
    python scripts/run_initial_requests.py --limit 2

Dry run without API calls:
    python scripts/run_initial_requests.py --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root_from_config(config_path: Path) -> Path:
    """
    If config is experiment/config/runtime_config.yaml, root is experiment/.
    Otherwise use the current working directory.
    """
    config_path = config_path.resolve()
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return Path.cwd().resolve()


def resolve_project_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return root / path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
    return rows


def load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    tasks = load_jsonl(path)
    task_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError(f"Task without task_id in {path}")
        if task_id in task_map:
            raise ValueError(f"Duplicate task_id in task file: {task_id}")
        task_map[task_id] = task
    return task_map


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def existing_request_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                request_id = row.get("request_id")
                if request_id:
                    ids.add(request_id)
            except json.JSONDecodeError:
                # Do not fail here; validation script can inspect this later.
                continue
    return ids


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
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


def serialize_sdk_object(obj: Any) -> Any:
    """Convert OpenAI/Anthropic SDK objects into JSON-serializable structures."""
    if obj is None:
        return None

    # Pydantic v2 style
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()

    # Anthropic/OpenAI legacy helpers
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def extract_openai_text(response: Any) -> str:
    """Extract assistant text from an OpenAI Responses API response."""
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)

    data = serialize_sdk_object(response)
    chunks: List[str] = []
    for item in data.get("output", []) if isinstance(data, dict) else []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_anthropic_text(response: Any) -> str:
    """Extract assistant text from an Anthropic Messages API response."""
    data = serialize_sdk_object(response)
    chunks: List[str] = []
    if isinstance(data, dict):
        for block in data.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                chunks.append(block["text"])
    return "\n".join(chunks).strip()


def extract_usage(data: Any, provider: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return input_tokens, output_tokens, total_tokens if available.
    """
    if not isinstance(data, dict):
        return None, None, None

    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None, None

    # OpenAI Responses API
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    # Anthropic Messages API
    if input_tokens is None:
        input_tokens = usage.get("input_tokens")
    if output_tokens is None:
        output_tokens = usage.get("output_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens


def get_returned_model(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        return data.get("model")
    return None


def get_finish_reason(data: Any, provider: str) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    if provider == "OpenAI":
        # Responses API uses status/incomplete_details rather than a classic finish_reason.
        status = data.get("status")
        incomplete = data.get("incomplete_details")
        if incomplete:
            return f"{status}; incomplete={incomplete}"
        return status
    if provider == "Anthropic":
        return data.get("stop_reason")
    return None


def is_non_retryable_api_error(exc: Exception) -> bool:
    """
    Return True for errors that should not be retried because retrying will not
    change the outcome without changing the request or credentials.
    """
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
    non_retryable_markers = [
        "Error code: 400",
        "Error code: 401",
        "Error code: 403",
        "invalid x-api-key",
        "Unsupported parameter",
        "Unsupported value",
    ]
    return any(marker in msg for marker in non_retryable_markers)


def build_request_preview(
    *,
    row: Dict[str, str],
    messages: List[Dict[str, str]],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Build the intended request payload for logging even if the API call fails.
    This intentionally excludes secrets.
    """
    provider = row.get("provider")
    max_output_tokens = to_int(row.get("max_output_tokens"), 512)

    if provider == "OpenAI":
        # GPT-5.5 rejects explicit temperature/top_p settings in this setup.
        # We therefore do not send unsupported sampling parameters in the
        # OpenAI request payload. The manifest records them as not_supported /
        # not_sent for transparency. store=False avoids provider-side response
        # storage where supported.
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
        # Claude Sonnet 4.6 in this setup rejects requests that specify both
        # temperature and top_p. We use temperature=0 for the Claude condition
        # and omit top_p when the manifest marks it as not_sent.
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


def build_initial_raw_record(
    *,
    row: Dict[str, str],
    task: Dict[str, Any],
    messages: List[Dict[str, str]],
    raw_request: Dict[str, Any],
    raw_response: Any,
    response_text: Optional[str],
    timestamp_start: str,
    timestamp_end: str,
    latency_ms: int,
    retry_count: int,
    error_status: Optional[str] = None,
    error_traceback: Optional[str] = None,
) -> Dict[str, Any]:
    provider = row.get("provider")
    raw_response_serialized = serialize_sdk_object(raw_response)
    input_tokens, output_tokens, total_tokens = extract_usage(raw_response_serialized, provider or "")
    return {
        "request_id": row["initial_request_id"],
        "trial_type": "initial",
        "run_id": to_int(row.get("run_id"), default=-1),
        "task_id": row["task_id"],
        "branch_id": None,
        "initial_request_id": None,
        "model_label": row.get("model_label"),
        "model_code": row.get("model_code"),
        "provider": provider,
        "api_interface": row.get("api_interface"),
        "provider_model_id_requested": row.get("provider_model_id_requested"),
        "provider_model_id_returned": get_returned_model(raw_response_serialized),
        "deployment_name": row.get("deployment_name"),
        "api_version": row.get("api_version"),
        "region": row.get("region"),
        "system_prompt_condition": row.get("system_prompt_condition"),
        "system_prompt_id": row.get("system_prompt_id"),
        "temperature": log_parameter_value(row.get("temperature"), numeric_default=0.0),
        "top_p": log_parameter_value(row.get("top_p"), numeric_default=1.0),
        "max_output_tokens": to_int(row.get("max_output_tokens"), default=512),
        "seed": row.get("seed"),
        "reasoning_effort": row.get("reasoning_effort"),
        "tools_enabled": to_bool(row.get("tools_enabled")),
        "web_enabled": to_bool(row.get("web_enabled")),
        "retrieval_enabled": to_bool(row.get("retrieval_enabled")),
        "memory_enabled": to_bool(row.get("memory_enabled")),
        "streaming_enabled": to_bool(row.get("streaming_enabled")),
        "task_dataset_version": row.get("task_dataset_version"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "gold_polarity": row.get("gold_polarity"),
        "ground_truth_type": row.get("ground_truth_type"),
        "task_order_index": to_int(row.get("task_order_index"), default=-1),
        "model_order_index": to_int(row.get("model_order_index"), default=-1),
        "initial_prompt_source_field": row.get("initial_prompt_source_field"),
        "initial_prompt_hash": row.get("initial_prompt_hash"),
        "full_message_history": messages,
        "raw_request": raw_request,
        "raw_response": raw_response_serialized,
        "response_text": response_text,
        "finish_reason": get_finish_reason(raw_response_serialized, provider or ""),
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


def filter_manifest_rows(
    rows: List[Dict[str, str]],
    *,
    run_id: Optional[int],
    provider: Optional[str],
    model_label: Optional[str],
    request_id: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, str]]:
    filtered = rows

    if run_id is not None:
        filtered = [r for r in filtered if to_int(r.get("run_id"), -1) == run_id]

    if provider:
        filtered = [r for r in filtered if r.get("provider") == provider]

    if model_label:
        filtered = [r for r in filtered if r.get("model_label") == model_label]

    if request_id:
        filtered = [r for r in filtered if r.get("initial_request_id") == request_id]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def run() -> int:
    parser = argparse.ArgumentParser(description="Run initial API requests for the sycophancy experiment.")
    parser.add_argument("--config", default="config/runtime_config.yaml", help="Path to runtime_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N manifest rows after filtering.")
    parser.add_argument("--run-id", type=int, default=None, help="Run only one run_id, e.g. 1.")
    parser.add_argument("--provider", default=None, choices=["OpenAI", "Anthropic"], help="Run only one provider.")
    parser.add_argument("--model-label", default=None, help="Run only one model_label, e.g. 'GPT-5.5'.")
    parser.add_argument("--request-id", default=None, help="Run only one specific initial_request_id.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print requests without API calls.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip request IDs already present in output.")
    args = parser.parse_args()

    config_path = Path(args.config)
    root = project_root_from_config(config_path)
    config = load_yaml(config_path)

    load_dotenv(root / ".env")

    task_file = resolve_project_path(root, config["experiment"]["task_file"])
    manifest_file = resolve_project_path(root, config["experiment"]["initial_manifest"])
    output_file = resolve_project_path(root, config["output"]["raw_initial"])

    timeout_seconds = int(config.get("runtime", {}).get("timeout_seconds", 120))
    max_retries = int(config.get("runtime", {}).get("max_retries", 5))
    retry_backoff_seconds = float(config.get("runtime", {}).get("retry_backoff_seconds", 2))
    skip_existing = bool(config.get("runtime", {}).get("skip_existing_request_ids", True))
    if args.no_skip_existing:
        skip_existing = False

    tasks = load_tasks(task_file)
    manifest_rows = load_manifest(manifest_file)
    rows = filter_manifest_rows(
        manifest_rows,
        run_id=args.run_id,
        provider=args.provider,
        model_label=args.model_label,
        request_id=args.request_id,
        limit=args.limit,
    )

    if not rows:
        print("No manifest rows selected. Check filters.", file=sys.stderr)
        return 1

    already_done = existing_request_ids(output_file) if skip_existing else set()

    openai_client = None
    anthropic_client = None

    if not args.dry_run:
        if any(r.get("provider") == "OpenAI" for r in rows):
            if OpenAI is None:
                raise RuntimeError("Package 'openai' is not installed. Run: pip install -r requirements.txt")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env")
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if any(r.get("provider") == "Anthropic" for r in rows):
            if Anthropic is None:
                raise RuntimeError("Package 'anthropic' is not installed. Run: pip install -r requirements.txt")
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to .env")
            anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"Project root: {root}")
    print(f"Task file: {task_file}")
    print(f"Manifest file: {manifest_file}")
    print(f"Output file: {output_file}")
    print(f"Selected rows: {len(rows)}")
    print(f"Skip existing: {skip_existing}")
    print(f"Dry run: {args.dry_run}")

    for row in tqdm(rows, desc="Initial requests"):
        request_id = row["initial_request_id"]
        if request_id in already_done:
            continue

        task_id = row["task_id"]
        if task_id not in tasks:
            raise KeyError(f"Task {task_id} from manifest not found in task file.")

        task = tasks[task_id]
        initial_prompt = task["initial_prompt"]
        messages = [{"role": "user", "content": initial_prompt}]

        if args.dry_run:
            print("\n--- DRY RUN ---")
            print(f"request_id: {request_id}")
            print(f"provider: {row.get('provider')}")
            print(f"model: {row.get('provider_model_id_requested')}")
            print(f"prompt preview: {initial_prompt[:300].replace(chr(10), ' ')}...")
            continue

        timestamp_start = utc_now_iso()
        start = time.perf_counter()
        raw_response = None
        response_text = None
        raw_request: Dict[str, Any] = build_request_preview(
            row=row,
            messages=messages,
            timeout_seconds=timeout_seconds,
        )
        error_status = None
        error_traceback = None
        retry_count = 0

        for attempt in range(max_retries + 1):
            retry_count = attempt
            try:
                provider = row.get("provider")
                if provider == "OpenAI":
                    assert openai_client is not None
                    raw_response, response_text, raw_request = call_openai(
                        client=openai_client,
                        row=row,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                    )
                elif provider == "Anthropic":
                    assert anthropic_client is not None
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

            except Exception as exc:  # deliberately broad for robust experiment logging
                error_status = f"{type(exc).__name__}: {exc}"
                error_traceback = traceback.format_exc()

                if is_non_retryable_api_error(exc) or attempt >= max_retries:
                    raw_response = None
                    response_text = None
                    break

                sleep_seconds = retry_backoff_seconds * (2 ** attempt)
                time.sleep(sleep_seconds)

        end = time.perf_counter()
        timestamp_end = utc_now_iso()
        latency_ms = int((end - start) * 1000)

        record = build_initial_raw_record(
            row=row,
            task=task,
            messages=messages,
            raw_request=raw_request,
            raw_response=raw_response,
            response_text=response_text,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            latency_ms=latency_ms,
            retry_count=retry_count,
            error_status=error_status,
            error_traceback=error_traceback,
        )
        append_jsonl(output_file, record)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())

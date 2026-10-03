from __future__ import annotations

import json
import math
import os
import random
import re
import secrets
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from fingerprint import analyze_global_outputs, generate_challenges, parse_numbers
from bank_builder import build_bank, read_rows
from challenge_suite import fingerprint_suite


PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "gpt_reference.jsonl"
BANK_FILE = PROJECT / "data" / "gpt_bank.json"

# urllib 默认的 Python-urllib User-Agent 会被 Cloudflare/WAF 网关直接拦成 403，
# 因此伪装成真实客户端（与 gpt56 检测器使用的 UA 一致）。
DEFAULT_UPSTREAM_USER_AGENT = (
    "Codex Desktop/0.147.0-alpha.1.2 (Windows 10.0.26200; x86_64) unknown "
    "(codex_exec; 0.147.0-alpha.1.2)"
)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0


def upstream_user_agent() -> str:
    override = (
        os.environ.get("MODELTRACE_USER_AGENT")
        or os.environ.get("GPT56_USER_AGENT")
        or ""
    ).strip()
    return override or DEFAULT_UPSTREAM_USER_AGENT


def bank_summary(bank: dict) -> dict:
    return {
        "model_count": len(bank["models"]),
        "response_count": sum(model["response_count"] for model in bank["models"]),
        "number_count": sum(model["valid_number_count"] for model in bank["models"]),
        "models": [
            {
                "id": model["id"],
                "display_name": model["display_name"],
                "responses": model["response_count"],
                "valid_numbers": model["valid_number_count"],
            }
            for model in bank["models"]
        ],
    }


def make_row(
    model_label: str,
    text: str,
    condition: str,
    challenge_id: str,
    expected_count: int = 0,
    temperature: str | float = "unknown",
    bank_id: str = "reference-bank",
    wrapper_transport: str | None = None,
    provider: str = "api",
    prompt: str | None = None,
    base_prompt: str | None = None,
    system_prompt: str | None = None,
    user_prefix: str | None = None,
) -> dict:
    numbers = parse_numbers(text)
    threshold = max(80, math.ceil(expected_count * 0.55)) if expected_count else 80
    row_id = f"{condition}-{secrets.token_hex(8)}"
    return {
        "row_id": row_id,
        "parent_row_id": row_id,
        "bank_id": bank_id,
        "source": model_label,
        "model_id": model_label,
        "exact_version": model_label,
        "condition_id": condition,
        "nuisance_condition_id": condition,
        "wrapper_id": condition,
        "wrapper_transport": wrapper_transport or ("manual_import" if condition == "manual" else "clean"),
        "provider": "manual" if condition == "manual" else provider,
        "challenge_id": challenge_id,
        "task_index": 0,
        "requested_count": expected_count,
        "parsed_count": len(numbers),
        "strict_threshold": threshold,
        "strict_valid": len(numbers) >= threshold,
        "temperature": temperature,
        "config_id": "enrollment",
        "prompt": prompt,
        "base_prompt": base_prompt,
        "system_prompt": system_prompt,
        "user_prefix": user_prefix,
        "text": text,
        "error": None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def append_rows(rows: list[dict], data_file: Path = DATA_FILE) -> None:
    with data_file.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_bank(data_file: Path = DATA_FILE, bank_file: Path = BANK_FILE) -> dict:
    previous = json.loads(bank_file.read_text(encoding="utf-8")) if bank_file.exists() else None
    bank = build_bank(read_rows(data_file), previous["calibration"] if previous else None)
    bank_file.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bank


def enroll_manual(
    model_label: str,
    pasted: str,
    data_file: Path = DATA_FILE,
    bank_file: Path = BANK_FILE,
    bank_id: str = "reference-bank",
) -> dict:
    blocks = [
        block.strip()
        for block in re.split(r"(?m)^\s*===OUTPUT===\s*$", pasted)
        if block.strip()
    ]
    rows = [
        make_row(
            model_label=model_label,
            text=block,
            condition="manual",
            challenge_id=f"manual-{secrets.token_hex(6)}",
            bank_id=bank_id,
        )
        for block in blocks
    ]
    accepted = [row for row in rows if row["strict_valid"]]
    append_rows(accepted, data_file)
    bank = rebuild_bank(data_file, bank_file)
    return {
        "submitted": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "parsed_numbers": [row["parsed_count"] for row in rows],
        "bank": bank_summary(bank),
    }


# 自动探测时依次尝试的接口格式
AUTO_FORMATS = ("responses", "openai", "anthropic")


def completion_url(base_url: str, api_format: str = "openai") -> str:
    normalized = base_url.rstrip("/")
    if api_format == "anthropic":
        if normalized.endswith("/messages"):
            return normalized
        if normalized.endswith("/v1"):
            return normalized + "/messages"
        return normalized + "/v1/messages"
    if api_format == "responses":
        if normalized.endswith("/responses"):
            return normalized
        if normalized.endswith("/v1"):
            return normalized + "/responses"
        return normalized + "/v1/responses"
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _looks_like_waf_block(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("cloudflare", "just a moment", "cf-ray", "access denied", "attention required")
    )


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _compact_upstream_error(details: str, fallback: str) -> str:
    text = (details or fallback or "").strip()
    if not text:
        return "上游接口返回错误"
    if "a timeout occurred" in text.lower() or "error code 524" in text.lower():
        return "上游网关等待模型响应超时（Cloudflare 524）"
    if _looks_like_waf_block(text):
        return (
            "请求被上游网关拦截（Cloudflare/WAF 拦截页）。"
            "请确认 base_url 指向 API 端点而非网页地址、API Key 有效，"
            "或该服务是否限制当前网络/IP"
        )
    if text.startswith("<") or "{" not in text and "html" in text.lower():
        return text[:200]
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            return json.dumps(payload, ensure_ascii=False)[:500]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return text[:500]


def minimum_numbers(expected_count: int) -> int:
    return max(80, math.ceil(expected_count * 0.55))


def _build_request(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str,
    system_prompt: str = "",
    stream: bool = False,
) -> tuple[str, dict, bytes]:
    if api_format == "anthropic":
        body_data = {
            "model": api_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body_data["system"] = system_prompt
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    elif api_format == "responses":
        body_data = {"model": api_model, "input": prompt, "store": False}
        if system_prompt:
            body_data["instructions"] = system_prompt
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    else:
        body_data = {
            "model": api_model,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    if temperature is not None:
        body_data["temperature"] = temperature
    if stream:
        body_data["stream"] = True
        headers["Accept"] = "text/event-stream"
    body = json.dumps(body_data).encode("utf-8")
    return completion_url(base_url, api_format), headers, body


def _check_stop_reason(reason: str | None, api_format: str) -> None:
    if api_format == "anthropic":
        if reason == "refusal":
            raise RuntimeError("模型拒绝生成，本次回答不计入")
        if reason == "max_tokens":
            raise RuntimeError("回答因 max_tokens 截断，本次回答不计入")
    elif api_format == "responses":
        if reason == "refusal":
            raise RuntimeError("模型拒绝生成，本次回答不计入")
        if reason:
            raise RuntimeError(f"回答未正常完成（{reason}），本次回答不计入")
    elif reason in {"length", "content_filter"}:
        raise RuntimeError(f"回答未正常完成（{reason}），本次回答不计入")


def _extract_content(payload: dict, api_format: str) -> str:
    if api_format == "anthropic":
        content = "".join(
            block.get("text", "")
            for block in payload["content"]
            if block.get("type") == "text"
        )
        _check_stop_reason(payload.get("stop_reason"), api_format)
    elif api_format == "responses":
        content, reason = _responses_output(payload)
        _check_stop_reason(reason, api_format)
    else:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        _check_stop_reason(choice.get("finish_reason"), api_format)
    return str(content)


def _responses_output(payload: dict) -> tuple[str, str | None]:
    """从 Responses API 的 response 对象取出文本与未正常完成的原因。"""
    status = payload.get("status")
    if status == "failed":
        error = payload.get("error") or {}
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(f"上游生成失败：{message or status}")
    texts = []
    refused = False
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "refusal":
                refused = True
    content = "".join(texts) or payload.get("output_text") or ""
    reason = None
    if status == "incomplete":
        reason = (payload.get("incomplete_details") or {}).get("reason") or "incomplete"
    elif refused and not content:
        reason = "refusal"
    return str(content), reason


def _request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str,
    system_prompt: str = "",
) -> str:
    # Responses 也以流式请求：长回答不会因网关空闲超时（如 Cloudflare 524）被切断
    url, headers, body = _build_request(
        base_url, api_key, api_model, prompt, temperature, api_format, system_prompt,
        stream=api_format == "responses",
    )
    payload = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw = response.read().decode("utf-8")
            if _looks_like_sse(raw):
                # 部分网关（如 Codex 中转）无论 stream 取值都返回 SSE，且未必标注 Content-Type
                return "".join(_parse_sse(raw.splitlines(), api_format))
            payload = json.loads(raw)
            break
        except urllib.error.HTTPError as error:
            details = _read_error_body(error)
            message = _compact_upstream_error(details, error.reason)
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            if attempt < MAX_ATTEMPTS and error.code in RETRYABLE_STATUS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            raise RuntimeError(f"HTTP {error.code}: {message}{retried}") from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", str(error))
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            raise RuntimeError(f"无法连接接口：{reason}{retried}") from error
    return _extract_content(payload, api_format)


class _StreamUnsupported(Exception):
    """流式请求在收到任何内容前失败，可改用普通请求重试。"""


def _looks_like_sse(text: str) -> bool:
    return text.lstrip().startswith(("event:", "data:"))


def _stream_events(response, api_format: str):
    """读取上游响应，逐个 yield 文本增量。"""
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        yield from _parse_sse(response, api_format)
        return
    # 上游忽略了 stream 参数直接返回完整 JSON，或返回了未标注类型的 SSE
    raw = response.read().decode("utf-8")
    if _looks_like_sse(raw):
        yield from _parse_sse(raw.splitlines(), api_format)
        return
    yield _extract_content(json.loads(raw), api_format)


def _parse_sse(lines, api_format: str):
    """解析 SSE 行（bytes 或 str），逐个 yield 文本增量；结束时检查停止原因。"""
    stop_reason = None
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "error" or (api_format == "openai" and event.get("error")):
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"上游流式输出错误：{message or data[:200]}")
        if api_format == "responses":
            kind = event.get("type", "")
            if kind == "response.output_text.delta" and event.get("delta"):
                yield event["delta"]
            elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                _, stop_reason = _responses_output(event.get("response") or {})
            elif kind == "response.refusal.done":
                stop_reason = "refusal"
        elif api_format == "anthropic":
            if event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]
            elif event.get("type") == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
        else:
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            text = (choice.get("delta") or {}).get("content")
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            if text:
                yield text
            stop_reason = choice.get("finish_reason") or stop_reason
    _check_stop_reason(stop_reason, api_format)


def _stream_format(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str,
    system_prompt: str = "",
):
    """以单一协议流式请求，yield 事件 dict，返回完整文本。"""
    url, headers, body = _build_request(
        base_url, api_key, api_model, prompt, temperature, api_format, system_prompt, stream=True
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        yield {"type": "status", "stage": "connecting", "format": api_format, "attempt": attempt}
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        received = False
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                yield {"type": "status", "stage": "waiting", "format": api_format, "attempt": attempt}
                text = ""
                for chunk in _stream_events(response, api_format):
                    if not received:
                        received = True
                        yield {"type": "status", "stage": "streaming", "format": api_format, "attempt": attempt}
                    text += chunk
                    yield {
                        "type": "delta",
                        "text": chunk,
                        "chars": len(text),
                        "parsed": len(parse_numbers(text)),
                    }
                return text
        except urllib.error.HTTPError as error:
            details = _read_error_body(error)
            message = _compact_upstream_error(details, error.reason)
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            if attempt < MAX_ATTEMPTS and error.code in RETRYABLE_STATUS:
                yield {"type": "status", "stage": "retry", "format": api_format, "attempt": attempt, "message": f"HTTP {error.code}"}
                yield {"type": "reset"}
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            failure = RuntimeError(f"HTTP {error.code}: {message}{retried}")
            if error.code in {401, 403, 404, 405}:
                raise failure from error
            raise _StreamUnsupported(str(failure)) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            reason = getattr(error, "reason", str(error))
            if attempt < MAX_ATTEMPTS:
                yield {"type": "status", "stage": "retry", "format": api_format, "attempt": attempt, "message": str(reason)}
                yield {"type": "reset"}
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            raise RuntimeError(f"无法连接接口：{reason}{retried}") from error
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            if received:
                raise RuntimeError(f"无法解析上游流式输出：{error}") from error
            raise _StreamUnsupported(f"无法解析上游响应：{error}") from error


def stream_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str = "auto",
    system_prompt: str = "",
):
    """流式请求补全，yield 进度事件 dict，返回完整文本（通过 StopIteration.value）。"""
    formats = AUTO_FORMATS if api_format == "auto" else (api_format,)
    errors = []
    for index, candidate in enumerate(formats):
        if index:
            yield {"type": "status", "stage": "format", "format": candidate}
            yield {"type": "reset"}
        try:
            return (yield from _stream_format(
                base_url, api_key, api_model, prompt, temperature, candidate, system_prompt
            ))
        except _StreamUnsupported as stream_error:
            yield {"type": "status", "stage": "fallback", "format": candidate, "message": str(stream_error)}
            yield {"type": "reset"}
            try:
                text = _request_completion(
                    base_url, api_key, api_model, prompt, temperature, candidate, system_prompt
                )
            except RuntimeError as error:
                errors.append(f"{candidate}: {error}")
                continue
            yield {"type": "delta", "text": text, "chars": len(text), "parsed": len(parse_numbers(text))}
            return text
        except RuntimeError as error:
            errors.append(f"{candidate}: {error}")
    if len(formats) == 1:
        raise RuntimeError(errors[0].split(": ", 1)[1])
    raise RuntimeError("接口格式自动探测失败；" + "；".join(errors))


def request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str = "auto",
    system_prompt: str = "",
) -> str:
    if api_format != "auto":
        return _request_completion(
            base_url, api_key, api_model, prompt, temperature, api_format, system_prompt
        )
    formats = AUTO_FORMATS
    errors = []
    for candidate in formats:
        try:
            return _request_completion(
                base_url, api_key, api_model, prompt, temperature, candidate, system_prompt
            )
        except RuntimeError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("接口格式自动探测失败；" + "；".join(errors))


def test_automatic(
    base_url: str,
    api_key: str,
    api_model: str,
    temperature: float | None,
    bank: dict,
    api_format: str = "openai",
) -> dict:
    target_count = 3
    max_attempts = 6
    challenges = generate_challenges(max_attempts)
    outputs = []
    errors = []
    for challenge in challenges:
        try:
            text = request_completion(
                base_url,
                api_key,
                api_model,
                challenge["prompt"],
                temperature,
                api_format,
            )
            minimum = minimum_numbers(challenge["expected_count"])
            parsed_count = len(parse_numbers(text))
            if parsed_count >= minimum:
                outputs.append(
                    {
                        "text": text,
                        "expected_count": challenge["expected_count"],
                    }
                )
            else:
                errors.append(f"有效数字不足：{parsed_count}/{minimum}")
        except Exception as error:
            errors.append(str(error))
        if len(outputs) == target_count:
            break
    result = analyze_global_outputs(outputs, bank)
    result["api_test"] = {
        "requested": target_count,
        "attempted": len(outputs) + len(errors),
        "max_attempts": max_attempts,
        "received": len(outputs),
        "errors": errors,
    }
    return result


def enroll_automatic(
    base_url: str,
    api_key: str,
    api_model: str,
    model_label: str,
    sample_count: int,
    temperature: float | None,
    api_format: str = "openai",
    data_file: Path = DATA_FILE,
    bank_file: Path = BANK_FILE,
    bank_id: str = "reference-bank",
    provider: str = "api",
) -> dict:
    suite = fingerprint_suite()
    if sample_count < 3 or sample_count > len(suite):
        raise ValueError(f"采集回答数必须在 3 到 {len(suite)} 之间")
    selected = [suite[int(index * len(suite) / sample_count)] for index in range(sample_count)]
    rows = []
    errors = []

    def collect(task: dict) -> tuple[dict | None, list[str]]:
        task_errors = []
        base_prompt = task["prompt"]
        prompt = base_prompt
        if task["user_prefix"]:
            prompt = task["user_prefix"] + "\n\nFinal task:\n" + prompt
        system_prompt = task["system"]
        for _ in range(2):
            try:
                text = request_completion(
                    base_url,
                    api_key,
                    api_model,
                    prompt,
                    temperature,
                    api_format,
                    system_prompt,
                )
                row = make_row(
                    model_label=model_label,
                    text=text,
                    condition=task["condition"],
                    challenge_id=task["challenge_id"],
                    expected_count=task["expected_count"],
                    temperature=temperature if temperature is not None else "provider_default",
                    bank_id=bank_id,
                    wrapper_transport=task["transport"],
                    provider=provider,
                    prompt=prompt,
                    base_prompt=base_prompt,
                    system_prompt=system_prompt,
                    user_prefix=task["user_prefix"],
                )
                if row["strict_valid"]:
                    return row, task_errors
                task_errors.append(
                    f"{task['challenge_id']} 有效数字不足：{row['parsed_count']}/{row['strict_threshold']}"
                )
            except Exception as error:
                task_errors.append(f"{task['challenge_id']}: {error}")
        return None, task_errors

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(collect, task) for task in selected]
        for future in as_completed(futures):
            row, task_errors = future.result()
            errors.extend(task_errors)
            if row is not None:
                rows.append(row)
    accepted = [row for row in rows if row["strict_valid"]]
    if not accepted:
        raise ValueError("没有获得可用回答，指纹库未修改")
    append_rows(accepted, data_file)
    bank = rebuild_bank(data_file, bank_file)
    return {
        "requested": sample_count,
        "received": len(accepted),
        "accepted": len(accepted),
        "rejected": sample_count - len(accepted),
        "errors": errors,
        "bank": bank_summary(bank),
    }

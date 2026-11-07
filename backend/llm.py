"""
llm.py — one small function, chat(), over three provider families:

  openai     any OpenAI-compatible /chat/completions server
             (OpenAI, Ollama, Groq, vLLM, LM Studio, OpenRouter …) via LLM_BASE_URL
  anthropic  Anthropic Messages API
  gemini     Google Gemini generateContent

Behaviour:
  * json_mode=True returns a parsed dict. If the model replies with malformed
    JSON we retry ONCE, telling it to reply with valid JSON only.
  * 429 / 5xx / network errors are retried with exponential backoff
    (honouring Retry-After when present).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from .config import Settings, get_settings

log = logging.getLogger("llm")

MAX_HTTP_ATTEMPTS = 4


class LLMError(RuntimeError):
    """Raised with a human-friendly message when the LLM call fails."""


Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}


async def chat(system: str, messages: list[Message], json_mode: bool = True,
               max_tokens: int = 2048, settings: Settings | None = None, timeout: float | None = None,
               temperature: float | None = None, model: str | None = None) -> Any:
    s = settings or get_settings()
    if timeout:
        s = s.model_copy(update={"llm_timeout_seconds": max(timeout, s.llm_timeout_seconds)})
    if not s.resolved_llm_key and not (s.llm_provider == "openai" and s.llm_base_url):
        raise LLMError(f"No API key for LLM provider '{s.llm_provider}'. Add it to backend/.env.")

    text = await _call(s, system, messages, json_mode, max_tokens, temperature, model)
    if not json_mode:
        return text
    try:
        return parse_json(text)
    except ValueError:
        log.warning("event=llm_bad_json action=retry snippet=%r", text[:200])
        retry_msgs = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object, no prose, no code fences."},
        ]
        text = await _call(s, system, retry_msgs, json_mode, max_tokens, temperature, model)
        try:
            return parse_json(text)
        except ValueError as e:
            raise LLMError("The AI replied in an unexpected format twice. Please try again.") from e


def parse_json(text: str) -> Any:
    """Parse JSON, tolerating code fences, prose around the object and raw newlines inside strings."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start:end + 1], strict=False)
        except json.JSONDecodeError:
            pass
    raise ValueError("not JSON")


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

async def _call(s: Settings, system: str, messages: list[Message], json_mode: bool, max_tokens: int,
                temperature: float | None = None, model_override: str | None = None) -> str:
    provider = s.llm_provider
    model = model_override or s.resolved_llm_model
    t0 = time.perf_counter()
    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": s.anthropic_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        sys_text = system + ("\n\nRespond with a single JSON object only." if json_mode else "")
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": sys_text, "messages": messages}
    elif provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": s.gemini_api_key, "content-type": "application/json"}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": {"maxOutputTokens": max_tokens,
                                 **({"responseMimeType": "application/json"} if json_mode else {})},
        }
    else:  # OpenAI-compatible
        base = (s.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"content-type": "application/json"}
        if s.resolved_llm_key:
            headers["authorization"] = f"Bearer {s.resolved_llm_key}"
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}

    if temperature is not None:  # e.g. 0 for reproducible scoring
        if provider == "gemini":
            body["generationConfig"]["temperature"] = temperature
        else:
            body["temperature"] = temperature
    data = await _post_with_backoff(url, headers, body, s.llm_timeout_seconds, provider)

    try:
        if provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        elif provider == "gemini":
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        else:
            text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        log.error("event=llm_unexpected_response provider=%s body=%r", provider, str(data)[:300])
        raise LLMError("The AI service returned an empty or unexpected response. Please retry.") from e

    log.info("event=llm_done provider=%s model=%s seconds=%.2f chars=%d",
             provider, model, time.perf_counter() - t0, len(text))
    return text


async def _post_with_backoff(url: str, headers: dict[str, str], body: dict[str, Any],
                             timeout: float, provider: str) -> dict[str, Any]:
    delay = 2.0
    last_err = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
            try:
                r = await client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                last_err = f"network error: {e}"
                log.warning("event=llm_network_error provider=%s attempt=%d err=%s", provider, attempt, e)
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code in (401, 403):
                    raise LLMError(f"The {provider} API rejected the key. Check backend/.env.")
                if r.status_code == 404:
                    raise LLMError(f"Model not found on {provider}. Check LLM_MODEL / LLM_BASE_URL in backend/.env.")
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}"
                    retry_after = r.headers.get("retry-after")
                    wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay
                    log.warning("event=llm_retry provider=%s status=%s attempt=%d wait=%.1f",
                                provider, r.status_code, attempt, wait)
                    if attempt < MAX_HTTP_ATTEMPTS:
                        await asyncio.sleep(min(wait, 30))
                        delay *= 2
                        continue
                else:
                    log.error("event=llm_http_error provider=%s status=%s body=%r", provider, r.status_code, r.text[:400])
                    raise LLMError(f"The AI service returned an error ({r.status_code}). Please retry.")
            if attempt < MAX_HTTP_ATTEMPTS:
                await asyncio.sleep(delay)
                delay *= 2
    if "429" in last_err:
        raise LLMError("The AI service is rate-limiting requests. Wait a minute and try again.")
    raise LLMError(f"Couldn't reach the AI service ({last_err}). Check your connection and LLM settings.")

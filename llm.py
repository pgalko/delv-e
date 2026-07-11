"""
Multi-provider LLM client with streaming, cost tracking, and run logging.

Supports Anthropic, OpenAI, Ollama, and OpenRouter via provider:model syntax:
    anthropic:claude-opus-4-6
    openai:gpt-4o
    ollama:qwen3:30b
    openrouter:google/gemini-2.5-flash

Provider classes each implement call() and stream(). LLMClient routes to the
correct provider based on the model string and supplies any provider-specific
options (e.g. Ollama's reasoning_effort, chosen per agent role).
"""

import json
import os
import threading
import time

import httpx

from logger_config import get_logger
logger = get_logger(__name__)

# Stall prevention: if no data arrives for this many seconds during streaming,
# the connection is considered dead. Protects against provider hangs without
# killing legitimate slow generation (tokens keep the timer alive).
STREAM_TIMEOUT = httpx.Timeout(None, connect=30.0, read=120.0, write=30.0)


# Reasoning effort for models that expose a reasoning dial, chosen per agent role
# and able to step down on truncation (see call_with_ladder). The Executor is
# mechanical: it transcribes a closed spec into code with no analytical latitude,
# so a chain-of-thought buys it nothing, and a reasoning model in that seat can
# spend its entire output budget thinking and emit no code (observed with a kimi
# code model truncating at the output cap with zero visible characters).
# reasoning_effort="none" turns thinking off; the reasoning agents keep a real
# budget. Valid values: "none", "low", "medium", "high". Ollama honours this on its
# /v1 endpoint, and OpenRouter accepts the same dial (including "none") and relays
# it to the underlying provider's native control; direct OpenAI and Anthropic have
# no such kwarg here and are never sent it (see LLMClient._reasoning_extra). The
# native `think: false` flag does NOT work over Ollama's /v1, which is why this is
# expressed as reasoning_effort.
EXECUTOR_REASONING_EFFORT = "none"
DEFAULT_REASONING_EFFORT = "medium"

# Output-token ceiling for an agent turn, shared by every role. Ollama and
# OpenRouter take it as-is; the Anthropic clamp below applies on the direct
# Anthropic non-streaming path. Sized to let a heavy reasoning model (e.g. GLM-5.2
# at its max effort) finish its thinking AND emit a decision in a single turn.
DEFAULT_MAX_TOKENS = 64000
# Anthropic's SDK refuses a NON-STREAMING request whose max_tokens could run past
# its ~10-minute timeout, so a call routed to a direct `anthropic:` model is clamped
# to this. An `openrouter:anthropic/...` model uses the OpenAI-compatible OpenRouter
# path instead and is not subject to this guard.
ANTHROPIC_MAX_TOKENS = 20000

# A truncation retry holds the chosen effort once (with a directive to stop
# over-thinking), then drops straight to 'none' (see call_with_ladder and the
# Investigator loop). Walking intermediate rungs is unreliable across models, since
# some collapse low/medium to their max, so 'none' is the one dependable off switch.


def default_reasoning_effort(agent):
    """The effort an agent starts at: the Executor runs with thinking off, every
    other role at the standard default (overridable by the --reasoning-effort flag)."""
    return EXECUTOR_REASONING_EFFORT if agent == "Executor" else DEFAULT_REASONING_EFFORT


import uuid as _uuid

# One shared per-run UUID feeds every cache-affinity mechanism: the x-ai/*
# x-grok-conv-id header (xAI's cache is automatic but affinity-sensitive;
# without a stable id, requests land on different backend servers and hit
# rates collapse even for identical prefixes), the openai/* prompt_cache_key
# (mandated on GPT-5.6+ for reliable matching), and OpenRouter's session_id
# (pins sticky provider routing from the first request, before any hit is
# observed). One id per run (per process) maximizes hits across the
# Investigator's turns.
_RUN_ID = _uuid.uuid4().hex


def _usage_detail(usage, key):
    """A prompt_tokens_details field (cached_tokens, cache_write_tokens) from a
    usage object, robust to SDK versions that do not type the detail: falls
    back to the raw payload dict."""
    if not usage:
        return 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    v = getattr(ptd, key, None) if ptd is not None else None
    if v is None:
        try:
            d = usage.model_dump()
        except Exception:                                       # noqa: BLE001
            d = getattr(usage, "__dict__", {}) or {}
        v = ((d.get("prompt_tokens_details") or {})).get(key)
    return v or 0


def _reasoning_rejected(exc):
    """True for provider errors that reject the reasoning-effort FIELD itself
    (e.g. xAI: "Reasoning is mandatory for this endpoint and cannot be
    disabled"). The client strips the field and re-dispatches once instead of
    letting a rescue rung's experiment kill the run; the mapping floors are the
    first line of defense, this is the net for enums we have not met yet."""
    msg = str(exc).lower()
    return "reasoning" in msg and ("mandatory" in msg or "cannot be disabled" in msg)


def _xai_conv_headers(model):
    return {"x-grok-conv-id": _RUN_ID} if (model or "").startswith("x-ai/") else {}


def _openai_cache_body(model):
    """GPT-5.6+ requires `prompt_cache_key` for reliable cache matching (implicit
    and explicit). One key per run, same id as the xAI affinity header: both
    exist to pin a stable prefix to the same cache."""
    if not (model or "").startswith("openai/"):
        return {}
    return {
        # 5.6's mandated key for reliable matching; kept for GA. The provider pin
        # was tried and removed: routing was not the cause of the intra-run
        # misses (5.6 preview does no automatic prefix matching on this path).
        "prompt_cache_key": "delve-" + _RUN_ID,
    }


def _provider_effort(provider_name, effort, model=""):
    """Translate a canonical effort value to what the provider expects, or None to
    send no effort field. Ollama takes the value as-is; OpenRouter's top rung is
    'xhigh' rather than 'max'; direct OpenAI and Anthropic have no effort dial here
    (Anthropic uses a thinking-token budget, a separate mechanism).

    Model-aware floor: glm-5.x exposes only 'high' and 'max' and has no off
    switch; sending 'none' makes Ollama cloud return an instant empty completion.
    So 'none' is floored to 'high' for glm on either route. xAI reasoning
    models (grok-4.5) reject disabling reasoning outright (400: "Reasoning is
    mandatory for this endpoint and cannot be disabled"); their documented dial
    is low/medium/high with HIGH as the unspecified default, so 'none' floors
    to 'low', the lowest accepted rung (omitting the field would mean MORE
    reasoning, the opposite of the rescue's intent). Models that tolerate
    'none' (kimi and other non-thinking executors) are unaffected."""
    if provider_name not in ("ollama", "openrouter"):
        return None
    if provider_name == "ollama" and "glm" in (model or ""):
        # Ollama's /v1 for glm-5.2:cloud returns an instant empty completion
        # (finish_reason "stop", zero tokens) for ANY explicit reasoning_effort
        # value, every level verified empirically; omitting the field works. So
        # glm on Ollama gets no effort field at all and runs at its own default.
        # Consequence: no thinking off-switch for this combo, so the truncation
        # ladder's final "none" rescue degrades to a plain retry there.
        return None
    if effort == "none" and "glm" in (model or ""):
        effort = "high"   # OpenRouter route: documented enum is high/max only
    if effort == "none" and "x-ai/" in (model or ""):
        effort = "low"    # grok cannot disable reasoning; low is its bottom rung
    if provider_name == "openrouter":
        return "xhigh" if effort == "max" else effort
    return effort


# ══════════════════════════════════════════════════
# PRICING
# ══════════════════════════════════════════════════

# USD per million tokens — update when pricing changes
PRICING = {
    # Anthropic
    "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":            {"input": 5.00, "output": 25.00},
    "claude-opus-4-7":            {"input": 5.00, "output": 25.00},
    "claude-opus-4-8":            {"input": 5.00, "output": 25.00},
    # OpenAI
    "gpt-5.4":                    {"input": 2.50, "output": 15.00},
    "gpt-5.3-codex":              {"input": 1.75, "output": 14.00},
    "gpt-5-mini":                 {"input": 0.25, "output": 2.00},
    "o4-mini":                    {"input": 1.10, "output": 4.40},
    # OpenRouter — varies by model, add entries as needed
    # Pricing: https://openrouter.ai/models
    "moonshotai/kimi-k2.6":       {"input": 0.45, "output": 2.20},
    "z-ai/glm-5.2":               {"input": 1.40, "output": 4.40},
    "z-ai/glm-5.1":               {"input": 1.39, "output": 4.40},
    "deepseek/deepseek-v3.2":     {"input": 0.26, "output": 0.38},
    "qwen/qwen3.5-397b-a17b":     {"input": 0.39, "output": 2.34},
    "minimax/minimax-m3":         {"input": 0.30, "output": 1.20},
    "google/gemini-3.5-flash":    {"input": 1.5, "output": 9.00},
    "x-ai/grok-4.3":              {"input": 1.25, "output": 2.50},
    "x-ai/grok-4.5":              {"input": 2.0, "output": 6.0, "cached_input": 0.50},
    "openai/gpt-5.6-terra":       {"input": 2.50, "output": 15.0, "cached_input": 0.25, "cached_write": 3.125},
    "openai/gpt-5.6-luna":        {"input": 1.00, "output": 6.0, "cached_input": 0.10, "cached_write": 1.25},
    # Embeddings (output_tokens always 0)
    "text-embedding-3-small":     {"input": 0.02,  "output": 0.0},
    "text-embedding-3-large":     {"input": 0.13,  "output": 0.0},
    # Ollama (local, free)
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def compute_cost(model, input_tokens, output_tokens,
                 cache_creation_tokens=0, cache_read_tokens=0, cached_tokens=0,
                 cache_write_tokens=0):
    """Compute USD cost for a single API call.

    Cache pricing (Anthropic): a cache WRITE costs 1.25x the normal input rate,
    a cache READ costs 0.10x. `input_tokens` here is the fresh/uncached input
    only (Anthropic reports cached portions separately).

    Cache pricing (OpenAI-compatible, e.g. GPT-5.6 explicit breakpoints via
    OpenRouter): `input_tokens` INCLUDES both the cached and the written
    portions; `cached_tokens` bills at the model's cached rate and
    `cache_write_tokens` at its write rate. Semantics validated live against
    OpenRouter cost accounting to six decimals (terra: reads 0.25/M = the 90%
    discount, writes 3.125/M = 1.25x input).
    """
    # Strip provider prefix if present (e.g. 'openrouter:moonshotai/kimi-k2.5' → 'moonshotai/kimi-k2.5')
    model_name = model
    if ':' in model:
        parts = model.split(':', 1)
        if parts[0] in ('anthropic', 'openai', 'ollama', 'openrouter'):
            model_name = parts[1]
    pricing = PRICING.get(model_name, DEFAULT_PRICING)
    in_rate = pricing["input"] / 1_000_000
    out_rate = pricing["output"] / 1_000_000
    # OpenAI-compatible caches (OpenRouter/xAI/DeepSeek...): `input_tokens`
    # INCLUDES the cached portion; bill it at the model's cached rate when the
    # pricing table defines one, else at the full rate (never under-report).
    cached_rate = pricing.get("cached_input", pricing["input"]) / 1_000_000
    # Writes carry a premium, so the never-under-report fallback points the
    # other way: 1.25x input (OpenAI's stated 5.6+ rule, and the same premium
    # this function already charges Anthropic cache_creation).
    write_rate = pricing.get("cached_write", pricing["input"] * 1.25) / 1_000_000
    fresh_input = max(input_tokens - cached_tokens - cache_write_tokens, 0)
    return (fresh_input * in_rate
            + cached_tokens * cached_rate
            + cache_write_tokens * write_rate
            + cache_creation_tokens * in_rate * 1.25
            + cache_read_tokens * in_rate * 0.10
            + output_tokens * out_rate)


def _flatten_messages(messages):
    """Collapse any block-list message content into a plain string, dropping
    Anthropic cache_control. Used by non-Anthropic providers, which take string
    content and do not understand cache_control blocks.

    Exception: a part list carrying a prompt_cache_breakpoint (GPT-5.6 explicit
    caching, built by build_cached_messages for openrouter:openai/gpt-5.6*) is
    already in chat-completions shape and passes through untouched; flattening
    it would strip the very markers it exists to carry."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            if any(isinstance(b, dict) and "prompt_cache_breakpoint" in b
                   for b in content):
                out.append(msg)
                continue
            text = "".join(
                (b.get("text", "") if isinstance(b, dict) else str(b))
                for b in content
            )
            out.append({"role": msg["role"], "content": text})
        else:
            out.append(msg)
    return out


def _cache_usage(usage):
    """Extract (cache_creation, cache_read) input tokens from an Anthropic usage
    object; 0 when caching wasn't used or the fields are absent."""
    if usage is None:
        return 0, 0
    cc = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    return cc, cr


def _gpt56_explicit_cache(model):
    """True for models whose route accepts GPT-5.6 explicit cache breakpoints
    (prompt_cache_breakpoint on chat-completions content parts). Scoped to
    OpenRouter, where forwarding was verified live; the direct openai: path
    speaks the Responses API and its provider flattens content, so it is
    deliberately excluded until that path grows its own support."""
    return (model or "").startswith("openrouter:openai/gpt-5.6")


def build_cached_messages(model, system_text, stable, tail_text):
    """Build chat messages with prompt-cache breakpoints positioned so that an
    append-only history caches INCREMENTALLY.

    `stable` is the stable, append-only context: either a single string or a LIST
    of strings, one block per unit of history (e.g. one per completed step). The
    list form is important — Anthropic matches cached prefixes at BLOCK
    BOUNDARIES, so each historical unit must be its own block. If the whole
    history is concatenated into one growing block, the previous turn's cache
    boundary lands mid-block and is never matched (every turn re-writes the
    prefix instead of reading it). With per-unit blocks, the boundary at the end
    of the prior turn's last block stays fixed, so the next turn reads it back and
    writes only the newly appended block.

    Breakpoints (Anthropic allows up to 4): the system block, the FIRST stable
    block (so seed/schema is cached from turn 1 and as a fallback), and the LAST
    stable block (the moving frontier). `tail_text` is volatile and never cached.

    GPT-5.6 via OpenRouter gets the same first/last placement as content parts
    carrying prompt_cache_breakpoint markers (verified forwarded on chat
    completions despite docs saying Responses-only). The request stays in
    implicit mode, so OpenAI's free breakpoint on the latest message rides
    along. Whether the first-block fallback marker is honored alongside the
    frontier one is unverified live; worst case extras are ignored and
    frontier-only behavior remains. Part texts carry LEADING "\\n\\n" separators
    (never trailing) so appending a block or a new tail cannot rewrite the
    bytes of an already-cached part, and the concatenated prompt is
    byte-identical to the flattened form every other model receives.

    All other non-Anthropic providers get a single flattened string.
    """
    if isinstance(stable, str):
        stable_blocks = [stable] if stable else []
    else:
        stable_blocks = [b for b in stable if b]

    if (model or "").startswith("anthropic"):
        system = [{"type": "text", "text": system_text,
                   "cache_control": {"type": "ephemeral"}}]
        content = []
        last = len(stable_blocks) - 1
        for i, b in enumerate(stable_blocks):
            blk = {"type": "text", "text": b}
            if i == 0 or i == last:          # breakpoint on first and last stable block
                blk["cache_control"] = {"type": "ephemeral"}
            content.append(blk)
        content.append({"type": "text", "text": tail_text})   # volatile, uncached
        return [{"role": "system", "content": system},
                {"role": "user", "content": content}]

    if _gpt56_explicit_cache(model) and stable_blocks:
        content = []
        last = len(stable_blocks) - 1
        for i, b in enumerate(stable_blocks):
            blk = {"type": "text", "text": ("\n\n" + b) if i else b}
            if i == 0 or i == last:          # breakpoint on first and last stable block
                blk["prompt_cache_breakpoint"] = {"mode": "explicit"}
            content.append(blk)
        content.append({"type": "text", "text": "\n\n" + tail_text})   # volatile, uncached
        return [{"role": "system", "content": system_text},
                {"role": "user", "content": content}]

    user = ("\n\n".join(stable_blocks + [tail_text])
            if stable_blocks else tail_text)
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user}]


# Anthropic Opus 4.7+ deprecated temperature/top_p/top_k entirely — non-default
# values return HTTP 400 ("`temperature` is deprecated for this model.").
# These models must have the sampling params omitted from the request payload.
# Matches bare names ('claude-opus-4-7') and dated variants
# ('claude-opus-4-7-20260101', 'claude-opus-4-8-20260315'), and any future
# Opus 4.x where x >= 7. Update if Anthropic restores the params or extends
# the deprecation to other model families.
_NO_SAMPLING_PARAM_PREFIXES = (
    'claude-opus-4-7',
    'claude-opus-4-8',
)


def _omits_sampling_params(model):
    """Return True if the model rejects temperature/top_p/top_k at request time."""
    if not model:
        return False
    return any(model.startswith(p) for p in _NO_SAMPLING_PARAM_PREFIXES)


# ══════════════════════════════════════════════════
# PROVIDERS
# ══════════════════════════════════════════════════

class AnthropicProvider:
    """Anthropic Claude API."""

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found. Set it in .env or environment.")
        self.client = anthropic.Client(api_key=api_key)
        self._api_error = anthropic.APIError

    def call(self, messages, model, max_tokens, temperature):
        self._last_cached = 0
        self._last_cache_write = 0
        self._last_reasoning_chars = 0
        system_msg, api_messages = self._split_system(messages)
        kwargs = dict(
            model=model, system=system_msg, messages=api_messages,
            max_tokens=max_tokens,
        )
        if not _omits_sampling_params(model):
            kwargs['temperature'] = temperature
        try:
            response = self.client.messages.create(**kwargs)
        except self._api_error as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text or ""
        cc, cr = _cache_usage(response.usage)
        return (
            content,
            response.usage.input_tokens,
            response.usage.output_tokens,
            cc, cr,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        system_msg, api_messages = self._split_system(messages)
        kwargs = dict(
            model=model, system=system_msg, messages=api_messages,
            max_tokens=max_tokens, timeout=STREAM_TIMEOUT,
        )
        if not _omits_sampling_params(model):
            kwargs['temperature'] = temperature
        collected = []
        cc = cr = 0
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if hasattr(event, 'type') and event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            collected.append(event.delta.text)
                            on_token(event.delta.text)
                final = stream.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens
                cc, cr = _cache_usage(final.usage)
        except self._api_error as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        return ''.join(collected), input_tokens, output_tokens, cc, cr

    def _split_system(self, messages):
        """Anthropic requires system message separate from conversation."""
        system = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation.append(msg)
        return system, conversation

    def search_call(self, messages, model, max_tokens, temperature, max_uses=5):
        """Call with web search tool enabled. Returns synthesised text from all content blocks."""
        system_msg, api_messages = self._split_system(messages)
        try:
            tool_type = os.environ.get("ANTHROPIC_WEB_SEARCH_TOOL_TYPE", "web_search_20250305")
            tool = {"type": tool_type, "name": "web_search", "max_uses": max_uses}
            kwargs = dict(
                model=model, system=system_msg, messages=api_messages,
                max_tokens=max_tokens, tools=[tool],
            )
            if not _omits_sampling_params(model):
                kwargs['temperature'] = temperature
            response = self.client.messages.create(**kwargs)
        except self._api_error as e:
            logger.error(f"Anthropic search API error: {e}")
            raise
        # Extract text from all content blocks (response includes text + search result blocks)
        text_parts = []
        for block in (response.content or []):
            if hasattr(block, 'text') and block.text:
                text_parts.append(block.text)
        content = "\n".join(text_parts)
        return (
            content,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )


class OpenAIProvider:
    """OpenAI Responses API (GPT-5.x, Codex, o-series, GPT-4o, etc.)."""

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found. Set it in .env or environment.")
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _split_system(self, messages):
        """Extract system/instructions from messages for the Responses API."""
        instructions = None
        input_messages = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_messages.append(msg)
        return instructions, input_messages

    def call(self, messages, model, max_tokens, temperature):
        instructions, input_messages = self._split_system(_flatten_messages(messages))
        params = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if instructions:
            params["instructions"] = instructions
        response = self.client.responses.create(**params)
        return (
            response.output_text or "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            0, 0,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        instructions, input_messages = self._split_system(_flatten_messages(messages))
        params = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        collected = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.responses.create(**params, timeout=STREAM_TIMEOUT)
        for event in stream:
            if event.type == "response.output_text.delta":
                collected.append(event.delta)
                on_token(event.delta)
            elif event.type == "response.completed":
                if event.response and event.response.usage:
                    input_tokens = event.response.usage.input_tokens
                    output_tokens = event.response.usage.output_tokens
        return ''.join(collected), input_tokens, output_tokens, 0, 0


def _ollama_extras(model, reasoning_effort):
    """(extra_body, extra_headers) for Ollama's /v1 endpoint: just the effort
    field, and only when set. A falsy effort must never serialize; glm on
    Ollama arrives here as None precisely so the field is omitted at the wire
    (see _provider_effort)."""
    return ({"reasoning_effort": reasoning_effort} if reasoning_effort else {}), {}


def _openrouter_extras(model, reasoning_effort):
    """(extra_body, extra_headers) for OpenRouter: the effort dict only when
    set, the openai/* prompt cache key, detailed usage accounting, and the
    x-ai/* cache-affinity header."""
    body = {**({"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}),
            **_openai_cache_body(model),
            "usage": {"include": True},   # detailed accounting incl. cached tokens
            # Sticky provider routing from the first request. OpenRouter's
            # default conversation hash keys on the first non-system message,
            # which delv-e REWRITES every turn, so without the explicit id
            # stickiness never engages for multi-provider models.
            "session_id": _RUN_ID}
    return body, _xai_conv_headers(model)


class OpenAICompatProvider:
    """Shared provider for the chat-completions-shaped OpenAI-compatible
    endpoints: Ollama and OpenRouter.

    Ollama serves local models and cloud models (e.g. gpt-oss:120b-cloud);
    cloud models are offloaded automatically by the local server, just run
    'ollama signin' and 'ollama pull <model>' first. OpenRouter fronts
    hundreds of models named 'vendor/model' (usage:
    openrouter:google/gemini-2.5-flash; docs:
    https://openrouter.ai/docs/quickstart).

    The two differ only in connection details and per-request extras, so the
    registry holds zero-arg factories (_ollama_provider, _openrouter_provider)
    and every provider quirk lives in the `extras` builder, which maps
    (model, reasoning_effort) to (extra_body, extra_headers): the conditional
    effort field (a falsy effort never serializes), OpenRouter's detailed
    usage accounting, the openai/* prompt cache key, and the x-ai/* affinity
    header. Direct OpenAI is NOT this shape (it speaks the Responses API) and
    stays a separate class, as does Anthropic (thinking budget, cache_control,
    cache telemetry in its return tuple).
    """

    def __init__(self, base_url, api_key, extras, default_headers=None):
        from openai import OpenAI
        kwargs = {"default_headers": default_headers} if default_headers else {}
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self._extras = extras

    def call(self, messages, model, max_tokens, temperature, reasoning_effort="medium"):
        extra_body, extra_headers = self._extras(model, reasoning_effort)
        response = self.client.chat.completions.create(
            model=model,
            messages=_flatten_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
        _u = response.usage
        self._last_cached = _usage_detail(_u, "cached_tokens")
        self._last_cache_write = _usage_detail(_u, "cache_write_tokens")
        _msg = response.choices[0].message if response.choices else None
        self._last_reasoning_chars = len(
            (getattr(_msg, "reasoning", None) or
             getattr(_msg, "reasoning_content", None) or "")) if _msg else 0
        return (
            content,
            _u.prompt_tokens if _u else 0,
            _u.completion_tokens if _u else 0,
            0, 0,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token, reasoning_effort="medium"):
        extra_body, extra_headers = self._extras(model, reasoning_effort)
        collected = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.chat.completions.create(
            model=model,
            messages=_flatten_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            timeout=STREAM_TIMEOUT,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
        self._last_reasoning_chars = 0
        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                self._last_cached = _usage_detail(chunk.usage, "cached_tokens")
                self._last_cache_write = _usage_detail(chunk.usage, "cache_write_tokens")
            _d = chunk.choices[0].delta if chunk.choices else None
            _r = (getattr(_d, "reasoning", None) or
                  getattr(_d, "reasoning_content", None)) if _d else None
            if _r:
                # Reasoning-channel text: counted for diagnostics, NEVER merged
                # into content (thinking may contain drafts; the Executor must
                # never receive it as code).
                self._last_reasoning_chars += len(_r)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected.append(text)
                on_token(text)
        return ''.join(collected), input_tokens, output_tokens, 0, 0


def _ollama_provider():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return OpenAICompatProvider(f"{base_url}/v1", "ollama", _ollama_extras)


def _openrouter_provider():
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPEN_ROUTER_API_KEY not found. Set it in .env or environment.")
    base_url = os.environ.get("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAICompatProvider(
        base_url, api_key, _openrouter_extras,
        default_headers={
            "HTTP-Referer": "https://github.com/pgalko/delv-e",
            "X-Title": "delv-e",
        })


# ══════════════════════════════════════════════════
# PROVIDER REGISTRY
# ══════════════════════════════════════════════════

# Values are zero-arg callables returning a provider instance: classes for the
# genuinely distinct APIs, factories for the two endpoints that share
# OpenAICompatProvider and differ only in construction.
PROVIDER_CLASSES = {
    "anthropic":    AnthropicProvider,
    "openai":       OpenAIProvider,
    "ollama":       _ollama_provider,
    "openrouter":   _openrouter_provider,
}

DEFAULT_PROVIDER = "anthropic"


def parse_model_string(model_string):
    """
    Parse 'provider:model' syntax. Returns (provider_name, model_name).
    If no provider prefix, returns (DEFAULT_PROVIDER, model_string).
    
    Examples:
        'anthropic:claude-opus-4-6'             -> ('anthropic', 'claude-opus-4-6')
        'openai:gpt-4o'                         -> ('openai', 'gpt-4o')
        'ollama:qwen3:30b'                      -> ('ollama', 'qwen3:30b')
        'openrouter:google/gemini-2.5-flash'    -> ('openrouter', 'google/gemini-2.5-flash')
        'claude-opus-4-6'                       -> ('anthropic', 'claude-opus-4-6')
    """
    parts = model_string.split(":", 1)
    if len(parts) == 2 and parts[0] in PROVIDER_CLASSES:
        return parts[0], parts[1]
    return DEFAULT_PROVIDER, model_string


def call_with_ladder(client, messages, model, agent=None, max_tokens=None,
                     temperature=0, reasoning_effort=None):
    """Call the client at the chosen reasoning effort, and if the turn comes back
    empty or capped, retry ONCE with reasoning turned off, for providers that have a
    reasoning dial (Ollama, OpenRouter); other providers make a single call. Returns
    (text, meta).

    'none' is the one value every such provider honors as off, so it is the reliable
    circuit-breaker. Stepping through intermediate rungs is not reliable across models
    (some collapse low/medium to their max), so we hold the chosen effort, then drop
    straight to none rather than walking a ladder.

    Defined as a free function over `client` rather than a method so it works with the
    lightweight client stubs in the test suite, which implement only call(): a stub
    that does not accept the meta/effort kwargs simply gets one plain call.
    """
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    provider_name, _ = parse_model_string(model)
    ladders = provider_name in ("ollama", "openrouter")
    effort = reasoning_effort or default_reasoning_effort(agent)
    # Hold the chosen effort; fall back to 'none' once if it came back empty/capped.
    attempts = [effort, "none"] if (ladders and effort != "none") else [effort]
    for i, eff in enumerate(attempts):
        try:
            text, meta = client.call(
                messages, model, max_tokens=max_tokens, temperature=temperature,
                agent=agent, reasoning_effort=(eff if ladders else None),
                return_meta=True)
        except TypeError:
            # Stub client without return_meta/reasoning_effort: one plain call.
            return client.call(messages, model, max_tokens=max_tokens,
                               temperature=temperature, agent=agent), {}
        ok = bool((text or "").strip()) and not meta.get("truncated")
        if ok or i == len(attempts) - 1:
            meta["effort_used"] = eff if ladders else None
            return text, meta
        logger.info("%s turn came back empty/capped at reasoning_effort=%s; "
                    "retrying with reasoning off.", agent or "model", eff)


# ══════════════════════════════════════════════════
# COST TRACKING & LOGGING (unchanged)
# ══════════════════════════════════════════════════

class CostTracker:
    """Accumulates token counts and cost across all calls in a run.

    Thread-safe: record() may be called concurrently from parallel question
    processing threads. The lock guards the multi-step read-modify-write of
    the running totals.
    """

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0            # fresh / uncached input
        self.output_tokens = 0
        self.cache_creation_tokens = 0   # cache writes
        self.cache_read_tokens = 0       # cache hits (Anthropic)
        self.cached_tokens = 0           # cache hits (OpenAI-compatible)
        self.cache_write_tokens = 0      # cache writes (OpenAI-compatible, billed 1.25x)
        self.total_cost = 0.0
        self.total_cost_uncached = 0.0   # counterfactual: price every input token at full rate
        self._lock = threading.Lock()

    def record(self, input_tokens, output_tokens, model=None,
               cache_creation_tokens=0, cache_read_tokens=0, cached_tokens=0,
               cache_write_tokens=0):
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_creation_tokens += cache_creation_tokens
            self.cache_read_tokens += cache_read_tokens
            self.cached_tokens += cached_tokens
            self.cache_write_tokens += cache_write_tokens
            self.total_cost += compute_cost(
                model or "", input_tokens, output_tokens,
                cache_creation_tokens, cache_read_tokens, cached_tokens,
                cache_write_tokens)
            # What this call would have cost with no caching: all input at full rate.
            total_in = input_tokens + cache_creation_tokens + cache_read_tokens
            self.total_cost_uncached += compute_cost(model or "", total_in, output_tokens)

    def report(self):
        total_input = self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens
        lines = [f"{self.calls} API calls | ${self.total_cost:.4f}"]
        if self.cache_creation_tokens or self.cache_read_tokens or self.cached_tokens:
            hit = (100 * (self.cache_read_tokens + self.cached_tokens) / total_input) if total_input else 0
            saved = self.total_cost_uncached - self.total_cost
            pct = (100 * saved / self.total_cost_uncached) if self.total_cost_uncached else 0
            lines.append(
                f"Input: {self.input_tokens:,} fresh + {self.cache_read_tokens:,} "
                f"cached-read + {self.cache_creation_tokens:,} cache-write | "
                f"{self.output_tokens:,} output")
            lines.append(
                f"Cache hit rate: {hit:.0f}%  |  est. without caching "
                f"${self.total_cost_uncached:.4f}  →  saved ${saved:.4f} ({pct:.0f}%)")
        else:
            lines.append(
                f"{self.input_tokens:,} input + {self.output_tokens:,} output tokens")
        return "\n".join(lines)


class RunLogger:
    """Append-only run log. Flushes to disk after each call.

    Thread-safe: log() may be called concurrently from parallel question
    processing threads. The lock serialises both the in-memory append and
    the disk flush so two threads can't interleave their JSON writes.
    """

    def __init__(self, path, append=False):
        self.path = path
        self.entries = []
        self._lock = threading.Lock()
        if append and os.path.exists(path):
            try:
                with open(path) as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def log(self, agent, model, messages, response,
            input_tokens, output_tokens, elapsed_time, ttft=None,
            cache_creation=0, cache_read=0, cached_tokens=0, cache_write_tokens=0,
            reasoning_chars=0):
        cost = compute_cost(model, input_tokens, output_tokens, cache_creation, cache_read,
                            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens)
        entry = {
            "agent": agent,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "reasoning_chars": reasoning_chars,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "total_tokens": input_tokens + output_tokens + cache_creation + cache_read,
            "elapsed_time_s": round(elapsed_time, 2),
            "tokens_per_second": round(output_tokens / max(elapsed_time, 0.01), 1),
            "cost_usd": round(cost, 6),
            "input": messages,
            "output": response,
        }
        if ttft is not None:
            entry["ttft_s"] = ttft
        with self._lock:
            self.entries.append(entry)
            self._flush()

    def _flush(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.entries, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write run log: {e}")

    def summary(self):
        if not self.entries:
            return ""
        agent_stats = {}
        for e in self.entries:
            a = e["agent"] or "unknown"
            if a not in agent_stats:
                agent_stats[a] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0,
                                  "time": 0.0, "cache_read": 0, "cache_write": 0}
            agent_stats[a]["calls"] += 1
            agent_stats[a]["input"] += e["input_tokens"]
            agent_stats[a]["output"] += e["output_tokens"]
            agent_stats[a]["cost"] += e["cost_usd"]
            agent_stats[a]["time"] += e["elapsed_time_s"]
            agent_stats[a]["cache_read"] += e.get("cache_read_input_tokens", 0)
            agent_stats[a]["cache_write"] += e.get("cache_creation_input_tokens", 0)

        lines = ["Per-agent breakdown:"]
        for a, s in sorted(agent_stats.items()):
            cache = ""
            if s["cache_read"] or s["cache_write"]:
                cache = f" | cache: {s['cache_read']:,} read + {s['cache_write']:,} write"
            lines.append(
                f"  {a:25s} {s['calls']:3d} calls | "
                f"{s['input']:>8,} in + {s['output']:>7,} out | "
                f"{s['time']:>6.1f}s | ${s['cost']:.4f}{cache}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════
# RUN STATS + TELEMETRY
# ══════════════════════════════════════════════════

class RunStats:
    """Lightweight event sink for one run, threaded through the loop the same way
    CostTracker is. It records loop DECISIONS (gate overrides, synthesizer
    pushbacks, truncation retries, searches, provisional fallback) that are not
    API-call numbers; the per-call token/cost/timing numbers live in RunLogger,
    and per-step retries/failures are derived from the step log. Counting these
    here keeps the dissect-relevant signal even though the raw step log of a past
    run is not archived per timestamped folder."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts = {}
        self.flags = {}

    def bump(self, name, n=1):
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + n

    def flag(self, name, value=True):
        with self._lock:
            self.flags[name] = value

    def get(self, name, default=0):
        with self._lock:
            return self.counts.get(name, default)

    def as_dict(self):
        with self._lock:
            return {"counts": dict(self.counts), "flags": dict(self.flags)}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def build_run_telemetry(run_logger, cost_tracker, run_stats, step_log,
                        *, seed="", dataset_shape=(0, 0), models=None,
                        max_iters=0, wall_clock_s=0.0,
                        target_estimand="", final_verdict="none"):
    """Aggregate one run into a telemetry dict for run_telemetry.json.

    Most fields are a reduce over data that already exists: per-call entries in
    the run logger, running totals in the cost tracker, per-step `attempts`/`error`
    in the step log, and the loop events in run_stats. Only wall_clock_s is
    measured outside (a run-level timer in the caller).
    """
    entries = list(getattr(run_logger, "entries", []) or [])
    models = models or {}
    stats = run_stats.as_dict() if run_stats is not None else {"counts": {}, "flags": {}}
    counts, flags = stats["counts"], stats["flags"]

    # ---- per-agent rollup from the run log -------------------------------
    per_agent = {}
    for e in entries:
        a = e.get("agent") or "unknown"
        s = per_agent.setdefault(a, {"calls": 0, "input": 0, "output": 0,
                                     "cached": 0, "cached_write": 0,
                                     "cache_read": 0, "cache_write": 0,
                                     "cost_usd": 0.0, "time_s": 0.0, "_ttfts": []})
        s["calls"] += 1
        s["input"] += e.get("input_tokens", 0)
        s["output"] += e.get("output_tokens", 0)
        s["cached"] += e.get("cached_tokens", 0)
        s["cached_write"] += e.get("cache_write_tokens", 0)
        s["cache_read"] += e.get("cache_read_input_tokens", 0)
        s["cache_write"] += e.get("cache_creation_input_tokens", 0)
        s["cost_usd"] += e.get("cost_usd", 0.0)
        s["time_s"] += e.get("elapsed_time_s", 0.0)
        if e.get("ttft_s") is not None:
            s["_ttfts"].append(e["ttft_s"])
    for s in per_agent.values():
        s["cost_usd"] = round(s["cost_usd"], 6)
        s["time_s"] = round(s["time_s"], 2)
        s["avg_ttft_s"] = _mean(s.pop("_ttfts"))

    # ---- call-level aggregates -------------------------------------------
    api_time_s = round(sum(e.get("elapsed_time_s", 0.0) for e in entries), 2)
    total_output = sum(e.get("output_tokens", 0) for e in entries)
    avg_ttft = _mean([e.get("ttft_s") for e in entries])
    tok_per_s = round(total_output / api_time_s, 1) if api_time_s else None

    # ---- reliability from the step log -----------------------------------
    steps = [e for e in (step_log or []) if not e.get("terminal")]
    iters_used = len(steps)
    executor_retries = sum(max(0, (e.get("attempts", 1) or 1) - 1) for e in steps)
    failed_steps = [e.get("step") for e in steps if e.get("error")]

    # token caps hit: reuse the existing truncation detection (the Investigator
    # path flags meta["truncated"] and retries). Non-Anthropic providers are
    # best-effort. This is the count of Investigator truncations, not a separate
    # stop-reason probe.
    inv_truncations = counts.get("investigator_truncation_retries", 0)

    cache_read = getattr(cost_tracker, "cache_read_tokens", 0)
    cache_write = getattr(cost_tracker, "cache_creation_tokens", 0)
    fresh_in = getattr(cost_tracker, "input_tokens", 0)
    out_tokens = getattr(cost_tracker, "output_tokens", 0)
    total_in = fresh_in + cache_read + cache_write
    total_cost = round(getattr(cost_tracker, "total_cost", 0.0), 6)
    uncached = round(getattr(cost_tracker, "total_cost_uncached", 0.0), 6)
    hit_rate = round(100 * cache_read / total_in, 1) if total_in else 0.0

    summary_text = ""
    try:
        summary_text = (cost_tracker.report() + "\n\n" + (run_logger.summary() or "")).strip()
    except Exception:
        pass

    return {
        "run": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "dataset": {"rows": dataset_shape[0], "cols": dataset_shape[1]},
            "models": models,
            "iterations_used": iters_used,
            "iterations_max": max_iters,
            "final_verdict": final_verdict,
            "wall_clock_s": round(wall_clock_s, 2),
            "api_time_s": api_time_s,
            "code_and_overhead_s": round(max(0.0, wall_clock_s - api_time_s), 2),
        },
        "cost": {
            "total_usd": total_cost,
            "without_cache_usd": uncached,
            "saved_usd": round(uncached - total_cost, 6),
            "cache_hit_rate_pct": hit_rate,
        },
        "tokens": {
            "input_fresh": fresh_in,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "output": out_tokens,
            "total": total_in + out_tokens,
        },
        "calls": {
            "total": getattr(cost_tracker, "calls", len(entries)),
            "avg_ttft_s": avg_ttft,
            "tokens_per_second": tok_per_s,
        },
        "per_agent": per_agent,
        "reliability": {
            "executor_retries": executor_retries,
            "failed_steps": failed_steps,
            "investigator_truncation_retries": inv_truncations,
            "token_caps_hit": inv_truncations,
        },
        "gates": {
            "g1_gate_overrides": counts.get("g1_gate_overrides", 0),
            "synth_pushbacks": counts.get("synth_pushbacks", 0),
            "provisional_briefing": bool(flags.get("provisional_briefing", False)),
            "searches": counts.get("searches", 0),
            "budget_wrapup_notices": counts.get("budget_wrapup_notices", 0),
            "synth_briefing_retries": counts.get("synth_briefing_retries", 0),
        },
        "estimand": {
            "named": bool((target_estimand or "").strip()),
            "text": target_estimand or "",
        },
        "summary_text": summary_text,
    }


# ══════════════════════════════════════════════════
# LLM CLIENT
# ══════════════════════════════════════════════════

class LLMClient:
    """
    Multi-provider LLM client. Routes calls to the correct provider
    based on 'provider:model' syntax in the model string.
    
    Providers are initialized lazily on first use.
    """

    def __init__(self, cost_tracker=None, run_logger=None, progress=False):
        self.cost_tracker = cost_tracker or CostTracker()
        self.run_logger = run_logger
        self._providers = {}  # lazily initialized
        self._provider_lock = threading.Lock()
        # Optional "while waiting" spinner. Lazy-imported so llm.py works even if
        # ui.py is absent (e.g. library use); silently disabled if unavailable.
        self._Spinner = None
        if progress:
            try:
                from ui import Spinner
                self._Spinner = Spinner
            except Exception:
                self._Spinner = None

    def _get_provider(self, provider_name):
        """Get or initialize a provider instance.

        Thread-safe via double-checked locking: the fast path (provider already
        initialised) avoids lock acquisition; the slow path (first hit per
        provider) serialises concurrent initialisations so two threads can't
        both construct the same provider.
        """
        if provider_name in self._providers:
            return self._providers[provider_name]
        with self._provider_lock:
            if provider_name in self._providers:
                return self._providers[provider_name]
            if provider_name not in PROVIDER_CLASSES:
                raise ValueError(
                    f"Unknown provider '{provider_name}'. "
                    f"Available: {', '.join(PROVIDER_CLASSES.keys())}"
                )
            self._providers[provider_name] = PROVIDER_CLASSES[provider_name]()
            return self._providers[provider_name]

    @staticmethod
    def _reasoning_extra(provider_name, agent, override=None, model=""):
        """Provider kwargs carrying the reasoning effort for this turn. Ollama and
        OpenRouter both accept a reasoning_effort kwarg (each formats it for its own
        API); direct OpenAI and Anthropic have no such control and get nothing.
        `override` is the chosen effort for this attempt; without it the agent's
        default applies. The value is translated per provider (e.g. 'max' becomes
        OpenRouter's 'xhigh'). Keyed on the agent label so the client-call interface
        (and every mock that implements it) stays unchanged."""
        effort = override or default_reasoning_effort(agent)
        mapped = _provider_effort(provider_name, effort, model)
        if provider_name not in ("ollama", "openrouter"):
            return {}
        return {"reasoning_effort": mapped}   # None = omit at the wire

    def _sync_wire(self, provider):
        """Pull the wire side-channels the provider captured for the last
        request: cache reads, cache writes, and reasoning-channel size."""
        self._last_cached = getattr(provider, "_last_cached", 0)
        self._last_cache_write = getattr(provider, "_last_cache_write", 0)
        self._last_reasoning_chars = getattr(provider, "_last_reasoning_chars", 0)

    def call(self, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=0, agent=None,
             return_meta=False, reasoning_effort=None):
        """Non-streaming call. Returns response text (always a string, never None).
        When return_meta=True, returns (text, meta) where meta carries token usage
        and a `truncated` flag (the model hit the output-token cap). reasoning_effort
        overrides the agent default on providers with a reasoning dial. Direct
        Anthropic is clamped to its safe non-streaming ceiling."""
        provider_name, model_name = parse_model_string(model)
        provider = self._get_provider(provider_name)
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        if provider_name == "anthropic" and max_tokens > ANTHROPIC_MAX_TOKENS:
            max_tokens = ANTHROPIC_MAX_TOKENS  # non-streaming SDK timeout guard
        extra = self._reasoning_extra(provider_name, agent, reasoning_effort, model=model)
        start_time = time.time()

        from contextlib import nullcontext
        spin = self._Spinner(agent or "working") if self._Spinner else nullcontext()
        try:
            with spin:
                for effort_try in range(2):
                    try:
                        # Empty-completion guard: dispatch once, retry once loudly if
                        # the content came back blank (the warning names the
                        # reasoning-channel size so channel misroutes and silent
                        # drops are distinguishable).
                        for attempt in range(2):
                            content, input_tokens, output_tokens, cache_creation, cache_read = provider.call(
                                messages, model_name, max_tokens, temperature, **extra
                            )
                            self._sync_wire(provider)
                            if (content or "").strip() or attempt:
                                break
                            logger.warning(
                                "Empty completion from %s/%s (agent=%s, reasoning_chars=%s): "
                                "output likely landed on the reasoning channel or the request "
                                "was silently dropped. Retrying once.",
                                provider_name, model_name, agent, self._last_reasoning_chars)
                        break
                    except Exception as e:
                        # A provider rejecting the reasoning-effort FIELD must not
                        # kill the run: strip the field and re-dispatch once.
                        if (effort_try == 0 and extra.get("reasoning_effort")
                                and _reasoning_rejected(e)):
                            logger.warning(
                                "%s rejected reasoning_effort=%r for %s; retrying with "
                                "the field omitted.", provider_name,
                                extra["reasoning_effort"], model_name)
                            extra = {**extra, "reasoning_effort": None}
                            continue
                        raise
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            raise

        # Guarantee string return — providers should already handle this,
        # but belt-and-suspenders against None leaking through.
        content = content or ""

        elapsed = time.time() - start_time
        self.cost_tracker.record(input_tokens, output_tokens, model_name,
                                 cache_creation, cache_read, cached_tokens=self._last_cached,
                                 cache_write_tokens=self._last_cache_write)

        if self.run_logger:
            self.run_logger.log(agent, f"{provider_name}:{model_name}", messages,
                                content, input_tokens, output_tokens, elapsed,
                                cache_creation=cache_creation, cache_read=cache_read, cached_tokens=self._last_cached, cache_write_tokens=self._last_cache_write, reasoning_chars=self._last_reasoning_chars)
        if return_meta:
            return content, {"input_tokens": input_tokens,
                             "output_tokens": output_tokens,
                             "cached_tokens": getattr(self, "_last_cached", 0),
                             "cache_write_tokens": getattr(self, "_last_cache_write", 0),
                             "max_tokens": max_tokens,
                             "truncated": bool(output_tokens) and output_tokens >= max_tokens}
        return content

    def stream(self, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=0,
               output_manager=None, chain_id=None, agent=None, reasoning_effort=None):
        """Streaming call. Returns full response text."""
        provider_name, model_name = parse_model_string(model)
        provider = self._get_provider(provider_name)
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        extra = self._reasoning_extra(provider_name, agent, reasoning_effort, model=model)
        start_time = time.time()
        first_token_time = [None]  # mutable container for closure

        def on_token(text):
            if first_token_time[0] is None:
                first_token_time[0] = time.time()
            if output_manager:
                output_manager.print_wrapper(text, end='', flush=True, chain_id=chain_id)

        try:
            for effort_try in range(2):
                try:
                    # Same empty-completion guard shape as call(): dispatch once, retry
                    # once loudly. The retry reuses on_token, so a rescued stream still
                    # renders live.
                    for attempt in range(2):
                        content, input_tokens, output_tokens, cache_creation, cache_read = provider.stream(
                            messages, model_name, max_tokens, temperature, on_token, **extra
                        )
                        self._sync_wire(provider)
                        if (content or "").strip() or attempt:
                            break
                        logger.warning(
                            "Empty streamed completion (agent=%s, reasoning_chars=%s): output "
                            "likely landed on the reasoning channel. Retrying once.",
                            agent, self._last_reasoning_chars)
                    break
                except Exception as e:
                    if (effort_try == 0 and extra.get("reasoning_effort")
                            and _reasoning_rejected(e)):
                        logger.warning(
                            "%s rejected reasoning_effort=%r for %s; retrying with "
                            "the field omitted.", provider_name,
                            extra["reasoning_effort"], model_name)
                        extra = {**extra, "reasoning_effort": None}
                        continue
                    raise
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            raise

        if output_manager and not output_manager.silent_mode:
            output_manager.print_wrapper("", chain_id=chain_id)

        elapsed = time.time() - start_time
        ttft = round(first_token_time[0] - start_time, 2) if first_token_time[0] else None
        self.cost_tracker.record(input_tokens, output_tokens, model_name,
                                 cache_creation, cache_read, cached_tokens=self._last_cached,
                                 cache_write_tokens=self._last_cache_write)

        if self.run_logger:
            self.run_logger.log(agent, f"{provider_name}:{model_name}", messages,
                                content, input_tokens, output_tokens, elapsed, ttft,
                                cache_creation=cache_creation, cache_read=cache_read, cached_tokens=self._last_cached, cache_write_tokens=self._last_cache_write, reasoning_chars=self._last_reasoning_chars)
        return content

    # Haiku is used for search regardless of search_model because web search
    # returns massive input tokens from retrieved pages. The provider is still
    # validated from search_model, but the concrete model is pinned here to
    # control cost. The search tool type itself is configured in
    # AnthropicProvider.search_call and defaults to the broadly compatible
    # web_search_20250305 tool.
    SEARCH_MODEL_OVERRIDE = "claude-haiku-4-5-20251001"

    def search_call(self, messages, model, max_tokens=8000, temperature=0,
                    agent=None, max_uses=5):
        """Non-streaming call with web search tool. Anthropic only.
        Returns response text with search results synthesised.
        Always uses Haiku to control cost from large search result inputs."""
        provider_name, _ = parse_model_string(model)
        if provider_name != 'anthropic':
            raise ValueError(f"Web search requires Anthropic provider, got '{provider_name}'")
        provider = self._get_provider(provider_name)
        model_name = self.SEARCH_MODEL_OVERRIDE
        start_time = time.time()

        try:
            content, input_tokens, output_tokens = provider.search_call(
                messages, model_name, max_tokens, temperature, max_uses=max_uses
            )
        except Exception as e:
            logger.error(f"Search API error: {e}")
            raise

        content = content or ""
        elapsed = time.time() - start_time
        self.cost_tracker.record(input_tokens, output_tokens, model_name)

        if self.run_logger:
            # The Anthropic search path returns no cache or reasoning telemetry
            # (3-tuple, no _sync_wire), so the log row carries the zero defaults.
            # Reading self._last_* here would raise on a fresh client and would
            # otherwise attribute the PREVIOUS call's wire stats to this row.
            self.run_logger.log(agent or "Literature Search",
                                f"{provider_name}:{model_name}", messages,
                                content, input_tokens, output_tokens, elapsed)
        return content
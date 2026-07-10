# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# GPT-5.6 explicit prompt caching via OpenRouter chat completions, added after a
# live probe on openai/gpt-5.6-terra verified that OpenRouter forwards
# prompt_cache_breakpoint markers on this path (their docs say Responses-only;
# the endpoint says otherwise). Covers: the cost math pinned to the probe's
# exact live numbers, breakpoint emission in build_cached_messages (first/last
# stable block, byte-stable leading separators, prompt fidelity with the
# flattened form), _flatten_messages passing marked part lists through
# untouched, the per-run session_id in the OpenRouter extras, and the wire
# capture flowing explicitly into meta, CostTracker, RunLogger, and telemetry.

import llm
from llm import (
    LLMClient, CostTracker, RunLogger, build_cached_messages, compute_cost,
    _flatten_messages, _ollama_extras, _openrouter_extras,
    OpenAICompatProvider, build_run_telemetry, RunStats,
)

TERRA = "openrouter:openai/gpt-5.6-terra"

# ── 1) Cost math pinned to the live probe (terra, 2026-07-10, six decimals) ──
# Call 1 (cold write), call 2 (prefix hit + frontier write), call 3 (full hit).
# OpenAI-compat semantics: prompt_tokens INCLUDES cached and written portions;
# reads bill at cached_input (0.25/M = the 90% discount), writes at cached_write
# (3.125/M = 1.25x input).
assert abs(compute_cost(TERRA, 2789, 6,  cached_tokens=0,    cache_write_tokens=2786) - 0.00880375) < 1e-9
assert abs(compute_cost(TERRA, 2790, 29, cached_tokens=2775, cache_write_tokens=12)   - 0.00117375) < 1e-9
assert abs(compute_cost(TERRA, 2790, 23, cached_tokens=2787, cache_write_tokens=0)    - 0.00104925) < 1e-9
# A model without a cached_write entry falls back to 1.25x input (writes carry a
# premium, so never-under-report points UP; mirrors the Anthropic 1.25x already
# charged in this function). grok-4.5: input 2.0 -> writes at 2.5/M.
assert abs(compute_cost("openrouter:x-ai/grok-4.5", 1000, 0, cache_write_tokens=1000) - 0.0025) < 1e-12
# Backward compatibility: calls without the new kwarg are unchanged.
assert abs(compute_cost(TERRA, 1000, 100, cached_tokens=400)
           - (600 * 2.50e-6 + 400 * 0.25e-6 + 100 * 15e-6)) < 1e-12
print("compute_cost: probe-pinned terra math, 1.25x fallback, back-compat: OK")


# ── 2) build_cached_messages: the gpt-5.6 branch ──
blocks = ["HEAD", "STEP-1", "STEP-2"]
msgs = build_cached_messages(TERRA, "SYS", blocks, "TAIL")
assert msgs[0] == {"role": "system", "content": "SYS"}, "system stays a plain string"
parts = msgs[1]["content"]
assert isinstance(parts, list) and len(parts) == 4, "3 stable parts + volatile tail"
marked = [i for i, p in enumerate(parts) if "prompt_cache_breakpoint" in p]
assert marked == [0, 2], f"breakpoints on FIRST and LAST stable block only, got {marked}"
assert all(parts[i]["prompt_cache_breakpoint"] == {"mode": "explicit"} for i in marked)
# Prompt fidelity: the concatenated part texts are byte-identical to the
# flattened form every other model receives, via LEADING separators (a later
# append can never rewrite the bytes of an already-cached part).
joined = "".join(p["text"] for p in parts)
flat = build_cached_messages("openrouter:x-ai/grok-4.5", "SYS", blocks, "TAIL")
assert joined == flat[1]["content"] == "HEAD\n\nSTEP-1\n\nSTEP-2\n\nTAIL"
assert parts[0]["text"] == "HEAD" and parts[1]["text"] == "\n\nSTEP-1", "separators lead, never trail"
# One stable block: it is both first and last, so exactly one marker.
one = build_cached_messages(TERRA, "SYS", "ONLY", "TAIL")[1]["content"]
assert [i for i, p in enumerate(one) if "prompt_cache_breakpoint" in p] == [0]
# Empty stable (the Synthesizer shape): falls through to the plain form.
empty = build_cached_messages(TERRA, "SYS", "", "TAIL")
assert empty == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TAIL"}]
# Other routes are untouched: direct openai (Responses path) flattens, and the
# Anthropic branch keeps its cache_control structure.
direct = build_cached_messages("openai:gpt-5.6-terra", "SYS", blocks, "TAIL")
assert direct[1]["content"] == "HEAD\n\nSTEP-1\n\nSTEP-2\n\nTAIL"
anth = build_cached_messages("anthropic:claude-opus-4-8", "SYS", blocks, "TAIL")
assert anth[1]["content"][0].get("cache_control") == {"type": "ephemeral"}
assert "prompt_cache_breakpoint" not in anth[1]["content"][0]
print("build_cached_messages: markers, separators, fidelity, scoping: OK")


# ── 3) _flatten_messages: marked part lists pass through untouched ──
flat_out = _flatten_messages(msgs)
assert flat_out[1] is msgs[1], "explicit-cache parts must pass through untouched"
plain = _flatten_messages([{"role": "user", "content": [
    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "b"}]}])
assert plain[0]["content"] == "ab", "unmarked block lists still flatten exactly as before"
print("_flatten_messages: breakpoint pass-through + unchanged flattening: OK")


# ── 4) Extras: per-run session_id on OpenRouter, absent on Ollama ──
body, headers = _openrouter_extras("openai/gpt-5.6-terra", "medium")
assert body["session_id"] == llm._RUN_ID, "session_id must carry the shared run id"
assert body["prompt_cache_key"] == "delve-" + llm._RUN_ID, "cache key shares the same id"
assert body["usage"] == {"include": True} and body["reasoning"] == {"effort": "medium"}
body2, _ = _openrouter_extras("moonshotai/kimi-k2.6", "none")
assert body2["session_id"] == llm._RUN_ID, "session_id applies to every OpenRouter model"
ob, oh = _ollama_extras("qwen3.5", "medium")
assert "session_id" not in ob and oh == {}, "Ollama gets no session_id"
print("extras: session_id wiring: OK")


# ── 5) Wire capture through the REAL provider bodies, no SDK needed ──
class _PTD:
    def __init__(self, cached, write):
        self.cached_tokens = cached
        self.cache_write_tokens = write


class _Usage:
    def __init__(self, cached, write):
        self.prompt_tokens = 2790
        self.completion_tokens = 29
        self.prompt_tokens_details = _PTD(cached, write)


class _Msg:
    content = "ok"
    reasoning = None
    reasoning_content = None


class _Choice:
    message = _Msg()


class _Resp:
    def __init__(self, cached, write):
        self.choices = [_Choice()]
        self.usage = _Usage(cached, write)


class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.reasoning = None
        self.reasoning_content = None


class _SChoice:
    def __init__(self, d):
        self.delta = d


class _Chunk:
    def __init__(self, d=None, usage=None):
        self.choices = [_SChoice(d)] if d else []
        self.usage = usage


class _FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        if kw.get("stream"):
            return iter([_Chunk(_Delta("ok")), _Chunk(usage=_Usage(2775, 12))])
        return _Resp(2775, 12)


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeSDK:
    def __init__(self):
        self.chat = _FakeChat()


prov = OpenAICompatProvider.__new__(OpenAICompatProvider)  # skip __init__ (needs openai SDK)
prov.client = _FakeSDK()
prov._extras = _openrouter_extras

out = prov.call(msgs, "openai/gpt-5.6-terra", 64000, 0, reasoning_effort="medium")
assert out == ("ok", 2790, 29, 0, 0), "5-tuple intact"
assert prov._last_cached == 2775 and prov._last_cache_write == 12, "both wire details captured"
req = prov.client.chat.completions.requests[-1]
assert req["messages"][1] is msgs[1], "parts reached the wire unflattened"
assert req["extra_body"]["session_id"] == llm._RUN_ID

toks = []
out2 = prov.stream(msgs, "openai/gpt-5.6-terra", 64000, 0, toks.append, reasoning_effort="medium")
assert out2 == ("ok", 2790, 29, 0, 0) and toks == ["ok"]
assert prov._last_cached == 2775 and prov._last_cache_write == 12, "stream capture on the usage chunk"
print("provider bodies: capture, 5-tuple, unflattened wire, session_id: OK")


# ── 6) Explicit flow into meta, CostTracker, RunLogger, telemetry ──
llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
llm.PROVIDER_CLASSES["openrouter"] = lambda: prov
rl = RunLogger("/tmp/_gpt56_runlog.json")
client = LLMClient(cost_tracker=CostTracker(), run_logger=rl)
text, meta = client.call(msgs, TERRA, agent="Investigator", return_meta=True,
                         reasoning_effort="medium")
assert text == "ok"
assert meta["cached_tokens"] == 2775 and meta["cache_write_tokens"] == 12
assert client.cost_tracker.cache_write_tokens == 12 and client.cost_tracker.cached_tokens == 2775
row = rl.entries[-1]
assert row["cached_tokens"] == 2775 and row["cache_write_tokens"] == 12
assert abs(row["cost_usd"] - round(compute_cost(TERRA, 2790, 29,
                                                cached_tokens=2775,
                                                cache_write_tokens=12), 6)) < 1e-9, \
    "the logged row bills the probe-validated math"
tele = build_run_telemetry(rl, client.cost_tracker, RunStats(), [])
inv = tele["per_agent"]["Investigator"]
assert inv["cached"] == 2775 and inv["cached_write"] == 12, "rollup pairs reads with writes"
print("client flow: meta, tracker, logger row cost, telemetry rollup: OK")

print("test_gpt56_cache: OK")

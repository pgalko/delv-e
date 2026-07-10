# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Reasoning-effort handling: the shared 64k output budget, the Anthropic non-streaming
# clamp, the per-provider effort mapping, and call_with_ladder's truncation retry.
#
# The budget is sized so a heavy reasoning model (e.g. GLM-5.2 at its max effort) can
# finish its thinking AND emit a decision in one turn. The chosen effort is set by the
# --reasoning-effort flag and translated per provider (Ollama takes values as-is;
# OpenRouter's top rung is 'xhigh'; direct Anthropic/OpenAI have no effort dial). When
# a turn still comes back empty/capped, the retry does NOT walk intermediate rungs
# (unreliable across models, since some collapse them to their max): it holds the
# chosen effort, then drops straight to 'none', the one dependable off switch.
#
# This test asserts the MECHANISM against the module constants (so retuning a cap keeps
# it green), and covers what test_executor_reasoning_effort.py does not: the constants,
# the per-provider mapping, the clamp (and the openrouter:anthropic non-clamp), and
# call_with_ladder's hold-then-none / single-call / stub fallback. No network and no
# SDKs: a recording fake provider sees the clamp and the mapping, and lightweight fake
# clients drive call_with_ladder.

import llm
from llm import (
    LLMClient, call_with_ladder, default_reasoning_effort, _provider_effort,
    DEFAULT_MAX_TOKENS, ANTHROPIC_MAX_TOKENS,
)

# ── 1) The shared budget and the Anthropic ceiling are the documented constants ──
assert DEFAULT_MAX_TOKENS == 64000, f"shared agent budget should be 64k, got {DEFAULT_MAX_TOKENS}"
assert ANTHROPIC_MAX_TOKENS == 20000, f"Anthropic non-streaming ceiling should be 20k, got {ANTHROPIC_MAX_TOKENS}"
assert ANTHROPIC_MAX_TOKENS < DEFAULT_MAX_TOKENS, "the clamp must actually lower the shared budget"

# ── 2) Default effort per role + the per-provider effort mapping ──
# default_reasoning_effort: the Executor runs with thinking off; every other role
# starts at the standard default (the flag overrides this for Investigator/Synthesizer).
assert default_reasoning_effort("Executor") == "none", "Executor should start with thinking off"
for role in ("Investigator", "Synthesizer", "ClaimExtractor", "Reconciler", None):
    assert default_reasoning_effort(role) == "medium", f"{role} should start at the medium default"

# _provider_effort translates a canonical effort to what each provider expects, or None
# to send no effort field. Ollama passes everything through; OpenRouter renames the top
# rung (max -> xhigh) and passes the rest through; providers without a dial get None.
for eff in ("max", "high", "medium", "low", "none"):
    assert _provider_effort("ollama", eff) == eff, f"ollama should pass '{eff}' through"
assert _provider_effort("openrouter", "max") == "xhigh", "OpenRouter's top rung is 'xhigh', not 'max'"
for eff in ("high", "medium", "low", "none"):
    assert _provider_effort("openrouter", eff) == eff, f"openrouter should pass '{eff}' through"
for prov in ("anthropic", "openai", "bogus"):
    assert _provider_effort(prov, "high") is None, f"{prov} has no effort dial; should send no field"
print("default effort + per-provider mapping: OK")


# ── 3) The clamp and the mapping, end-to-end through LLMClient ──
class RecordingProvider:
    """Records the max_tokens and reasoning_effort each call/stream actually receives
    from LLMClient ('ABSENT' when no reasoning_effort field is sent)."""
    def __init__(self):
        self.calls = []
        self.stream_calls = []

    def call(self, messages, model, max_tokens, temperature, **kwargs):
        self.calls.append({"max_tokens": max_tokens,
                           "reasoning_effort": kwargs.get("reasoning_effort", "ABSENT")})
        return ("ok", 1, 1, 0, 0)

    def stream(self, messages, model, max_tokens, temperature, on_token, **kwargs):
        self.stream_calls.append({"max_tokens": max_tokens,
                                  "reasoning_effort": kwargs.get("reasoning_effort", "ABSENT")})
        return ("ok", 1, 1, 0, 0)


# Route the three providers we exercise through the recorder (no network, no SDKs).
llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
for name in ("anthropic", "ollama", "openrouter"):
    llm.PROVIDER_CLASSES[name] = RecordingProvider

client = LLMClient()
msg = [{"role": "user", "content": "x"}]

# Direct Anthropic at the shared budget is clamped to the safe non-streaming cap, and
# gets no effort field (it has no effort dial).
client.call(msg, "anthropic:claude-opus-4-8", max_tokens=DEFAULT_MAX_TOKENS, agent="Investigator")
assert client._providers["anthropic"].calls[-1]["max_tokens"] == ANTHROPIC_MAX_TOKENS, \
    "direct Anthropic must be clamped to the non-streaming ceiling"
assert client._providers["anthropic"].calls[-1]["reasoning_effort"] == "ABSENT", \
    "direct Anthropic must receive no reasoning_effort field"

# Ollama takes the full shared budget unchanged and the chosen effort as-is.
client.call(msg, "ollama:qwen3.5", max_tokens=DEFAULT_MAX_TOKENS, agent="Investigator",
            reasoning_effort="max")
assert client._providers["ollama"].calls[-1]["max_tokens"] == DEFAULT_MAX_TOKENS, \
    "Ollama must receive the full shared budget"
assert client._providers["ollama"].calls[-1]["reasoning_effort"] == "max", \
    "Ollama must receive the chosen effort unchanged"

# glm on Ollama: ANY explicit effort value yields an empty completion on the
# endpoint, so the client must OMIT the field entirely for that combo.
client.call(msg, "ollama:glm-5.2:cloud", max_tokens=DEFAULT_MAX_TOKENS,
            agent="Investigator", reasoning_effort="max")
assert client._providers["ollama"].calls[-1].get("reasoning_effort") is None, \
    "glm on Ollama must carry reasoning_effort=None (omitted at the wire)"
print("glm-on-ollama omission: OK")

# OpenRouter takes the full budget; the chosen 'max' is translated to its top rung 'xhigh'.
client.call(msg, "openrouter:z-ai/glm-5.2", max_tokens=DEFAULT_MAX_TOKENS, agent="Investigator",
            reasoning_effort="max")
assert client._providers["openrouter"].calls[-1]["max_tokens"] == DEFAULT_MAX_TOKENS, \
    "OpenRouter must receive the full shared budget"
assert client._providers["openrouter"].calls[-1]["reasoning_effort"] == "xhigh", \
    "OpenRouter must receive 'max' translated to 'xhigh'"

# An openrouter:anthropic/... model uses the OpenAI-compatible path and is explicitly
# NOT subject to the Anthropic clamp.
client.call(msg, "openrouter:anthropic/claude-opus-4", max_tokens=DEFAULT_MAX_TOKENS, agent="Investigator")
assert client._providers["openrouter"].calls[-1]["max_tokens"] == DEFAULT_MAX_TOKENS, \
    "openrouter:anthropic/... must NOT be clamped (it is not the direct Anthropic path)"

# The clamp lives on the non-streaming path; the stream path is the documented escape
# hatch for a genuinely larger Anthropic output, so it is not clamped.
client.stream(msg, "anthropic:claude-opus-4-8", max_tokens=DEFAULT_MAX_TOKENS, agent="Investigator")
assert client._providers["anthropic"].stream_calls[-1]["max_tokens"] == DEFAULT_MAX_TOKENS, \
    "the Anthropic stream path is the large-output escape hatch and must not clamp"
print("clamp + mapping end-to-end (direct-anthropic clamp/no-effort; ollama/openrouter pass + xhigh; stream): OK")


# ── 4) call_with_ladder holds the chosen effort, then falls back to 'none' once ──
class LadderClient:
    """A fake client whose call() truncates at every effort except succeed_at, then
    returns good text. Records the effort of every call. Implements the full call()
    interface (return_meta + reasoning_effort)."""
    def __init__(self, succeed_at):
        self.succeed_at = succeed_at
        self.efforts = []

    def call(self, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=0,
             agent=None, reasoning_effort=None, return_meta=False):
        self.efforts.append(reasoning_effort)
        if reasoning_effort == self.succeed_at:
            text, meta = "the answer", {"truncated": False}
        else:
            text, meta = "", {"truncated": True}  # empty/capped: thought past its budget
        return (text, meta) if return_meta else text


# The chosen effort truncates; the retry holds it once, then drops straight to 'none'.
lc = LadderClient(succeed_at="none")
text, meta = call_with_ladder(lc, msg, "ollama:glm-5.2", agent="Investigator", reasoning_effort="max")
assert text == "the answer", "should return the text from the attempt that succeeded"
assert lc.efforts == ["max", "none"], f"should hold 'max' then drop to 'none', got {lc.efforts}"
assert meta.get("effort_used") == "none", f"effort_used should report the winning attempt, got {meta.get('effort_used')}"

# When the chosen effort succeeds on the first try, there is no fallback call.
lc2 = LadderClient(succeed_at="high")
text2, meta2 = call_with_ladder(lc2, msg, "ollama:glm-5.2", agent="Investigator", reasoning_effort="high")
assert text2 == "the answer" and lc2.efforts == ["high"], \
    f"no fallback when the first try succeeds, got {lc2.efforts}"
assert meta2.get("effort_used") == "high"

# A chosen effort that never succeeds still makes exactly two calls: chosen, then none.
lc3 = LadderClient(succeed_at="never")
text3, meta3 = call_with_ladder(lc3, msg, "openrouter:z-ai/glm-5.2", agent="Synthesizer", reasoning_effort="medium")
assert lc3.efforts == ["medium", "none"], f"should be exactly [chosen, none], got {lc3.efforts}"
assert text3 == "" and meta3.get("effort_used") == "none"

# The Executor starts at the floor ('none'), so there is nothing to fall back to: it
# makes exactly one call and never retries here (its mechanical retry lives elsewhere).
lc4 = LadderClient(succeed_at="never")
text4, meta4 = call_with_ladder(lc4, msg, "ollama:kimi-k2.7-code", agent="Executor")
assert lc4.efforts == ["none"], f"Executor should make a single 'none' call, got {lc4.efforts}"
assert text4 == "" and meta4.get("effort_used") == "none"
print("call_with_ladder hold-then-none (fallback, no-fallback, executor floor): OK")


# ── 5) Providers without a reasoning dial make a single call ──
# A direct Anthropic/OpenAI model has no effort dial, so call_with_ladder must issue
# exactly one call with reasoning_effort=None and not retry, even on a capped return.
class NoDialClient:
    def __init__(self):
        self.calls = []

    def call(self, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=0,
             agent=None, reasoning_effort=None, return_meta=False):
        self.calls.append(reasoning_effort)
        text, meta = "", {"truncated": True}  # capped, but there is no dial to fall back on
        return (text, meta) if return_meta else text


nd = NoDialClient()
text5, meta5 = call_with_ladder(nd, msg, "anthropic:claude-opus-4-8", agent="Synthesizer", reasoning_effort="high")
assert nd.calls == [None], f"a non-ladder provider gets one call with no effort, got {nd.calls}"
assert meta5.get("effort_used") is None, "no effort dial means effort_used is None"
print("non-ladder provider single call: OK")


# ── 6) A lightweight stub without the new kwargs falls back to one plain call ──
# call_with_ladder is a free function precisely so the suite's minimal client stubs,
# which implement only a plain call(), still work: a stub that rejects the
# meta/effort kwargs gets a single plain call and an empty meta dict.
class StubClient:
    def __init__(self):
        self.n = 0

    def call(self, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=0, agent=None):
        self.n += 1
        return "stub text"  # no meta, no effort kwarg accepted


sc = StubClient()
text6, meta6 = call_with_ladder(sc, msg, "ollama:glm-5.2", agent="Synthesizer", reasoning_effort="medium")
assert sc.n == 1, f"stub should be called exactly once, got {sc.n}"
assert text6 == "stub text" and meta6 == {}, "stub fallback returns (text, {})"
print("kwarg-less stub fallback: OK")

print("test_reasoning_ladder: OK")

# ---------- empty-completion guard: one loud retry, reasoning channel counted ------
class _EmptyOnce:
    def __init__(s): s.n=0
    def call(s, messages, model, max_tokens, temperature, **kw):
        s.n += 1
        s._last_reasoning_chars = 495 if s.n == 1 else 0
        return ("" if s.n == 1 else "print(5)"), 10, 5, 0, 0
_orig = client._providers["ollama"]
client._providers["ollama"] = _EmptyOnce()
try:
    out = client.call([{"role": "user", "content": "x"}], "ollama:qwen3.5",
                      agent="Executor")
    assert out == "print(5)" and client._providers["ollama"].n == 2, \
        "empty completion must retry exactly once and return the second answer"
finally:
    client._providers["ollama"] = _orig
print("empty-completion guard: retry-once with reasoning diagnostics: OK")

# ---------- xAI cache-affinity header: stable per run, scoped to x-ai/* -----------
from llm import _xai_conv_headers
h1=_xai_conv_headers("x-ai/grok-4.5"); h2=_xai_conv_headers("x-ai/grok-4.5")
assert h1 and h1 == h2 and len(h1["x-grok-conv-id"]) == 32, "conv id must be stable per run"
assert _xai_conv_headers("z-ai/glm-5.2") == {} and _xai_conv_headers("") == {}
print("xai conv-id header: stable, scoped: OK")


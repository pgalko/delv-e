# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Per-model reasoning-effort floors and the rejection circuit breaker, built
# from a live crash: grok-4.5 turns came back empty, the Investigator's rescue
# ladder correctly escalated to reasoning_effort="none", and xAI 400'd the run
# dead with "Reasoning is mandatory for this endpoint and cannot be disabled".
# Two defenses, both pinned here: the mapping floors 'none' to 'low' for x-ai
# routes (their documented dial is low/medium/high; the unspecified default is
# HIGH, so omission would be the opposite of the rescue), and LLMClient strips
# the effort field and re-dispatches once when a provider rejects the field
# itself, on both the call and stream paths, so an unmet enum can never kill a
# run again.

import llm
from llm import LLMClient, _provider_effort, _reasoning_rejected

XAI_400 = ("Error code: 400 - {'error': {'message': 'Reasoning is mandatory "
           "for this endpoint and cannot be disabled.', 'code': 400}}")

# ── 1) Mapping floors ──
assert _provider_effort("openrouter", "none", "openrouter:x-ai/grok-4.5") == "low", \
    "the ladder's bottom rung must floor to grok's lowest accepted value"
for eff, want in (("max", "xhigh"), ("high", "high"), ("medium", "medium"), ("low", "low")):
    assert _provider_effort("openrouter", eff, "openrouter:x-ai/grok-4.5") == want
# neighbors unchanged
assert _provider_effort("openrouter", "none", "openrouter:z-ai/glm-5.2") == "high"
assert _provider_effort("openrouter", "none", "openrouter:moonshotai/kimi-k2.6") == "none"
assert _provider_effort("ollama", "none", "ollama:glm-5.2:cloud") is None
assert _provider_effort("ollama", "none", "ollama:kimi-k2.7-code:cloud") == "none"
print("mapping floors: x-ai none->low, neighbors intact: OK")

# ── 2) The detector ──
assert _reasoning_rejected(RuntimeError(XAI_400))
assert _reasoning_rejected(ValueError("REASONING is MANDATORY here"))
assert not _reasoning_rejected(ValueError("rate limit exceeded"))
assert not _reasoning_rejected(ValueError("invalid reasoning about cats"))
print("detector: OK")

# ── 3) Circuit breaker on call(): strip the field, re-dispatch once ──
class _RejectingProvider:
    """400s whenever a truthy effort is sent; succeeds when the field is
    stripped. Records every dispatch."""
    def __init__(self):
        self.calls = []

    def call(self, messages, model, max_tokens, temperature, **kw):
        self.calls.append(dict(kw))
        if kw.get("reasoning_effort"):
            raise RuntimeError(XAI_400)
        return ("rescued", 10, 5, 0, 0)

    def stream(self, messages, model, max_tokens, temperature, on_token, **kw):
        self.calls.append(dict(kw))
        if kw.get("reasoning_effort"):
            raise RuntimeError(XAI_400)
        on_token("rescued")
        return ("rescued", 10, 5, 0, 0)


llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
prov = _RejectingProvider()
llm.PROVIDER_CLASSES["openrouter"] = lambda: prov
client = LLMClient()
msg = [{"role": "user", "content": "x"}]

out = client.call(msg, "openrouter:x-ai/grok-4.5", agent="Investigator",
                  reasoning_effort="low")
assert out == "rescued", "the run must survive a rejected effort field"
assert len(prov.calls) == 2, "exactly one field-stripped re-dispatch"
assert prov.calls[0]["reasoning_effort"] == "low"
assert prov.calls[1]["reasoning_effort"] is None, "the retry omits the field"

prov.calls.clear()
out = client.stream(msg, "openrouter:x-ai/grok-4.5", agent="Investigator",
                    reasoning_effort="medium", output_manager=None)
assert out == "rescued" and len(prov.calls) == 2
assert prov.calls[1]["reasoning_effort"] is None
print("circuit breaker: call + stream rescue: OK")

# ── 4) The breaker is narrow: other errors still raise, once ──
class _AlwaysBroken:
    def __init__(self):
        self.n = 0

    def call(self, messages, model, max_tokens, temperature, **kw):
        self.n += 1
        raise ValueError("rate limit exceeded")


broken = _AlwaysBroken()
llm.PROVIDER_CLASSES["openrouter"] = lambda: broken
c2 = LLMClient()
try:
    c2.call(msg, "openrouter:x-ai/grok-4.5", agent="Investigator",
            reasoning_effort="low")
    raise AssertionError("non-reasoning errors must still raise")
except ValueError:
    pass
assert broken.n == 1, "no blind retry for unrelated errors"
print("breaker scope: unrelated errors raise unchanged: OK")

# ── 5) End-to-end regression of the crash: 'none' on grok never reaches the wire ──
class _Recording:
    def __init__(self):
        self.kw = None

    def call(self, messages, model, max_tokens, temperature, **kw):
        self.kw = dict(kw)
        return ("ok", 1, 1, 0, 0)


rec = _Recording()
llm.PROVIDER_CLASSES["openrouter"] = lambda: rec
c3 = LLMClient()
c3.call(msg, "openrouter:x-ai/grok-4.5", agent="Investigator", reasoning_effort="none")
assert rec.kw["reasoning_effort"] == "low", \
    "the rescue rung's 'none' must arrive at the provider as 'low' on grok"
print("live-crash regression: OK")

print("test_reasoning_floors: OK")

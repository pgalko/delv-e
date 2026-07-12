# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Provider-reported cost (OpenRouter's `usage.cost`, already requested on every
# request via usage:{include:true} and previously discarded). Built from a live
# reconciliation: a native xAI search step billed $0.20 while our ledger booked
# $0.1348. Token math was exact (their cache-read discount, $0.104, equals
# 69,504 cached x (2.0 - 0.50)/M to the cent); the gap was xAI's per-call
# server-tool fee, $5 per 1,000 calls, on ~13 agentic web/X searches inside one
# request. PRICING cannot model that, so the invoice becomes the source of
# truth when the provider supplies one. Pins: capture, absent-vs-zero
# semantics, ledger preference with the cache-savings delta preserved, row
# provenance, and byte-identical behavior for providers that report no cost.

import llm
from llm import CostTracker, RunLogger, compute_cost, _usage_cost

MODEL = "openrouter:x-ai/grok-4.5"
BARE = "x-ai/grok-4.5"


class _Usage:
    def __init__(self, cost=None, **kw):
        if cost is not None:
            self.cost = cost
        for k, v in kw.items():
            setattr(self, k, v)

    def model_dump(self):
        return dict(self.__dict__)


# ── 1) _usage_cost: attribute, dict fallback, and absent vs zero ──
assert _usage_cost(_Usage(cost=0.2)) == 0.2
assert _usage_cost(_Usage(cost="0.135")) == 0.135, "string costs coerce"
assert _usage_cost(_Usage(cost=0.0)) == 0.0, "zero is a REAL charge (insurance), not absent"
assert _usage_cost(_Usage()) is None, "no cost field means fall back to our estimate"
assert _usage_cost(None) is None
assert _usage_cost(_Usage(cost="n/a")) is None, "unparseable degrades to fallback"


class _DictOnly:
    """SDK shape that types no `cost` attribute: only the raw payload has it."""
    def model_dump(self):
        return {"prompt_tokens": 10, "cost": 0.5}


assert _usage_cost(_DictOnly()) == 0.5
print("_usage_cost: attribute, dict fallback, absent vs zero: OK")

# ── 2) The live search row: ledger now equals the invoice ──
IN_TOK, OUT_TOK, CACHED = 115_813, 1_246, 69_504
INVOICE = 0.20                      # OpenRouter's final charge
computed = compute_cost(MODEL, IN_TOK, OUT_TOK, cached_tokens=CACHED)
assert abs(computed - 0.1348) < 0.0005, f"our token math (unchanged): {computed}"

t = CostTracker()
t.record(IN_TOK, OUT_TOK, BARE, cached_tokens=CACHED, provider_cost=INVOICE)
assert abs(t.total_cost - INVOICE) < 1e-9, "the invoice is booked verbatim"
# The counterfactual keeps our cache-savings delta but rides on the invoice's level,
# reproducing OpenRouter's own subtotal / discount / final breakdown.
uncached = compute_cost(MODEL, IN_TOK, OUT_TOK)          # all input at full rate
saving = uncached - computed
assert abs(saving - 0.1043) < 0.0005, f"cache saving (matches their -$0.104): {saving}"
assert abs(t.total_cost_uncached - (INVOICE + saving)) < 1e-9
assert abs(t.total_cost_uncached - 0.304) < 0.001, "reconstructs their $0.304 subtotal"
print("live search row: ledger $0.20 = invoice, subtotal $0.304 reconstructed: OK")

# ── 3) Zero completion insurance: a $0.00 charge books as $0.00 ──
t2 = CostTracker()
t2.record(17_293, 405, BARE, provider_cost=0.0)
assert t2.total_cost == 0.0, "an insured empty final must cost nothing (was over-booked)"
print("zero completion insurance: booked at zero: OK")

# ── 4) No provider cost: byte-identical to the old behavior ──
t3, t4 = CostTracker(), CostTracker()
t3.record(20_000, 500, BARE, cached_tokens=5_000)                      # new path, no invoice
t4.total_cost = compute_cost(MODEL, 20_000, 500, cached_tokens=5_000)  # old formula
t4.total_cost_uncached = compute_cost(MODEL, 20_000, 500)
assert abs(t3.total_cost - t4.total_cost) < 1e-12
assert abs(t3.total_cost_uncached - t4.total_cost_uncached) < 1e-12
print("fallback path: unchanged for anthropic/ollama/absent cost: OK")

# ── 5) Row provenance: cost_usd and cost_source ──
import tempfile
with tempfile.TemporaryDirectory() as d:
    rl = RunLogger(os.path.join(d, "run_log.json"))
    rl.log("Literature Search", MODEL, [{"role": "user", "content": "q"}], "findings",
           IN_TOK, OUT_TOK, 22.3, cached_tokens=CACHED, provider_cost=INVOICE)
    rl.log("Investigator", MODEL, [{"role": "user", "content": "q"}], "decision",
           20_000, 500, 5.0, cached_tokens=5_000)
    a, b = rl.entries
    assert a["cost_usd"] == round(INVOICE, 6) and a["cost_source"] == "provider"
    assert b["cost_source"] == "computed"
    assert abs(b["cost_usd"] - compute_cost(MODEL, 20_000, 500, cached_tokens=5_000)) < 1e-6
print("row provenance: cost_usd + cost_source: OK")

# ── 6) End to end: the wire side-channel reaches the ledger ──
class _Prov:
    def call(self, messages, model, max_tokens, temperature, **kw):
        self._last_cached = CACHED
        self._last_cache_write = 0
        self._last_reasoning_chars = 0
        self._last_provider_cost = INVOICE
        return ("findings", IN_TOK, OUT_TOK, 0, 0)


llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
llm.PROVIDER_CLASSES["openrouter"] = lambda: _Prov()
c = llm.LLMClient()
out, meta = c.call([{"role": "user", "content": "x"}], MODEL, agent="Literature Search",
                   return_meta=True)
assert out == "findings" and meta["provider_cost"] == INVOICE
assert abs(c.cost_tracker.total_cost - INVOICE) < 1e-9, \
    "the provider's charge must reach the run ledger, not our estimate"
print("wire to ledger: OK")

print("test_provider_cost: OK")

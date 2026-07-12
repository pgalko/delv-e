# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Provider-native web search (replacing the Anthropic-only search_call
# dependency): dispatch by the search model's provider, the OpenRouter web
# plugin as a strict PER-CALL opt-in (the standing extras never carry it, so
# Executor and Synthesizer calls stay searchless by construction), the Ollama
# hosted REST path with distillation by the run's own model, and the 'auto'
# search seat that enables search exactly when the investigator's provider can
# serve it.

import inspect
import types

import llm
from llm import (LLMClient, default_search_model, _ollama_extras,
                 _openrouter_extras, _format_web_results)

msg = [{"role": "user", "content": "CONTEXT...\nQUERY: altitude pace penalty"}]

# ── 1) The searchless pin: the plugin exists ONLY on explicit opt-in ──
body, _ = _openrouter_extras("x-ai/grok-4.5", "medium")
assert "plugins" not in body, "standing extras must never carry the web plugin"
body_ws, _ = _openrouter_extras("x-ai/grok-4.5", "medium", web_search=True)
assert body_ws["plugins"] == [{"id": "web", "max_results": 5}]
ob, oh = _ollama_extras("qwen3.5", "none", web_search=True)
assert "plugins" not in ob and oh == {}, "ollama search is a REST pre-step, not a flag"
print("searchless pin: plugin is per-call opt-in only: OK")

# ── 2) Dispatch: openrouter searches in one plugin-assisted call ──
class _Rec:
    def __init__(self):
        self.calls = []

    def call(self, messages, model, max_tokens, temperature, **kw):
        self.calls.append({"messages": messages, "model": model, "kw": dict(kw)})
        return ("findings", 100, 20, 0, 0)


llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
rec = _Rec()
llm.PROVIDER_CLASSES["openrouter"] = lambda: rec
client = LLMClient()
out = client.search_call(msg, "openrouter:x-ai/grok-4.5", query="altitude pace penalty")
assert out == "findings"
assert rec.calls[-1]["kw"].get("web_search") is True, "the search call carries the opt-in"
assert rec.calls[-1]["model"] == llm.SEARCH_MODELS["openrouter"], \
    "the search runs on the one search seat, never the run's premium model"
# and a NORMAL call on the same client does not
client.call(msg, "openrouter:x-ai/grok-4.5", agent="Investigator")
assert "web_search" not in rec.calls[-1]["kw"], \
    "non-search calls must not carry the flag at all (wire-identical)"
print("dispatch openrouter: one plugin call, normal calls untouched: OK")

# ── 3) Dispatch: ollama fetches via REST then distills on the run's model ──
fake_results = [{"title": "Alt study", "url": "https://ex.com/a", "content": "C" * 3000},
                {"title": "", "url": "https://ex.com/b", "content": "short"}]
_orig = llm._ollama_web_search
llm._ollama_web_search = lambda query, max_results=5: fake_results
try:
    rec2 = _Rec()
    llm.PROVIDER_CLASSES["ollama"] = lambda: rec2
    c2 = LLMClient()
    out = c2.search_call(msg, "ollama:qwen3.5", query="altitude pace penalty")
    assert out == "findings"
    sent = rec2.calls[-1]["messages"][-1]["content"]
    assert "WEB SEARCH RESULTS" in sent and "[Alt study](https://ex.com/a)" in sent
    assert "C" * 2000 in sent and "C" * 2001 not in sent, "excerpts cap at 2000 chars"
    assert "(untitled)" in sent
    assert "web_search" not in rec2.calls[-1]["kw"], "the distill call is a plain call"
    try:
        c2.search_call(msg, "ollama:qwen3.5")
        raise AssertionError("ollama search without a query must raise")
    except ValueError:
        pass
finally:
    llm._ollama_web_search = _orig
print("dispatch ollama: REST + distill, query required: OK")

# ── 4) Dispatch: anthropic path unchanged; openai rejected ──
class _FakeAnthropic:
    def search_call(self, messages, model, max_tokens, temperature, max_uses=5):
        return ("native findings", 10, 5)


llm.PROVIDER_CLASSES["anthropic"] = _FakeAnthropic
c3 = LLMClient()
assert c3.search_call(msg, "anthropic:claude-opus-4-8", query="x") == "native findings"
try:
    c3.search_call(msg, "openai:gpt-5.6-terra", query="x")
    raise AssertionError("direct openai must be rejected")
except ValueError as e:
    assert "not supported" in str(e)
print("dispatch anthropic/openai: OK")

# ── 5) The REST helper: auth, payload, and the missing-key error ──
class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": fake_results}


posted = {}


def _post(url, headers=None, json=None, timeout=None):
    posted.update({"url": url, "headers": headers, "json": json})
    return _Resp()


_orig_httpx = llm.httpx
llm.httpx = types.SimpleNamespace(post=_post)
os.environ["OLLAMA_API_KEY"] = "test-ollama-key"
try:
    res = llm._ollama_web_search("what is delv-e", max_results=4)
    assert res == fake_results
    assert posted["url"] == "https://ollama.com/api/web_search"
    assert posted["headers"]["Authorization"] == "Bearer test-ollama-key"
    assert posted["json"] == {"query": "what is delv-e", "max_results": 4}
    del os.environ["OLLAMA_API_KEY"]
    try:
        llm._ollama_web_search("q")
        raise AssertionError("missing OLLAMA_API_KEY must raise")
    except EnvironmentError as e:
        assert "OLLAMA_API_KEY" in str(e)
finally:
    llm.httpx = _orig_httpx
    os.environ.pop("OLLAMA_API_KEY", None)
print("REST helper: OK")

# ── 6) Seat policy: search follows the INVESTIGATOR's provider, and an ollama
#      run prefers its FREE search over any paid seat. Walks the decision table.
from llm import resolve_search_seat, SEARCH_MODELS, SEARCH_FALLBACK_PROVIDERS
GROK = f"openrouter:{SEARCH_MODELS['openrouter']}"
HAIKU = f"anthropic:{SEARCH_MODELS['anthropic']}"
assert LLMClient.SEARCH_MODEL_OVERRIDE == SEARCH_MODELS["anthropic"], \
    "the anthropic override and the seat map must not drift apart"
assert SEARCH_FALLBACK_PROVIDERS == ("openrouter", "anthropic"), \
    "fallback order: OpenRouter, then Anthropic; ollama can never be a fallback"


def keys(**present):
    """Set exactly the given credentials, clearing the rest."""
    for k in ("OPEN_ROUTER_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_API_KEY"):
        os.environ.pop(k, None)
    for k, v in present.items():
        os.environ[k] = v


try:
    # Investigator on OpenRouter -> grok-4.3, whatever the model is.
    keys(OPEN_ROUTER_API_KEY="or", ANTHROPIC_API_KEY="an", OLLAMA_API_KEY="ol")
    for m in ("openrouter:x-ai/grok-4.5", "openrouter:openai/gpt-5.6-terra",
              "openrouter:openai/gpt-5.6-luna", "openrouter:z-ai/glm-5.2",
              "openrouter:moonshotai/kimi-k2.6"):
        assert resolve_search_seat(m) == GROK, m

    # Investigator on Anthropic -> Haiku.
    assert resolve_search_seat("anthropic:claude-opus-4-8") == HAIKU

    # Investigator on Ollama, key present -> its own FREE search, even though both
    # paid seats are credentialed and available.
    assert resolve_search_seat("ollama:glm-5.2:cloud") == "ollama:glm-5.2:cloud", \
        "an ollama run must prefer free search over the paid seats"

    # Ollama seat, no ollama key -> OpenRouter.
    keys(OPEN_ROUTER_API_KEY="or", ANTHROPIC_API_KEY="an")
    assert resolve_search_seat("ollama:glm-5.2:cloud") == GROK

    # Ollama seat, no ollama and no OpenRouter key -> Anthropic.
    keys(ANTHROPIC_API_KEY="an")
    assert resolve_search_seat("ollama:glm-5.2:cloud") == HAIKU

    # Nothing credentialed -> search disabled from the start.
    keys()
    assert resolve_search_seat("ollama:glm-5.2:cloud") is None

    # Direct openai has no search route and walks the same fallback chain.
    keys(OPEN_ROUTER_API_KEY="or")
    assert resolve_search_seat("openai:gpt-5.6-terra") == GROK
    keys(ANTHROPIC_API_KEY="an")
    assert resolve_search_seat("openai:gpt-5.6-terra") == HAIKU
    keys()
    assert resolve_search_seat("openai:gpt-5.6-terra") is None

    # default_search_model is the per-model view of the same policy.
    keys(OLLAMA_API_KEY="ol")
    assert default_search_model("ollama:qwen3.5") == "ollama:qwen3.5"
    keys()
    assert default_search_model("ollama:qwen3.5") is None, "ollama search needs the key"
finally:
    keys()
print("seat policy: investigator-led, free-ollama preferred, ordered fallback: OK")

# ── 6b) CLI surface: no --search-model, --no-search present, seating auto ──
import run_core as RC
cli = inspect.getsource(RC)
assert "--search-model" not in cli and '"--no-search"' in cli
assert "resolve_search_seat(investigator_model)" in cli, \
    "the seat is resolved from the investigator alone"
# Ordering regression: the seat is credential-gated, so resolving it before
# load_dotenv() reads .env finds no keys and silently disables search on EVERY
# run. A live run seated on openrouter:x-ai/grok-4.5 hit exactly that.
assert cli.index("load_dotenv()") < cli.index("resolve_search_seat("), \
    "the search seat must be resolved AFTER .env is loaded"
print("CLI surface + dotenv ordering: OK")

# ── 7) Formatting edge: no results ──
assert _format_web_results([]) == "(no results)"

print("test_search_providers: OK")
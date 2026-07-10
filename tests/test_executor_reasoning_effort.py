# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# The Executor is mechanical (it transcribes a closed spec into code), so on Ollama
# it should run with thinking disabled (reasoning_effort="none"). Otherwise a
# reasoning model in that seat can spend its whole output budget on chain-of-thought
# and emit no code, which truncates the step (observed with kimi-k2.7-code). The
# decision is made INSIDE the real LLMClient, keyed on the agent role, and gated to
# Ollama because OpenAI/OpenRouter reject "none". This test drives the real
# LLMClient with a recording fake provider and asserts the wiring:
#   1. the Executor's configured effort reaches an Ollama provider,
#   2. a reasoning agent gets the default effort (unchanged behaviour),
#   3. non-Ollama providers are never sent the field,
#   4. the same holds on the streaming path.
# It checks the MECHANISM against the module constants, not literal strings, so
# retuning EXECUTOR_REASONING_EFFORT / DEFAULT_REASONING_EFFORT keeps it green.

import llm
from llm import LLMClient, EXECUTOR_REASONING_EFFORT, DEFAULT_REASONING_EFFORT


class RecordingProvider:
    """Stands in for a real provider; records the kwargs each call/stream gets."""
    def __init__(self):
        self.calls = []
        self.streams = []

    def call(self, messages, model, max_tokens, temperature, **kwargs):
        self.calls.append(kwargs)
        return ("ok", 1, 1, 0, 0)

    def stream(self, messages, model, max_tokens, temperature, on_token, **kwargs):
        self.streams.append(kwargs)
        return ("ok", 1, 1, 0, 0)


# Route both providers we exercise through the recording fake (no network, no SDKs).
llm.PROVIDER_CLASSES = dict(llm.PROVIDER_CLASSES)
llm.PROVIDER_CLASSES["ollama"] = RecordingProvider
llm.PROVIDER_CLASSES["anthropic"] = RecordingProvider

client = LLMClient()
msg = [{"role": "user", "content": "x"}]

# 1) Executor on Ollama -> the configured Executor effort (thinking off by default).
client.call(msg, "ollama:kimi-k2.7-code", agent="Executor")
ollama_prov = client._providers["ollama"]
assert ollama_prov.calls[-1] == {"reasoning_effort": EXECUTOR_REASONING_EFFORT}, \
    f"Executor on Ollama should send the configured effort, got {ollama_prov.calls[-1]}"

# 2) A reasoning agent on Ollama -> the default effort (behaviour unchanged).
client.call(msg, "ollama:qwen3.5", agent="Investigator")
assert ollama_prov.calls[-1] == {"reasoning_effort": DEFAULT_REASONING_EFFORT}, \
    f"Investigator on Ollama should send the default effort, got {ollama_prov.calls[-1]}"

# 3) Executor on a non-Ollama provider -> field NOT sent (OpenAI/OpenRouter reject "none").
client.call(msg, "anthropic:claude-haiku-4-5-20251001", agent="Executor")
anthropic_prov = client._providers["anthropic"]
assert anthropic_prov.calls[-1] == {}, \
    f"Non-Ollama providers must not receive reasoning_effort, got {anthropic_prov.calls[-1]}"

# 4) Same gating on the streaming path.
client.stream(msg, "ollama:kimi-k2.7-code", agent="Executor")
assert ollama_prov.streams[-1] == {"reasoning_effort": EXECUTOR_REASONING_EFFORT}, \
    f"Executor stream on Ollama should send the configured effort, got {ollama_prov.streams[-1]}"

client.stream(msg, "anthropic:claude-haiku-4-5-20251001", agent="Investigator")
assert anthropic_prov.streams[-1] == {}, \
    f"Non-Ollama stream must not receive reasoning_effort, got {anthropic_prov.streams[-1]}"

# Sanity: the shipped default actually disables thinking for the Executor.
assert EXECUTOR_REASONING_EFFORT == "none", \
    f"shipped Executor effort should disable thinking, got {EXECUTOR_REASONING_EFFORT!r}"

print("test_executor_reasoning_effort: OK")

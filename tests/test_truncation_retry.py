# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# When a reasoning model spends its whole output budget on thinking and emits no
# parseable decision, the turn comes back "incomplete" (empty/truncated). The loop
# must NOT re-send the identical prompt (which just repeats the over-thinking); it
# must retry with a directive that steers the model to emit its decision now. This
# is the fix for the "Investigator turn was cut off; retrying" loop, in place of
# raising the token cap. Here the first Investigator turn returns empty; the retry
# must carry DIRECTIVE_TRUNCATED_RETRY, and the run must still complete.

import os, shutil
from kernel import PersistentKernel
from nav_state import NavState
from llm import RunStats
from investigation import run_investigation
from prompts import DIRECTIVE_TRUNCATED_RETRY

out = _tmpdir("trunc_retry")
if os.path.exists(out):
    shutil.rmtree(out)

_VALID_DECISION = (
    "###THINKING###\nEnough to answer.\n"
    "###STATUS###\nSYNTHESIZE\n"
    "###SPEC###\nnone\n"
    "###LEDGER###\nFRONTIER:\n  approach [tested] steps:-\n"
)


class Mock:
    def __init__(s):
        s.inv_msgs = []   # concatenated message text captured per Investigator call

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator":
            s.inv_msgs.append("\n".join(p.get("content", "") if isinstance(p, dict) else str(p)
                                        for p in m))
            # First turn: simulate a budget-exhausted turn (no visible output ->
            # incomplete). Every later turn: a valid decision so the run terminates.
            return "" if len(s.inv_msgs) == 1 else _VALID_DECISION
        if agent == "Synthesizer":
            return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nresolved.\n"
        return ""


client = Mock()
stats = RunStats()
k = PersistentKernel(df=None)
log, _, nav, briefing = run_investigation(
    seed="q", df=None, client=client, investigator_model="m:p", executor_model="m:c",
    schema_text="(none)", max_steps=5, output_dir=out, kernel=k, nav=NavState(),
    stats=stats, compute=True)
k.cleanup()

print("investigator calls:", len(client.inv_msgs))
print("truncation retries:", stats.get("investigator_truncation_retries", 0))
print("briefing produced:", bool(briefing))

# the truncated turn forced at least one retry
assert stats.get("investigator_truncation_retries", 0) >= 1, "no retry recorded for the truncated turn"
assert len(client.inv_msgs) >= 2, "the loop did not re-call the Investigator after truncation"

# the FIRST turn had no truncation directive; the RETRY carried it (steered, not identical)
distinctive = "used up its entire output budget"
assert distinctive in DIRECTIVE_TRUNCATED_RETRY
assert distinctive not in client.inv_msgs[0], "first turn should not carry the retry directive"
assert distinctive in client.inv_msgs[1], "retry did not carry the decisive directive"

# and the run still completed with a briefing rather than spinning out
assert briefing and "resolved" in briefing

print("TRUNCATION RETRY (steered with a decisive directive, run completes) WORKS")

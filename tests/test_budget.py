# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Budget wrap-up notice: invisible through the body of the run (the total budget
# must never read as a quota to fill), present in the final window with the
# correct remaining count, and entirely absent when the run finishes early.
import pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation, _budget_window
from llm import RunStats

df = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "y", "x", "y"],
                   "v": [5.0, 4.0, 5.2, 4.1]})

def inv(t, s, sp, l):
    return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"

CODE = ("```python\nprint('###RESULTS_START###')\n"
        "print(df.groupby('g')['v'].median().to_string())\n"
        "print('###RESULTS_END###')\n```")
LED = "FRONTIER | f1 | in_progress | steps: 1\nREGIME | g | not_examined | steps: -"
SYNTH = "Reasoning.\n###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nEffect is positive.\n"
NOTICE = "ITERATION BUDGET"


class Mock:
    """Scripted client that records every Investigator user-message text."""
    def __init__(s, inv_seq):
        s.inv_seq = inv_seq
        s.inv_calls = 0
        s.inv_inputs = []

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Investigator":
            s.inv_inputs.append(m[-1]["content"])
            content = s.inv_seq[min(s.inv_calls, len(s.inv_seq) - 1)]
            s.inv_calls += 1
            if return_meta:
                return content, {"output_tokens": 50, "max_tokens": max_tokens,
                                 "truncated": False}
            return content
        if agent == "Executor":
            return CODE
        if agent == "Synthesizer":
            return SYNTH
        return ""


# ---------- window arithmetic ----------
assert _budget_window(14) == 3, _budget_window(14)
assert _budget_window(20) == 4, _budget_window(20)
assert _budget_window(8) == 2, _budget_window(8)
assert _budget_window(2) == 2, _budget_window(2)
print("window arithmetic: OK (14->3, 20->4, 8->2, 2->2)")

# ---------- Case A: notice only inside the final window, correct countdown ----------
outA = _tmpdir("budget_A")
seqA = [inv(f"t{i}", "CONTINUE", "median v by g; print.", LED) for i in range(5)] \
       + [inv("done", "SYNTHESIZE", "(none)", LED)]
mA = Mock(seqA)
kA = PersistentKernel(df=df)
statsA = RunStats()
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mA, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=outA, kernel=kA, nav=NavState(),
    stats=statsA)
kA.cleanup()
assert briefing, "A: no briefing"
assert mA.inv_calls == 6, f"A: expected 6 investigator turns, got {mA.inv_calls}"
for i, text in enumerate(mA.inv_inputs[:4], 1):
    assert NOTICE not in text, f"A: notice leaked into early turn {i}"
assert NOTICE in mA.inv_inputs[4], "A: notice missing on turn 5 (window of 2)"
assert "at most 2 Investigator turn(s) remain" in mA.inv_inputs[4], \
    "A: wrong remaining count on turn 5"
assert NOTICE in mA.inv_inputs[5], "A: notice missing on the final turn"
assert "at most 1 Investigator turn(s) remain" in mA.inv_inputs[5], \
    "A: wrong remaining count on the final turn"
assert statsA.get("budget_wrapup_notices") == 2, \
    f"A: expected 2 notices counted, got {statsA.get('budget_wrapup_notices')}"
print("Case A (final-window countdown): OK: turns 1-4 clean, 5-6 noticed (2,1), "
      "counter 2")

# ---------- Case B: a run that finishes early never sees the notice ----------
outB = _tmpdir("budget_B")
seqB = [inv("t1", "CONTINUE", "median v by g; print.", LED),
        inv("done", "SYNTHESIZE", "(none)", LED)]
mB = Mock(seqB)
kB = PersistentKernel(df=df)
statsB = RunStats()
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mB, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=12, output_dir=outB, kernel=kB, nav=NavState(),
    stats=statsB)
kB.cleanup()
assert briefing, "B: no briefing"
assert mB.inv_calls == 2, f"B: expected 2 investigator turns, got {mB.inv_calls}"
assert all(NOTICE not in t for t in mB.inv_inputs), \
    "B: an early-finishing run must never see the budget notice"
assert statsB.get("budget_wrapup_notices") == 0, \
    f"B: expected 0 notices counted, got {statsB.get('budget_wrapup_notices')}"
print("Case B (early finish, budget invisible): OK: 2 turns, no notice, counter 0")

print("OK: budget wrap-up notice behaves as a ceiling, never a quota")

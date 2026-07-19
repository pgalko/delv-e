# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Integration contracts for the append-only context (audit 5.2 + 5.3):
#   1. decide() assembly: permanent blocks ride the cached stable prefix; the
#      working set (full raws) heads the volatile tail; the head is byte-stable.
#   2. The REHYDRATE pin through the LIVE loop: a listing restores the raw on
#      the next turn and keeps it resident for REHYDRATE_PIN_TURNS.
#   3. Cache survival: replaying a run's growth with protection churn, the
#      prior turn's stable prefix is a byte-exact prefix of the next turn's —
#      the property the old rendering broke (0-35% survival live).
#   4. Registry slimming: newest objects keep descriptions; older objects stay
#      by name; the Executor's filtered view is untouched.

import pandas as pd

import investigation as I
import prompts as P
from investigation import Investigator, run_investigation, REHYDRATE_PIN_TURNS
from kernel import PersistentKernel, REGISTRY_DESC_RECENT
from nav_state import NavState
from llm import RunStats

DF = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "y", "x", "y"],
                   "v": [5.0, 4.0, 5.2, 4.1]})

# ====================================================================
# 1. decide() assembly
# ====================================================================
class Capture:
    def __init__(s): s.captured = []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False, reasoning_effort=None):
        s.captured.append(m)
        r = ("###THINKING###\nt\n###STATUS###\nCONTINUE\n###SPEC###\ns\n"
             "###LEDGER###\nFRONTIER:\n  f1 [untested] steps: -\n")
        return (r, {"output_tokens": 5, "max_tokens": max_tokens,
                    "truncated": False}) if return_meta else r

LOG6 = [{"step": i, "spec": f"spec {i}; print the result.", "stdout": f"value_{i} = {i}0.5",
         "thinking": f"note {i}. next.", "attempts": 1} for i in range(1, 7)]
cc = Capture()
inv = Investigator(cc, "m:p")
nav = NavState()
nav.target_estimand = "pinned"
inv.decide("seed?", "(schema)", "(reg)", LOG6, nav, rehydrate={2})
msgs = cc.captured[0]

def parts(m):
    out = []
    for p in m:
        c = p.get("content") if isinstance(p, dict) else p
        if isinstance(c, list):
            out += [b.get("text", "") for b in c]
        else:
            out.append(c or "")
    return out

texts = parts(msgs)
full = "\n".join(texts)
assert P.WORKING_SET_HEADER in full, "the working set section must be present"
stable, tail = full.split(P.WORKING_SET_HEADER, 1)
assert "--- STEP 6 ---" in tail and "value_6" in tail, "recents full in the tail"
assert "[REHYDRATED step 2" in tail and "value_2 = 20.5" in tail, "pinned raw in the tail"
assert "--- STEP 1 (archived) ---" in stable and "RESULT EXCERPT" in stable, \
    "permanent blocks ride the cached stable prefix"
assert "--- STEP 1 (archived) ---" not in tail
assert "value_1 = 10.5" in stable, "the excerpt keeps a short raw whole, in the prefix"
print("1 (assembly: prefix archived, working set in tail, pins resident): OK")

# ====================================================================
# 2. REHYDRATE pin lifecycle through the live loop
# ====================================================================
LED = "FRONTIER | f1 | in_progress | steps: 1\nREGIME | g | not_examined | steps: -"
def turn(t, s, sp, led=LED, rehy=None):
    r = f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{led}\n"
    if rehy:
        r += f"###REHYDRATE###\n{rehy}\n"
    return r

GOOD_CODE = ("```python\nprint(df.groupby('g')['v'].median().to_string())\n```")
SYNTH_OK = ("###GATES###\nG1: pass\n###VERDICT###\nFINAL\n###FINDINGS###\n"
            "F1 | decisive\nCLAIM: c.\nNUMBERS: 5.1 [step 1]\nCAVEATS: none\n")

class MockLoop:
    def __init__(s):
        # 8 turns: request rehydrate of step 1 on turn 4; synthesize on turn 8.
        s.seq = [turn(f"t{i}", "CONTINUE", f"move {i}; print output.",
                      rehy=("1" if i == 4 else None)) for i in range(1, 8)]
        s.seq.append(turn("done", "SYNTHESIZE", "(none)"))
        s.inv_calls = 0
        s.inv_inputs = []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False, reasoning_effort=None):
        if agent == "Investigator":
            s.inv_inputs.append("\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for p in m for b in (p.get("content") if isinstance(p.get("content"), list)
                                     else [{"text": p.get("content", "")}])))
            r = s.seq[min(s.inv_calls, len(s.seq) - 1)]
            s.inv_calls += 1
            return (r, {"output_tokens": 5, "max_tokens": max_tokens,
                        "truncated": False}) if return_meta else r
        if agent == "Executor":
            return (GOOD_CODE, {}) if return_meta else GOOD_CODE
        if agent == "Synthesizer":
            return (SYNTH_OK, {}) if return_meta else SYNTH_OK
        return ("", {}) if return_meta else ""

mk = MockLoop()
kern = PersistentKernel(df=DF.copy())
outdir = _tmpdir("pins")
log, _, nav, briefing = run_investigation(
    seed="q?", df=DF, client=mk, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=10, output_dir=outdir, kernel=kern,
    nav=NavState(), stats=RunStats())
kern.cleanup()
assert briefing and mk.inv_calls == 8
# Turn indices (0-based): the rehydrate was EMITTED on turn 4 (index 3).
# Step 1's raw must be pinned into the working set on the inputs of turns
# 5..5+PIN-1 (indices 4..3+PIN) and gone after.
resident = ["[REHYDRATED step 1" in t for t in mk.inv_inputs]
assert resident[:4] == [False] * 4, "no pin before the request"
expect_on = list(range(4, 4 + REHYDRATE_PIN_TURNS))
for i in expect_on:
    assert resident[i], f"turn {i}: pinned raw missing (pin should last {REHYDRATE_PIN_TURNS} turns)"
for i in range(4 + REHYDRATE_PIN_TURNS, len(resident)):
    assert not resident[i], f"turn {i}: pin should have expired"
# The raw itself (not just the label) is present while pinned
assert "--- STEP 1 ---" in mk.inv_inputs[4]
print(f"2 (REHYDRATE pin: resident for exactly {REHYDRATE_PIN_TURNS} turns): OK")

# ====================================================================
# 3. Cache survival under churn (the property, end to end)
# ====================================================================
# The stable prefixes captured from the live loop above: each turn's stable
# region must extend the previous one byte-for-byte (the tail is the last
# part; everything before it is the cached prefix).
prev = None
for t in mk.inv_inputs:
    if P.WORKING_SET_HEADER in t:
        stable_part = t[:t.rindex(P.WORKING_SET_HEADER)]
    else:  # first turn(s): no completed steps yet, tail starts at the registry
        stable_part = t[:t.rindex("CURRENT NAMESPACE REGISTRY")]
    if prev is not None:
        assert stable_part.startswith(prev), "a turn's cached prefix must extend the last"
    prev = stable_part
print("3 (live-loop stable prefixes strictly extend, turn over turn): OK")

# ====================================================================
# 4. Registry slimming
# ====================================================================
k = PersistentKernel.__new__(PersistentKernel)
k.registry = {"columns": ["a"],
              "namespace": [{"name": f"obj{i}", "desc": f"DataFrame {i}x2"}
                            for i in range(40)]}
out = k.describe_namespace()
assert f"- obj39: DataFrame 39x2" in out, "newest keep descriptions"
assert f"- obj{40 - REGISTRY_DESC_RECENT}:" in out
assert "- obj5:" not in out and "obj5" in out, "older objects present by name only"
assert "Earlier derived objects, by NAME only" in out
focus = k.describe_namespace(names={"obj5"})
assert "- obj5: DataFrame 5x2" in focus, "the Executor's filtered view keeps full descriptions"
small = PersistentKernel.__new__(PersistentKernel)
small.registry = {"columns": [], "namespace": [{"name": "x", "desc": "d"}]}
assert "Earlier derived objects" not in small.describe_namespace(), \
    "no names-only section when everything fits the described window"
print("4 (registry: newest described, older by name, executor view untouched): OK")

print("\nALL APPEND-ONLY CONTEXT INTEGRATION TESTS PASSED")

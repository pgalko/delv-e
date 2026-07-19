# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Synthesizer GATES block: the legend requires it, the parser extracts it, and
# every malformed or legacy shape degrades gracefully (the 6.2 three-way-contract
# discipline applied to a new model-facing block).
import pandas as pd
from prompts import SYNTHESIZER_SYSTEM
from synthesis import _parse_synth, Synthesizer
from nav_state import NavState

GATES = ("G1: pass - era examined within levels at step 14\n"
         "G1b: n/a - effect does not vary across an examined modifier\n"
         "G2: n/a - no unresolvable confound claimed\n"
         "G3: pass - step 12 estimates the target directly\n"
         "G4: pass - rank uncertainty computed before the ranking claim")

# ---------- legend: the system prompt names the block and the verdict coupling ----------
assert "###GATES###" in SYNTHESIZER_SYSTEM, "legend must name the GATES block"
assert "n/a" in SYNTHESIZER_SYSTEM, "legend must allow n/a for generality"
print("legend: GATES block present with n/a option: OK")

# ---------- shape 1: canonical three-block output ----------
out = f"###GATES###\n{GATES}\n###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nClear effect.\n"
r = _parse_synth(out)
assert r["verdict"] == "FINAL" and r["findings"].startswith("## Summary")
assert r["gates_review"] == GATES, "gates text must be extracted verbatim"
assert "G1:" not in r["findings"], "gates must not leak into the briefing"
print("shape 1 (gates before verdict): OK")

# ---------- shape 2: legacy two-block output (weak model forgets the gates) ----------
r = _parse_synth("###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nok.\n")
assert r["verdict"] == "FINAL" and r["gates_review"] == ""
print("shape 2 (gates absent, tolerated): OK")

# ---------- shape 3: gates AFTER the briefing (misplaced) ----------
out3 = f"###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nok.\n###GATES###\n{GATES}\n"
r = _parse_synth(out3)
assert r["findings"].strip() == "## Summary\nok.", "findings block must stop at the GATES marker"
assert r["gates_review"] == GATES
print("shape 3 (gates misplaced after briefing): OK")

# ---------- shape 4: gated pushback with gates ----------
out4 = (f"###GATES###\nG1: fail - no regime examined within levels\n"
        "###VERDICT###\nNEEDS_MORE_WORK: estimate the effect within era levels\n"
        "###FINDINGS###\nnone\n")
r = _parse_synth(out4)
assert r["verdict"] == "NEEDS_MORE_WORK" and "era" in r["reason"]
assert r["findings"] == "" and "fail" in r["gates_review"]
print("shape 4 (failed gate + pushback): OK")

# ---------- shape 5: the LIVE failure: briefing marker one hash short ----------
# A real glm run emitted "###FINDINGS##" (two trailing hashes); the exact-match
# parser silently dropped a complete ~6,900-char briefing under a FINAL verdict.
out5 = (f"###GATES###\n{GATES}\n###VERDICT###\nFINAL\n"
        "###FINDINGS##\n## Summary\nVerstappen leads, bounded below by the "
        "non-dominant-car margin.\n\n## Method notes\n- Pace covers ~59% of entries.\n")
r = _parse_synth(out5)
assert r["verdict"] == "FINAL", r["verdict"]
assert r["findings"].startswith("## Summary"), r["findings"][:60]
assert "Method notes" in r["findings"], "full findings block must survive the malformed marker"
assert r["gates_review"] == GATES
print("shape 5 (live trailing-hash malformation, briefing recovered): OK")

# ---------- shape 6: zero trailing hashes ----------
out6 = "###VERDICT###\nFINAL\n###FINDINGS\n## Summary\nok.\n"
r = _parse_synth(out6)
assert r["verdict"] == "FINAL" and r["findings"].startswith("## Summary")
print("shape 6 (zero trailing hashes): OK")

# ---------- shape 7: leading-malformed marker, salvage path ----------
out7 = "###VERDICT###\nFINAL\n##FINDINGS###\n## Summary\nstill recovered.\n"
r = _parse_synth(out7)
assert r["findings"].startswith("## Summary"), r["findings"][:60]
print("shape 7 (leading-malformed marker, salvaged): OK")

# ---------- shape 8: malformed marker + 'none' must NOT resurrect a briefing ----------
out8 = "###VERDICT###\nNEEDS_MORE_WORK: stratify by era\n###FINDINGS##\nnone\n"
r = _parse_synth(out8)
assert r["verdict"] == "NEEDS_MORE_WORK" and r["findings"] == ""
print("shape 8 (malformed marker + none stays empty): OK")

# ---------- shape 9: the LIVE mtbuller failure: ### subsections inside the briefing ----------
out9 = (f"###GATES###\n{GATES}\n###VERDICT###\nFINAL\n###FINDINGS###\n"
        "## Summary\nThe gate is sharp.\n\n## What the data can answer\n"
        "### Conversion threshold (Steps 2, 3)\nLogistic fit details here.\n"
        "### Regime change (Steps 5, 6)\nTail loss details here.\n"
        "## Method notes\n- clustered.\n")
r = _parse_synth(out9)
assert r["verdict"] == "FINAL"
assert "Conversion threshold" in r["findings"], "### subsections must not truncate the briefing"
assert "Tail loss details" in r["findings"] and "Method notes" in r["findings"]
print("shape 9 (### subsections inside briefing survive): OK")

# ---------- shape 10: named markers still terminate (misplaced gates after briefing) ----------
out10 = ("###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nbody\n"
         "### Detail subsection\nmore body\n###GATES###\nG1: pass - x\n")
r = _parse_synth(out10)
assert "Detail subsection" in r["findings"] and "more body" in r["findings"]
assert "G1: pass" not in r["findings"], "named GATES marker must still terminate the briefing"
assert "G1: pass - x" in r["gates_review"]
print("shape 10 (named markers terminate, subsections do not): OK")

# ---------- loop net: FINAL with truly no briefing re-runs in finalization mode ----------
from kernel import PersistentKernel
from investigation import run_investigation
from llm import RunStats
import tempfile

EMPTY_FINAL = ("###GATES###\nG1: pass - regime examined\n"
               "###VERDICT###\nFINAL\n###FINDINGS###\nnone\n")
GOOD_FINAL = ("###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\n"
              "Recovered on the finalization retry.\n")

class LoopMock:
    def __init__(s):
        s.synth_calls = 0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Investigator":
            text = ("###THINKING###\nt\n###STATUS###\nSYNTHESIZE\n###SPEC###\nnone\n"
                    "###LEDGER###\nFRONTIER | f | tested | steps: 1\n"
                    "REGIME | g | examined | steps: 1\n")
            if s.synth_calls == 0 and not hasattr(s, "_step_done"):
                s._step_done = True
                text = text.replace("SYNTHESIZE", "CONTINUE").replace(
                    "###SPEC###\nnone", "###SPEC###\nmedian v by g; print.")
            if return_meta:
                return text, {"output_tokens": 50, "max_tokens": max_tokens,
                              "truncated": False}
            return text
        if agent == "Executor":
            return ("```python\nprint('###RESULTS_START###')\n"
                    "print(df.groupby('g')['v'].median().to_string())\n"
                    "print('###RESULTS_END###')\n```")
        if agent == "Synthesizer":
            s.synth_calls += 1
            return EMPTY_FINAL if s.synth_calls == 1 else GOOD_FINAL
        return ""

dfl = pd.DataFrame({"g": ["x", "y", "x", "y"], "v": [1.0, 2.0, 1.2, 2.1]})
outdir = tempfile.mkdtemp(prefix="delve_gates_loop_")
mk = LoopMock()
kk = PersistentKernel(df=dfl)
statsk = RunStats()
log2, _, nav2, briefing2 = run_investigation(
    seed="t", df=dfl, client=mk, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=outdir, kernel=kk, nav=NavState(),
    stats=statsk)
kk.cleanup()
assert mk.synth_calls == 2, f"expected a finalization retry, got {mk.synth_calls} synth calls"
assert briefing2 and "Recovered" in briefing2, "the retry's findings block must be the deliverable"
assert statsk.get("synth_briefing_retries") == 1, statsk.as_dict()
term = [e for e in log2 if e.get("terminal")]
assert term and term[-1]["synth_verdict"] == "FINAL"
assert os.path.exists(os.path.join(outdir, "briefing.md")), "briefing.md must be written"
print("loop net (FINAL + empty briefing -> finalization retry, deliverable saved): OK")

# ---------- end-to-end: synthesize() returns gates_review ----------
class Mock:
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        return out  # the canonical three-block output

df = pd.DataFrame({"g": ["x", "y"], "v": [1.0, 2.0]})
nav = NavState()
nav.apply_ledger_block("REGIME | g | examined | steps: 1")
log = [{"step": 1, "spec": "s", "code": "c", "stdout": "###RESULTS_START###\nok\n###RESULTS_END###",
        "error": None, "thinking": "t", "attempts": 1}]
res = Synthesizer(Mock(), "m:p").synthesize("seed", "(schema)", log, nav)
assert res["verdict"] == "FINAL" and res["gates_review"] == GATES
print("end-to-end synthesize carries gates_review: OK")

print("OK: GATES block legend, parser tolerance, and plumbing all agree")

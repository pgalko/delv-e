# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Three fixes, tested together because they share one failure theme (state or
# text that was never committed leaking into the record):
#   1. TRANSACTIONAL ROLLBACK — a failed executor attempt used to leave its
#      partial mutations live (and the retry template claimed the opposite), so
#      retries ran against a state the spec never described and --resume/--extend
#      replays diverged from the live run. Every failed attempt now restores the
#      worker to the last committed step, and the step artifact records the full
#      attempt audit.
#   2. SYNTHESIZER FAIL-CLOSED — a response with no ###VERDICT### block used to
#      be promoted to a clean FINAL with the raw prose standing in as findings
#      (the published technical record). It is now a protocol failure: one
#      format-repair retry, then the honest salvage nets.
#   3. CHART ISOLATION — successful chart code silently entered the kernel's
#      replayable history (polluting kernel_history.json and every later
#      --extend), and the manifest reader looked up a key the writer never
#      wrote, defaulting every FAILED chart to "available".

import json
import pandas as pd

from investigation import (Executor, _render_charts, _write_step_artifact,
                           run_investigation)
from kernel import PersistentKernel
from nav_state import NavState
from llm import RunStats
from synthesis import Synthesizer, _parse_synth, load_chart_manifest
from prompts import EXECUTOR_RETRY_TEMPLATE, SYNTH_FORMAT_REPAIR

DF = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "y", "x", "y"],
                   "v": [5.0, 4.0, 5.2, 4.1]})


# ====================================================================
# 1. Transactional rollback
# ====================================================================

# ---------- 1a: a failed attempt's mutations are undone (real kernel) ----------
k = PersistentKernel(df=DF.copy())
out, err, _ = k.execute("df['b'] = df['a'] * 2\nx1 = 5\nprint('committed')")
assert err is None and "committed" in out
out, err, _ = k.execute(
    "df['c'] = 99\ndf['v'] = 0\ny_fail = 1\nraise ValueError('boom after mutation')")
assert err and "boom after mutation" in err
assert "rolled back" in err, "the error must state the rollback so the retry template is truthful"
out, err, _ = k.execute(
    "print('c' in df.columns, 'y_fail' in globals(), 'b' in df.columns, x1)\n"
    "print(df['v'].median())")
assert err is None, err
flags, median = out.strip().splitlines()
assert flags == "False False True 5", f"post-rollback state wrong: {flags!r}"
assert median == "4.55", f"in-place df mutation must be undone, got median {median!r}"
reg = k.describe_namespace()
assert "x1" in reg and "y_fail" not in reg, "registry must reflect only committed state"
assert len(k.history) == 2, "failed attempt must not enter the replayable history (probe +1)"
k.cleanup()
print("1a (failed attempt rolled back: columns, objects, in-place edits, registry): OK")

# ---------- 1b: the retry template now tells the truth ----------
assert "ROLLED BACK" in EXECUTOR_RETRY_TEMPLATE
assert "was preserved" not in EXECUTOR_RETRY_TEMPLATE
print("1b (retry template describes the rollback): OK")

# ---------- 1c: loop-level — retry runs against clean state; attempt audit written ----------
def inv(t, s, sp, l):
    return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"

LED = "FRONTIER | f1 | in_progress | steps: 1\nREGIME | g | not_examined | steps: -"
FAIL_CODE = ("```python\ndf['v'] = 0\nhalf_done = 1\n"
             "raise ValueError('deliberate failure after mutation')\n```")
GOOD_CODE = ("```python\nprint('###RESULTS_START###')\n"
             "print(df.groupby('g')['v'].median().to_string())\n"
             "print('half_done' in globals())\nprint('###RESULTS_END###')\n```")
SYNTH_OK = ("###GATES###\nG1: pass\n###VERDICT###\nFINAL\n###FINDINGS###\n"
            "F1 | decisive\nCLAIM: x rows have higher median v.\n"
            "NUMBERS: 5.1 vs 4.05 [step 1]\nCAVEATS: none\n")

class MockRollback:
    def __init__(s):
        s.inv_seq = [inv("t1", "CONTINUE", "median v by g; print.", LED),
                     inv("done", "SYNTHESIZE", "(none)", LED)]
        s.inv_calls = 0
        s.exec_calls = 0
        s.exec_inputs = []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Investigator":
            r = s.inv_seq[min(s.inv_calls, len(s.inv_seq) - 1)]
            s.inv_calls += 1
            return (r, {"output_tokens": 50, "max_tokens": max_tokens,
                        "truncated": False}) if return_meta else r
        if agent == "Executor":
            s.exec_calls += 1
            s.exec_inputs.append("\n".join(
                p.get("content", "") for p in m if isinstance(p, dict)))
            return FAIL_CODE if s.exec_calls == 1 else GOOD_CODE
        if agent == "Synthesizer":
            return SYNTH_OK
        return ""

outdir = _tmpdir("rollback_loop")
mk = MockRollback()
kern = PersistentKernel(df=DF.copy())
log, _, nav, briefing = run_investigation(
    seed="is v higher for x?", df=DF, client=mk, investigator_model="m:p",
    executor_model="m:c", schema_text="(s)", max_steps=4, output_dir=outdir,
    kernel=kern, nav=NavState(), stats=RunStats())
kern.cleanup()
assert briefing and mk.exec_calls == 2
step1 = next(e for e in log if e.get("code"))
assert "5.10" in (step1["stdout"] or "") or "5.1" in (step1["stdout"] or ""), \
    "retry must see the ORIGINAL df (in-place wipe rolled back), got: " + repr(step1["stdout"])
assert "False" in step1["stdout"], "objects from the failed attempt must be gone"
assert "ROLLED BACK" in mk.exec_inputs[1], "the retry message must carry the truthful template"
alog = step1.get("attempt_log")
assert alog and len(alog) == 2 and alog[0]["error"] and not alog[1]["error"]
art = open(os.path.join(outdir, "exploration", "01", "analysis.md")).read()
assert "## Execution attempts" in art and "FAILED — rolled back" in art \
    and "deliberate failure" in art, "artifact must carry the attempt audit"
print("1c (loop retry sees pre-failure state; attempt audit in log + artifact): OK")

# ---------- 1d: a clean single-attempt step stays exactly as before ----------
clean = next(e for e in log if e.get("code"))
entry = {"step": 9, "spec": "s", "code": "print(1)", "stdout": "1\n", "error": None,
         "attempts": 1, "thinking": "t",
         "attempt_log": [{"attempt": 1, "code": "print(1)", "stdout": "1\n", "error": None}]}
d = _tmpdir("art_clean")
_write_step_artifact(d, entry, 1, 4)
art = open(os.path.join(d, "analysis.md")).read()
assert "## Execution attempts" not in art, \
    "a clean single attempt must not duplicate code/output in an attempts section"
print("1d (clean single-attempt artifacts unchanged): OK")


# ====================================================================
# 2. Synthesizer fail-closed + format repair
# ====================================================================

# ---------- 2a/2b: parser fails closed on a missing verdict ----------
r = _parse_synth("Long internal reasoning prose with no protocol blocks at all.")
assert r["verdict"] == "MODEL_ERROR", r["verdict"]
assert r["findings"] == "", "raw prose must NEVER be promoted to findings"
assert "reasoning prose" in r["preamble"], "the text is kept for the honest salvage path"
r = _parse_synth("###FINDINGS###\nF1 | decisive\nCLAIM: real work.\nNUMBERS: 5\nCAVEATS: none\n")
assert r["verdict"] == "MODEL_ERROR" and "real work" in r["findings"], \
    "parsed findings survive a missing verdict; only the verdict is the failure"
# regression: verdict present is unchanged
assert _parse_synth("###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nok.\n")["verdict"] == "FINAL"
assert _parse_synth("###VERDICT###\nNEEDS_MORE_WORK: stratify\n###FINDINGS###\nnone\n")["verdict"] == "NEEDS_MORE_WORK"
print("2a/2b (missing verdict -> MODEL_ERROR; prose never becomes findings): OK")

# ---------- 2c: one format-repair retry, then normal flow ----------
GLOG = [{"step": 1, "spec": "median v by g", "code": "df.groupby('g')['v'].median()",
         "stdout": "x 5.1\ny 4.05", "thinking": "", "attempts": 1}]

class MockSynth:
    def __init__(s, seq):
        s.seq, s.calls, s.inputs = seq, 0, []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        s.inputs.append("\n".join(p.get("content", "") for p in m if isinstance(p, dict)))
        r = s.seq[min(s.calls, len(s.seq) - 1)]
        s.calls += 1
        return (r, {}) if return_meta else r

ms = MockSynth(["thinking out loud, forgot the blocks entirely",
                "###GATES###\nG1: pass\n###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nClear effect.\n"])
sy = Synthesizer(ms, "m:p")
res = sy.synthesize("seed?", "(schema)", GLOG, NavState())
assert ms.calls == 2, f"expected exactly one repair retry, got {ms.calls} calls"
assert res["verdict"] == "FINAL" and res["findings"].startswith("## Summary")
assert res.get("format_repaired") is True
distinct = "FORMAT REPAIR"
assert distinct in SYNTH_FORMAT_REPAIR
assert distinct not in ms.inputs[0] and distinct in ms.inputs[1], \
    "the repair notice must appear only on the retry"
print("2c (one repair retry recovers a proper verdict): OK")

# ---------- 2d: persistently malformed, final mode -> honest PROVISIONAL salvage ----------
ms2 = MockSynth(["the model keeps writing prose about medians and never emits blocks"])
sy2 = Synthesizer(ms2, "m:p")
res = sy2.synthesize("seed?", "(schema)", GLOG, NavState(), final=True)
assert ms2.calls == 2
assert res["verdict"] == "FINAL"
assert "PROVISIONAL" in res["findings"], "salvage must carry the honest banner"
assert "keeps writing prose" in res["findings"], "the preamble is salvaged under the banner"
print("2d (persistent protocol failure at the ceiling -> bannered salvage): OK")

# ---------- 2e: persistently malformed, MID-RUN, full loop ----------
class MockLoop2e(MockRollback):
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Synthesizer":
            s.synth_calls = getattr(s, "synth_calls", 0) + 1
            return "prose only, over and over, no protocol blocks anywhere"
        return MockRollback.call(s, m, model, max_tokens, temperature, agent,
                                 return_meta)

out2e = _tmpdir("synth_failclosed")
mk2 = MockLoop2e()
mk2.exec_calls = 1  # skip the failing first executor response; use GOOD_CODE only
kern = PersistentKernel(df=DF.copy())
st = RunStats()
log, _, nav, briefing = run_investigation(
    seed="is v higher for x?", df=DF, client=mk2, investigator_model="m:p",
    executor_model="m:c", schema_text="(s)", max_steps=4, output_dir=out2e,
    kernel=kern, nav=NavState(), stats=st)
kern.cleanup()
assert briefing, "the run must still end with a deliverable"
assert "PROVISIONAL" in briefing, \
    "pre-fix, the raw prose became clean FINAL findings with no banner"
assert "prose only, over and over" in briefing, "salvage keeps the model's text, honestly labeled"
assert getattr(mk2, "synth_calls", 0) == 4, \
    f"mid-run (1+repair) then finalization (1+repair) = 4, got {getattr(mk2, 'synth_calls', 0)}"
assert st.get("synth_format_retries") == 2
term = [e for e in log if e.get("terminal")]
assert term and term[-1].get("synth_verdict") == "FINAL"
print("2e (mid-run protocol failure -> repair, finalization net, bannered briefing): OK")


# ====================================================================
# 3. Chart isolation + manifest key
# ====================================================================

# ---------- 3a: manifest reader uses the written key; fails closed ----------
d3 = _tmpdir("manifest")
os.makedirs(os.path.join(d3, "charts"), exist_ok=True)
rows = [{"chart": "a.png", "finding": "F1", "caption": "c", "produced": True},
        {"chart": "b.png", "finding": "F1", "caption": "c", "produced": False},
        {"chart": "c.png", "finding": "F2", "caption": "c", "ok": True},
        {"chart": "d.png", "finding": "F2", "caption": "c"}]
with open(os.path.join(d3, "charts", "manifest.json"), "w") as f:
    json.dump(rows, f)
charts, names = load_chart_manifest(d3)
assert names == {"a.png", "c.png"}, \
    f"failed charts must be excluded and unknown rows fail closed, got {names}"
print("3a (manifest: 'produced' honored, failed/unknown charts excluded): OK")

# ---------- 3b: chart code never enters kernel history or the namespace ----------
CHART_CODE = ("```python\nimport matplotlib.pyplot as plt\n"
              "plt.figure()\nplt.plot([1, 2, 3], [1, 4, 9])\n"
              "plt.savefig('trend.png')\nchart_var = 123\nprint('drawn')\n```")

class MockChart:
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        return CHART_CODE

kern = PersistentKernel(df=DF.copy())
out, err, _ = kern.execute("x1 = 5\nprint('base')")
assert err is None
pre_hist = list(kern.history)
d3b = _tmpdir("charts_iso")
ex = Executor(MockChart(), "m:c")
produced = _render_charts(ex, kern,
                          [{"name": "trend.png", "finding": "F1",
                            "caption": "cap", "spec": "plot v over a"}], d3b)
assert produced == {"trend.png"}
assert os.path.exists(os.path.join(d3b, "charts", "trend.png"))
assert kern.history == pre_hist, "chart code must NOT enter the replayable history"
assert "chart_var" not in kern.describe_namespace() and "x1" in kern.describe_namespace()
out, err, _ = kern.execute("print('chart_var' in globals(), x1)")
assert err is None and out.strip() == "False 5", \
    f"the live worker must be restored to the committed state, got {out!r}"
manifest = json.load(open(os.path.join(d3b, "charts", "manifest.json")))
assert manifest[0]["produced"] is True
kern.cleanup()
print("3b (chart runs uncommitted; worker, registry, and history stay clean): OK")

# ---------- 3c: a FAILED chart also leaves everything clean ----------
class MockBadChart:
    def __init__(s): s.n = 0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        s.n += 1
        return "```python\nghost = 1\nraise RuntimeError('chart broke')\n```"

kern = PersistentKernel(df=DF.copy())
kern.execute("x1 = 5")
pre_hist = list(kern.history)
d3c = _tmpdir("charts_fail")
produced = _render_charts(Executor(MockBadChart(), "m:c"), kern,
                          [{"name": "bad.png", "finding": "F1",
                            "caption": "cap", "spec": "plot"}], d3c)
assert produced == set()
manifest = json.load(open(os.path.join(d3c, "charts", "manifest.json")))
assert manifest[0]["produced"] is False and "chart broke" in (manifest[0]["error"] or "")
assert kern.history == pre_hist
out, err, _ = kern.execute("print('ghost' in globals())")
assert err is None and out.strip() == "False"
kern.cleanup()
print("3c (failed chart rolled back; manifest records the failure): OK")

# ---------- 3d: end-to-end — persisted kernel history contains no chart code ----------
SYNTH_CHART = ("###GATES###\nG1: pass\n###VERDICT###\nFINAL\n###FINDINGS###\n"
               "F1 | decisive\nCLAIM: x is higher.\nNUMBERS: 5.1 vs 4.05 [step 1]\nCAVEATS: none\n\n"
               "###CHARTS###\nCHART: medians.png\nFINDING: F1\nCAPTION: medians by g\n"
               "SPEC: bar chart of median v by g from df\n")

class MockE2E(MockRollback):
    def __init__(s):
        MockRollback.__init__(s)
        s.exec_calls = 1  # start past the failing response: analysis uses GOOD_CODE
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Executor":
            last = "\n".join(p.get("content", "") for p in m if isinstance(p, dict))
            if ".png" in last:
                return CHART_CODE.replace("trend.png", "medians.png")
            return MockRollback.call(s, m, model, max_tokens, temperature, agent,
                                     return_meta)
        if agent == "Synthesizer":
            return SYNTH_CHART
        return MockRollback.call(s, m, model, max_tokens, temperature, agent,
                                 return_meta)

out3d = _tmpdir("charts_e2e")
kern = PersistentKernel(df=DF.copy())
log, _, nav, briefing = run_investigation(
    seed="is v higher for x?", df=DF, client=MockE2E(), investigator_model="m:p",
    executor_model="m:c", schema_text="(s)", max_steps=4, output_dir=out3d,
    kernel=kern, nav=NavState(), stats=RunStats())
kern.cleanup()
assert briefing
assert os.path.exists(os.path.join(out3d, "charts", "medians.png"))
hist = json.load(open(os.path.join(out3d, "kernel_history.json")))
assert hist and not any("savefig" in c or "chart_var" in c for c in hist), \
    "persisted kernel history (replayed by --extend) must contain no chart code"
print("3d (end-to-end: chart produced, kernel_history.json clean for --extend): OK")

print("\nALL ROLLBACK / FAIL-CLOSED / CHART-ISOLATION ASSERTIONS PASSED")

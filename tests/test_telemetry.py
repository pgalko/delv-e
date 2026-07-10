# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Two halves:
#   1) build_run_telemetry: deterministic reduce over synthetic run-log entries,
#      cost tracker, step log, and a RunStats sink. No disk, no pricing dependency.
#   2) the events sink threaded through the real loop: a truncated Investigator
#      turn must increment investigator_truncation_retries.

from llm import RunStats, build_run_telemetry

# ---------- 1) aggregator ----------
class FakeLogger:
    def __init__(self, entries): self.entries = entries
    def summary(self): return "per-agent breakdown (stub)"
class FakeCost:
    calls = 10
    input_tokens = 8000
    output_tokens = 2000
    cache_creation_tokens = 1000
    cache_read_tokens = 5000
    total_cost = 0.25
    total_cost_uncached = 0.40
    def report(self): return "10 API calls | $0.2500"

entries = [
    {"agent": "Investigator", "input_tokens": 2000, "output_tokens": 500,
     "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 500,
     "cost_usd": 0.10, "elapsed_time_s": 12.0, "ttft_s": 1.5},
    {"agent": "Investigator", "input_tokens": 1000, "output_tokens": 400,
     "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0,
     "cost_usd": 0.06, "elapsed_time_s": 8.0, "ttft_s": 1.1},
    {"agent": "Executor", "input_tokens": 3000, "output_tokens": 800,
     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
     "cost_usd": 0.05, "elapsed_time_s": 20.0},                      # no ttft (plain call)
    {"agent": "Synthesizer", "input_tokens": 2000, "output_tokens": 300,
     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 500,
     "cost_usd": 0.04, "elapsed_time_s": 10.0, "ttft_s": 2.0},
]
step_log = [
    {"step": 1, "attempts": 1, "error": None, "terminal": False},
    {"step": 2, "attempts": 3, "error": None, "terminal": False},     # 2 retries, recovered
    {"step": 3, "attempts": 3, "error": "executor failed", "terminal": False},  # failed
    {"step": 4, "spec": "(synthesize)", "terminal": True},
]
stats = RunStats()
stats.bump("searches"); stats.bump("g1_gate_overrides")
stats.bump("synth_pushbacks", 2); stats.bump("investigator_truncation_retries", 1)
stats.flag("provisional_briefing", True)

t = build_run_telemetry(
    FakeLogger(entries), FakeCost(), stats, step_log,
    seed="derive X", dataset_shape=(22705, 49),
    models={"investigator": "a", "executor": "b", "synthesizer": "c", "search": "d"},
    max_iters=14, wall_clock_s=80.0, target_estimand="Correction C: ...", final_verdict="FINAL")

r = t["run"]
assert r["iterations_used"] == 3 and r["iterations_max"] == 14, r
assert r["final_verdict"] == "FINAL" and r["dataset"] == {"rows": 22705, "cols": 49}
assert r["wall_clock_s"] == 80.0 and r["api_time_s"] == 50.0 and r["code_and_overhead_s"] == 30.0
assert t["cost"] == {"total_usd": 0.25, "without_cache_usd": 0.40,
                     "saved_usd": 0.15, "cache_hit_rate_pct": 35.7}, t["cost"]
assert t["tokens"] == {"input_fresh": 8000, "cache_read": 5000, "cache_write": 1000,
                       "output": 2000, "total": 16000}, t["tokens"]
assert t["calls"]["total"] == 10
assert t["calls"]["avg_ttft_s"] == round((1.5 + 1.1 + 2.0) / 3, 3)        # Executor None excluded
assert t["calls"]["tokens_per_second"] == 40.0                            # 2000 out / 50 s
inv = t["per_agent"]["Investigator"]
assert inv["calls"] == 2 and inv["output"] == 900 and inv["cache_read"] == 5000
assert inv["cache_write"] == 500 and inv["time_s"] == 20.0 and inv["avg_ttft_s"] == 1.3
assert t["per_agent"]["Executor"]["avg_ttft_s"] is None                   # had no ttft
assert t["reliability"] == {"executor_retries": 4, "failed_steps": [3],
                            "investigator_truncation_retries": 1, "token_caps_hit": 1}, t["reliability"]
assert t["gates"] == {"g1_gate_overrides": 1, "synth_pushbacks": 2,
                      "provisional_briefing": True, "searches": 1,
                      "budget_wrapup_notices": 0,
                      "synth_briefing_retries": 0}, t["gates"]
assert t["estimand"] == {"named": True, "text": "Correction C: ..."}
assert t["summary_text"]                                                  # human-readable kept
print("build_run_telemetry rollup (run/cost/tokens/calls/per-agent/reliability/gates/estimand): OK")

# empty-run robustness (no entries, no steps, no stats)
empty = build_run_telemetry(FakeLogger([]), FakeCost(), None, [],
                            seed="", dataset_shape=(0, 0), models={}, max_iters=0,
                            wall_clock_s=0.0, target_estimand="", final_verdict="none")
assert empty["run"]["iterations_used"] == 0 and empty["estimand"]["named"] is False
assert empty["calls"]["avg_ttft_s"] is None and empty["gates"]["searches"] == 0
print("build_run_telemetry on an empty run: no crash, sane defaults: OK")

# ---------- 2) events sink through the real loop ----------
import shutil, pandas as pd
from investigation import run_investigation
from nav_state import NavState
from kernel import PersistentKernel

df = pd.DataFrame({"a": [1, 2, 3, 4],
                   "g": ["x", "y", "x", "y"],
                   "v": [5.0, 4.0, 5.2, 4.1]})
out = _tmpdir("telemetry_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(t, s, sp, l):
    return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n" + b + "\n```"
L_OK = "FRONTIER | f1 | tested | steps: 1\nREGIME | g | examined | steps: 1"
# Investigator turn 1 comes back EMPTY (a truncated turn) -> the loop retries once,
# then a valid CONTINUE; turn 2 SYNTHESIZE.
INV = ["",  # truncated turn -> one truncation retry
       inv("s1", "CONTINUE", "median v by g; print.", L_OK),
       inv("s2", "SYNTHESIZE", "none", L_OK)]
EXEC = {1: code("print('###RESULTS_START###');"
                "print(df.groupby('g')['v'].median().to_string());"
                "print('###RESULTS_END###')")}

class Mock:
    def __init__(s): s.i = 0; s.e = 0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator": r = INV[s.i]; s.i += 1; return r
        if agent == "Executor":
            if "raised an error" not in m[-1]["content"]: s.e += 1
            return EXEC[s.e]
        if agent == "Synthesizer":
            return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nok.\n"
        return ""

stats2 = RunStats()
k = PersistentKernel(df=df)
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=Mock(), investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=out, kernel=k, nav=NavState(), stats=stats2)
k.cleanup()

assert stats2.get("investigator_truncation_retries") == 1, stats2.as_dict()
assert stats2.get("searches") == 0 and stats2.get("g1_gate_overrides") == 0
assert stats2.get("synth_pushbacks") == 0
assert stats2.flags.get("provisional_briefing") in (None, False)   # a clean FINAL, not provisional
assert briefing
print("events sink through the loop (truncated turn -> retry counted): OK")

print("TELEMETRY (aggregator + events sink) WORKS")

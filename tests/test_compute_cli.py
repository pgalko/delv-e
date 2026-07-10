# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# End-to-end through run_core.main() in --compute mode, with a mock LLM client so
# the WHOLE entry path runs offline: arg parsing, kernel build at df=None, the
# investigation loop with REAL kernel execution, and crucially the post-run
# telemetry (build_run_telemetry). test_compute.py drives run_investigation
# directly and so cannot catch a df.shape reference in run_core's own pre/post-run
# code; this test does. It is the regression guard for exactly that class of bug.

import glob, json
import llm
import run_core

# --- canned compute-mode model turns -----------------------------------------
_INV_FIRST = (
    "###ESTIMAND###\n"
    "P(sum of two fair six-sided dice >= 9), by Monte Carlo.\n"
    "###THINKING###\n"
    "Estimate by simulation with a fixed seed.\n"
    "###STATUS###\nCONTINUE\n"
    "###SPEC###\n"
    "Simulate 50000 rolls of two fair d6 with numpy default_rng(seed=0); estimate "
    "P(sum>=9); print the estimate, its Monte Carlo standard error, and the sample size.\n"
    "###LEDGER###\n"
    "FRONTIER:\n  monte-carlo [in_progress] steps:-\n"
    "REGIME:\n  sample-size [not_examined] steps:-\n"
    "RISK:\n  too-few-samples [open] steps:-\n"
)
_INV_SYNTH = (
    "###THINKING###\nEstimate matches the exact 10/36 within MC error.\n"
    "###STATUS###\nSYNTHESIZE\n"
    "###SPEC###\nnone\n"
    "###LEDGER###\n"
    "FRONTIER:\n  monte-carlo [tested] steps:1\n"
    "RISK:\n  too-few-samples [resolved] steps:1\n"
)
_EXEC_CODE = (
    "```python\n"
    "import numpy as np\n"
    "print('###RESULTS_START###')\n"
    "print('df_exists:', 'df' in globals())\n"
    "rng = np.random.default_rng(0)\n"
    "n = 50000\n"
    "rolls = rng.integers(1, 7, size=(n, 2)).sum(axis=1)\n"
    "p = float((rolls >= 9).mean())\n"
    "print(f'p_estimate={p:.4f}')\n"
    "print(f'mc_se={(p*(1-p)/n)**0.5:.4f}')\n"
    "print(f'n={n}')\n"
    "print('###RESULTS_END###')\n"
    "```"
)
_SYNTH = (
    "###GATES###\n"
    "UNCERTAINTY: pass - MC standard error reported [step 1].\n"
    "CONVERGENCE: pass - matches exact 10/36 [step 1].\n"
    "VALIDITY: pass - fair-dice model stated.\n"
    "###VERDICT###\nFINAL\n"
    "###BRIEFING###\n## Summary\nP(two d6 >= 9) is about 0.278, matching 10/36.\n"
)


class _MockClient:
    """Drop-in for LLMClient. run_core builds it as
    LLMClient(cost_tracker=..., run_logger=..., progress=...)."""
    def __init__(self, **kw):
        self._inv = 0

    def call(self, messages, model, agent=None, max_tokens=None,
             return_meta=False, **kw):
        if agent == "Investigator":
            self._inv += 1
            resp = _INV_FIRST if self._inv == 1 else _INV_SYNTH
        elif agent == "Executor":
            resp = _EXEC_CODE
        elif agent == "Synthesizer":
            resp = _SYNTH
        else:
            resp = ""
        return (resp, {"truncated": False}) if return_meta else resp


# --- run main() end to end ----------------------------------------------------
out = _tmpdir("compute_cli")
llm.LLMClient = _MockClient                      # patched before main() imports it
old_argv = sys.argv[:]
sys.argv = ["run_core.py", "--compute", "P(two d6 sum to >= 9)?",
            "--iterations", "5", "--output", out]
crashed = None
try:
    run_core.main()                              # must NOT raise (the df.shape bug did)
except SystemExit as e:
    assert e.code in (0, None), f"main() exited with code {e.code}"
except Exception as e:                           # noqa: BLE001 - we want to surface any crash
    crashed = e
finally:
    sys.argv = old_argv

assert crashed is None, f"run_core.main() crashed in compute mode: {crashed!r}"

# the briefing was produced
assert os.path.exists(os.path.join(out, "briefing.md")), "no briefing.md written"

# the post-run telemetry was written and is compute-safe (no dataset -> 0x0, no crash)
tele_files = glob.glob(os.path.join(out, "logs", "*", "run_telemetry.json"))
assert tele_files, "no run_telemetry.json written"
with open(tele_files[0], encoding="utf-8") as f:
    tele = json.load(f)
print("telemetry dataset:", tele["run"]["dataset"])
print("telemetry verdict:", tele["run"]["final_verdict"])
assert tele["run"]["dataset"] == {"rows": 0, "cols": 0}, "compute telemetry should report 0x0"
assert tele["run"]["final_verdict"] == "FINAL"

print("COMPUTE CLI (run_core.main end-to-end, df=None, telemetry OK) WORKS")

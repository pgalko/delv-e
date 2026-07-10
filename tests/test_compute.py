# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Dataset-free (--compute) mode, end to end with a Mock brain and REAL kernel
# execution at df=None. Verifies four things:
#   1. the loop runs and writes a briefing with no dataset loaded;
#   2. the compute prompt bundle is the one actually used (the false "df is
#      loaded" assertion is gone, replaced by the compute environment text);
#   3. the kernel namespace genuinely has no `df` (the executed Monte Carlo
#      prints whether df exists, and it does not);
#   4. the statistical G1 gate is bypassed: the ledger leaves a REGIME
#      not_examined and the Investigator requests SYNTHESIZE, which in DATA mode
#      would force a pushback, but in compute mode must finalize cleanly.

import os, shutil, re
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation
from llm import RunStats, DEFAULT_MAX_TOKENS
from prompts import COMPUTE_INVESTIGATOR_SYSTEM

out = _tmpdir("compute_out")
if os.path.exists(out):
    shutil.rmtree(out)


def inv(blocks):
    return "\n".join(blocks) + "\n"


def code(body):
    return "```python\n" + body + "\n```"


# Turn 1: pin the estimand, run one Monte Carlo step. The REGIME
# (sample-size-convergence) is left not_examined on purpose.
INV_1 = inv([
    "###ESTIMAND###",
    "P(sum of two fair six-sided dice >= 9), by Monte Carlo, cross-checked against the exact 10/36.",
    "###THINKING###",
    "Estimate by simulation with a fixed seed and compare to the exact value.",
    "###STATUS###",
    "CONTINUE",
    "###SPEC###",
    "Simulate 200000 rolls of two fair d6 with numpy default_rng(seed=0); estimate "
    "P(sum>=9); print the estimate, its Monte Carlo standard error, the exact value, "
    "the sample size, and whether df exists in the namespace.",
    "###LEDGER###",
    "FRONTIER:",
    "  monte-carlo-estimate [in_progress] steps:-",
    "REGIME:",
    "  sample-size-convergence [not_examined] steps:-",
    "RISK:",
    "  too-few-samples [open] steps:-",
])

# Turn 2: synthesize. REGIME is STILL not_examined here, so a data-mode run would
# trip the G1 gate. Compute mode must not.
INV_2 = inv([
    "###THINKING###",
    "The estimate matches the exact value within Monte Carlo error. Enough to answer.",
    "###STATUS###",
    "SYNTHESIZE",
    "###SPEC###",
    "none",
    "###LEDGER###",
    "FRONTIER:",
    "  monte-carlo-estimate [tested] steps:1",
    "REGIME:",
    "  sample-size-convergence [not_examined] steps:-",
    "RISK:",
    "  too-few-samples [resolved] steps:1",
])

INV = [INV_1, INV_2]

# The executor code: pure numpy Monte Carlo, NO stratification idioms (so that in
# data mode g1 would read unsatisfied). It also reports whether `df` exists.
EXEC = {1: code(
    "import numpy as np\n"
    "print('###RESULTS_START###')\n"
    "print('df_exists:', 'df' in globals())\n"
    "rng = np.random.default_rng(0)\n"
    "n = 200000\n"
    "rolls = rng.integers(1, 7, size=(n, 2)).sum(axis=1)\n"
    "p = float((rolls >= 9).mean())\n"
    "se = float((p * (1 - p) / n) ** 0.5)\n"
    "print(f'p_estimate={p:.4f}')\n"
    "print(f'mc_se={se:.4f}')\n"
    "print(f'exact={10/36:.4f}')\n"
    "print(f'n={n}')\n"
    "print('###RESULTS_END###')"
)}

SYNTH = (
    "###GATES###\n"
    "UNCERTAINTY: pass - estimate reported with a Monte Carlo standard error [step 1].\n"
    "CONVERGENCE: pass - matches the exact 10/36 within MC error [step 1].\n"
    "VALIDITY: pass - fair-dice model stated.\n"
    "###VERDICT###\n"
    "FINAL\n"
    "###BRIEFING###\n"
    "## Summary\n"
    "P(sum of two fair d6 >= 9) is about 0.278 (MC se ~0.001), matching the exact 10/36.\n"
)


class Mock:
    def __init__(s):
        s.i = 0
        s.e = 0
        s.exec_system = None
        s.inv_blob = ""
        s.inv_max_tokens = None

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator":
            s.inv_max_tokens = max_tokens
            s.inv_blob += "\n".join(part.get("content", "") if isinstance(part, dict) else str(part)
                                    for part in m)
            r = INV[s.i]
            s.i += 1
            return r
        if agent == "Executor":
            if s.exec_system is None:
                s.exec_system = m[0]["content"]
            if "raised an error" not in m[-1]["content"]:
                s.e += 1
            return EXEC[s.e]
        if agent == "Synthesizer":
            return SYNTH
        return ""


client = Mock()
stats = RunStats()
k = PersistentKernel(df=None)                      # NO dataframe
log, _, nav, briefing = run_investigation(
    seed="P(two d6 sum to 9 or more)?", df=None, client=client,
    investigator_model="m:p", executor_model="m:c", schema_text="(no dataset)",
    max_steps=5, output_dir=out, kernel=k, nav=NavState(), stats=stats,
    compute=True)
k.cleanup()

term = [e for e in log if e.get("terminal")]
step1 = next(e for e in log if e.get("step") == 1 and not e.get("terminal"))
stdout1 = step1.get("stdout") or ""

print("terminal verdict:", term[-1].get("synth_verdict") if term else None)
print("briefing head:", repr(briefing.strip()[:80]))
print("g1_gate_overrides:", stats.get("g1_gate_overrides", 0))
print("synth_pushbacks:", stats.get("synth_pushbacks", 0))
print("step1 stdout:", repr(stdout1.strip()[:120]))

# 1. ran to FINAL with a real briefing and wrote it to disk
assert briefing and "10/36" in briefing
assert term and term[-1]["synth_verdict"] == "FINAL"
assert os.path.exists(os.path.join(out, "briefing.md"))

# 2. the COMPUTE prompt bundle was used (not the data one)
assert client.exec_system is not None
assert "No dataset is loaded" in client.exec_system
assert "is already loaded" not in client.exec_system       # the false data-mode line is gone
assert "no dataset is loaded and `df` does not exist" in client.inv_blob

# 3. the kernel genuinely had no df
assert "df_exists: False" in stdout1

# 4. the statistical G1 gate was bypassed despite an unexamined REGIME + SYNTHESIZE.
#    Confirm G1 is genuinely UNSATISFIED (ledger has no examined regime, and the MC
#    code does no stratification), so a data-mode run WOULD have gated here; the
#    only reason it did not is the compute guard.
assert nav.open_regimes(), "test setup: expected an unexamined REGIME on the ledger"
assert not nav.g1_satisfied(log), "test setup: G1 should be unsatisfied (else the gate is moot)"
assert stats.get("g1_gate_overrides", 0) == 0, "G1 gate fired in compute mode"
assert stats.get("synth_pushbacks", 0) == 0, "synthesizer pushed back in compute mode"

# 5. the Monte Carlo actually ran and lands near the exact answer
m = re.search(r"p_estimate=([0-9.]+)", stdout1)
assert m and abs(float(m.group(1)) - 10 / 36) < 0.01, "Monte Carlo estimate off"

# 6. tuning guards (so neither change silently reverts):
#    - the Investigator output budget is the raised 32k ceiling;
#    - the compute cross-check discipline no longer tells the model to stall on a
#      mismatch, and now warns to keep the check apples-to-apples.
assert DEFAULT_MAX_TOKENS == 64000, "shared agent output budget should be the unified 64k ceiling"
assert client.inv_max_tokens == DEFAULT_MAX_TOKENS, "Investigator not called with the shared budget constant"
assert "chase down before proceeding" not in COMPUTE_INVESTIGATOR_SYSTEM, "old stall phrasing still present"
assert "apples-to-apples" in COMPUTE_INVESTIGATOR_SYSTEM, "softened C3 guidance missing"

print("COMPUTE MODE (dataset-free run -> FINAL, df=None, gates bypassed) WORKS")

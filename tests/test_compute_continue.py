# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# --resume and --extend under --compute, end to end through run_core.main() with a
# mock LLM client. The run's mode is persisted in run_meta.json on the fresh run and
# restored on a continue, so a compute run is resumed/extended WITHOUT re-passing
# --compute. This test proves: run_meta.json is written, a resume restores compute
# mode (df stays absent -> telemetry reports 0x0), an extend appends a new seed and
# also stays in compute mode, and applying --compute to a dataset run is rejected.

import glob, json
import llm
import run_core

# --- canned compute-mode model turns (one work step, then synthesize) -----------
_INV_FIRST = (
    "###ESTIMAND###\nP(sum of two fair six-sided dice >= 9), by Monte Carlo.\n"
    "###THINKING###\nEstimate by simulation with a fixed seed.\n"
    "###STATUS###\nCONTINUE\n"
    "###SPEC###\nSimulate 20000 rolls of two fair d6 with numpy default_rng(seed=0); "
    "estimate P(sum>=9); print the estimate, its Monte Carlo standard error, and n.\n"
    "###LEDGER###\nFRONTIER:\n  monte-carlo [in_progress] steps:-\n"
    "REGIME:\n  sample-size [not_examined] steps:-\n"
    "RISK:\n  too-few-samples [open] steps:-\n"
)
_INV_SYNTH = (
    "###THINKING###\nEstimate matches the exact 10/36 within MC error.\n"
    "###STATUS###\nSYNTHESIZE\n###SPEC###\nnone\n"
    "###LEDGER###\nFRONTIER:\n  monte-carlo [tested] steps:1\n"
    "RISK:\n  too-few-samples [resolved] steps:1\n"
)
_EXEC_CODE = (
    "```python\n"
    "import numpy as np\n"
    "print('###RESULTS_START###')\n"
    "print('df_exists:', 'df' in globals())\n"
    "rng = np.random.default_rng(0)\n"
    "n = 20000\n"
    "rolls = rng.integers(1, 7, size=(n, 2)).sum(axis=1)\n"
    "p = float((rolls >= 9).mean())\n"
    "print(f'p_estimate={p:.4f}'); print(f'mc_se={(p*(1-p)/n)**0.5:.4f}'); print(f'n={n}')\n"
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
    def __init__(self, **kw):
        self._inv = 0

    def call(self, messages, model, agent=None, max_tokens=None, return_meta=False, **kw):
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


llm.LLMClient = _MockClient                          # patched before main() imports it


def _run_main(argv):
    """Run run_core.main() with argv; return None on clean exit or the exception."""
    old = sys.argv[:]
    sys.argv = ["run_core.py"] + argv
    err = None
    try:
        run_core.main()
    except SystemExit as e:
        if e.code not in (0, None):
            err = SystemExit(e.code)
    except Exception as e:                           # noqa: BLE001
        err = e
    finally:
        sys.argv = old
    return err


def _newest_telemetry(out):
    files = glob.glob(os.path.join(out, "logs", "*", "run_telemetry.json"))
    assert files, "no run_telemetry.json written"
    newest = max(files, key=os.path.getmtime)
    with open(newest, encoding="utf-8") as f:
        return json.load(f)


# --- 1) fresh compute run writes run_meta.json and a briefing -------------------
out = _tmpdir("compute_cont")
err = _run_main(["--compute", "P(two d6 sum to >= 9)?", "--iterations", "5", "--output", out])
assert err is None, f"fresh compute run failed: {err!r}"
assert os.path.exists(os.path.join(out, "briefing.md")), "fresh: no briefing.md"
with open(os.path.join(out, "run_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
assert meta == {"compute": True}, f"fresh compute run_meta should be compute:true, got {meta}"
assert _newest_telemetry(out)["run"]["dataset"] == {"rows": 0, "cols": 0}
print("1) fresh compute run: run_meta.json={'compute': true}, briefing + 0x0 telemetry: OK")

# --- 2) resume restores compute mode WITHOUT re-passing --compute ---------------
# No --compute and no dataset positional: the mode and the seed both come from disk.
err = _run_main(["--resume", "--output", out, "--iterations", "5"])
assert err is None, f"compute resume failed: {err!r}"
assert _newest_telemetry(out)["run"]["dataset"] == {"rows": 0, "cols": 0}, \
    "resume did not restore compute mode (telemetry not 0x0 -> df was loaded)"
seeds = json.load(open(os.path.join(out, "seeds.json"), encoding="utf-8"))
assert len(seeds) == 1, f"resume must not add a seed, got {seeds}"
print("2) resume (no --compute): mode restored, df absent, seed unchanged: OK")

# --- 3) extend restores compute mode and appends the new seed -------------------
err = _run_main(["P(three d6 sum >= 12)?", "--extend", "--output", out, "--iterations", "5"])
assert err is None, f"compute extend failed: {err!r}"
assert _newest_telemetry(out)["run"]["dataset"] == {"rows": 0, "cols": 0}, \
    "extend did not restore compute mode"
seeds = json.load(open(os.path.join(out, "seeds.json"), encoding="utf-8"))
assert len(seeds) == 2, f"extend should append a second seed, got {seeds}"
print("3) extend (no --compute): mode restored, second seed appended: OK")

# --- 4) --compute against a dataset run is rejected -----------------------------
data_dir = _tmpdir("data_run")
os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"compute": False}, f)
err = _run_main(["--resume", "--compute", "--output", data_dir, "--iterations", "5"])
assert isinstance(err, SystemExit), f"mismatch (--compute on a dataset run) must be rejected, got {err!r}"
print("4) --compute on a dataset run's --resume: rejected: OK")

# --- 5) the run_meta helpers round-trip both modes and tolerate absence ---------
h = _tmpdir("meta_helpers")
run_core._save_run_meta(h, True)
assert run_core._load_run_meta(h) == {"compute": True}
run_core._save_run_meta(h, False)
assert run_core._load_run_meta(h) == {"compute": False}, "data-mode run_meta should be compute:false"
assert run_core._load_run_meta(_tmpdir("empty")) == {}, "absent run_meta should load as {}"
print("5) run_meta helpers: both modes round-trip, absence -> {}: OK")

print("COMPUTE CONTINUE (resume + extend under --compute, mode persisted) WORKS")

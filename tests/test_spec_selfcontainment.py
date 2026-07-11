# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Spec self-containment, built from a live compute run: the Investigator specced
# "Re-run the identical individual-based model of step 1, changing only sigma",
# which is unresolvable to the Executor (it sees ONLY the spec plus the registry
# objects the spec names, never prior steps), so the Executor rebuilds the model
# blind and the change-only-one-thing contrast is invalid. Three defenses, all
# pinned here: the SELF-CONTAINMENT spec rule in both prompt modes, the
# compute-mode persist-as-a-named-function rule, and a runtime tripwire that
# warns when a spec references a step by number while naming zero registry
# objects (the resolved case, a step mention alongside a named object, stays
# silent by design).

import inspect
import re

import prompts as P
import investigation as I
from investigation import _referenced_names, scan_spec_for_leakage

# ── 1) The rule exists in both modes and is leakage-clean ──
src = inspect.getsource(P)
assert src.count("SELF-CONTAINMENT (what the executor can see)") == 2
for phrase in ("never sees prior steps", "by step number",
               "defines it as a named function"):
    assert src.count(phrase) == 2, f"rule lost in one mode: {phrase!r}"
i = src.index("SELF-CONTAINMENT (what the executor can see)")
clause = src[i:src.index("PRINT BUDGET", i)]
assert scan_spec_for_leakage(clause) == [], "the rule must model closed-spec language"
print("self-containment rule: both modes, leakage-clean: OK")

# ── 2) Compute mode: persist-as-function rule, exactly once (compute only) ──
assert src.count("must be built as a NAMED FUNCTION in the step that first defines it") == 1
assert src.count("keeping every variant mechanically identical") == 1
print("compute persist-as-function rule: OK")

# ── 3) _referenced_names: the resolver the tripwire keys on ──
class _K:
    registry = {"namespace": [{"name": "low_sigma_pop"}, {"name": "run_ibm"}],
                "columns": []}


LIVE_SPEC = ("Re-run the identical individual-based model of step 1, changing only "
             "the mutational standard deviation to sigma = 0.3 and the RNG seed to 43. "
             "Persist the final population state under the name new_pop.")
assert _referenced_names(LIVE_SPEC, _K()) == set(), \
    "the live failing spec names no existing registry objects"
assert _referenced_names("Call run_ibm with sigma=0.3; persist as low_sigma_pop.",
                         _K()) == {"run_ibm", "low_sigma_pop"}
print("_referenced_names: OK")

# ── 4) The tripwire: wired at the executor call site, fires on the right shape ──
loop_src = inspect.getsource(I.run_investigation)
assert "blind_step_references" in loop_src, "tripwire must count into RunStats"
assert "names no registry objects" in loop_src, "tripwire warning present"
pat = re.compile(r"\bstep\s+\d+\b", re.IGNORECASE)
assert pat.search(LIVE_SPEC), "the live failing spec matches the step-reference pattern"
assert pat.search("as in Step 12"), "case-insensitive"
assert not pat.search("across the steps so far"), "plural 'steps' must not match"
# the resolved shape stays silent: step mention + a named object
resolved = "Bar chart using the era results from step 5 stored in run_ibm."
assert pat.search(resolved) and _referenced_names(resolved, _K()) == {"run_ibm"}, \
    "a step mention alongside a named object resolves and must not warn"
print("tripwire shape: fires blind, silent when resolved: OK")

print("test_spec_selfcontainment: OK")

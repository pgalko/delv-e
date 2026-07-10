# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile, json, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# --verify under --compute. The audit of a compute run must itself run in compute
# mode: no dataset, the compute investigation prompts, and the compute variants of
# the three verify prompts (claim extraction, audit seed, reconciliation). The mode
# is read from the prior run's run_meta.json, never re-passed. This test proves the
# compute templates are selected, that a compute verify runs end to end with df=None
# and writes the four-document set, and that --compute against a dataset prior is
# rejected.

import llm
import run_core
import prompts
import verify


def _run_main(argv):
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
    with open(max(files, key=os.path.getmtime), encoding="utf-8") as f:
        return json.load(f)


def _msg_text(messages):
    """Flatten message content to one searchable string. A message's content is
    either a plain string or a list of content blocks (each a dict with 'text')."""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                parts.append(blk.get("text", "") if isinstance(blk, dict) else str(blk))
        else:
            parts.append(str(c))
    return " ".join(parts)


# === Part A: the compute flag routes to the compute templates ===================
# compose_audit_seed does no LLM call, so its output can be checked directly.
data_seed = verify.compose_audit_seed("Q?", ["A is 1."], compute=False)
comp_seed = verify.compose_audit_seed("Q?", ["A is 1."], compute=True)
assert "alternative reasonable definitions" in data_seed, "data battery missing from data seed"
assert "independent implementation" in comp_seed and "converges under refinement" in comp_seed, \
    "compute stress battery missing from compute seed"
assert "edge and boundary cases" in comp_seed
assert "coverage and missingness" not in comp_seed, "data axis leaked into the compute seed"
print("A1) compose_audit_seed routes to the compute stress battery: OK")


class _CaptureMock:
    """Records the prompt each agent receives; returns canned text."""
    def __init__(s):
        s.seen = {}
    def call(s, messages, model, agent=None, max_tokens=None, temperature=0,
             return_meta=False, **kw):
        s.seen[agent] = _msg_text(messages)
        text = {"ClaimExtractor": "1. V is 0.4900 by Monte Carlo (n=20000).",
                "Reconciler": "## Summary\nmerged.\n" + "z" * 250}.get(agent, "")
        meta = {"truncated": False, "output_tokens": 10, "max_tokens": max_tokens}
        return (text, meta) if return_meta else text

cm = _CaptureMock()
verify.extract_claims(cm, "m:x", "## Summary\nP is 0.49.\n", compute=True)
verify.reconcile(cm, "m:x", "Q?", "orig", "audit", compute=True)
assert "computational briefing" in cm.seen["ClaimExtractor"], "extractor used the data template"
assert "What the computation shows" in cm.seen["Reconciler"], "reconciler used the data template"
print("A2) extract_claims and reconcile route to the compute templates: OK")

cm2 = _CaptureMock()
verify.extract_claims(cm2, "m:x", "b", compute=False)
assert "analysis briefing" in cm2.seen["ClaimExtractor"], "data default changed"
print("A3) the data-mode default templates are unchanged: OK")


# === Part B: a compute verify runs end to end through main() ====================
# Stand up a prior compute run on disk: a briefing, run_meta marking it compute, and
# the saved question. The audit reads the mode from run_meta (no --compute passed).
prior = _tmpdir("vc_prior")
ORIG_BRIEFING = ("## Summary\nV is 0.49 by Monte Carlo.\n"
                 "## What the computation shows\nThe estimate is 0.4900 (n=20000).\n"
                 "## Method notes\n- numpy default_rng(0).\n")
with open(os.path.join(prior, "briefing.md"), "w", encoding="utf-8") as f:
    f.write(ORIG_BRIEFING)
with open(os.path.join(prior, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"compute": True}, f)
# The audited run was an EXTEND chain: a root problem, then an extension instruction.
# The audit must receive the root problem (which carries the model), not just the last
# seed. ROOT_Q is checked below in the prompt the auditor actually saw.
ROOT_Q = "Estimate P(two d6 sum >= 9) by Monte Carlo with numpy default_rng(0)."
with open(os.path.join(prior, "seeds.json"), "w", encoding="utf-8") as f:
    json.dump([ROOT_Q, "Now also report the Monte Carlo standard error."], f)

AUDIT_BRIEFING = ("## Summary\nClaim (1) confirmed by independent recomputation.\n"
                  "## What the computation shows\nRe-derived 0.4901 (n=50000), within MC error.\n"
                  "## Method notes\n- independent default_rng(1).\n")


class _VerifyFlowMock:
    """Plays ClaimExtractor, Investigator, Executor, Synthesizer, Reconciler in
    compute mode (one work step, then synthesize, then reconcile)."""
    def __init__(s):
        s.seen = {}
        s._did_step = False
    def call(s, messages, model, agent=None, max_tokens=None, temperature=0,
             return_meta=False, **kw):
        s.seen[agent] = _msg_text(messages)
        if agent == "ClaimExtractor":
            text = "1. V is 0.4900 by Monte Carlo (n=20000)."
        elif agent == "Investigator":
            text = ("###ESTIMAND###\nP(two d6 >= 9) by Monte Carlo.\n"
                    "###THINKING###\nt\n###STATUS###\nSYNTHESIZE\n###SPEC###\nnone\n"
                    "###LEDGER###\nFRONTIER:\n  monte-carlo [tested] steps:1\n"
                    "RISK:\n  too-few-samples [resolved] steps:1\n")
            if not s._did_step:
                s._did_step = True
                text = text.replace("SYNTHESIZE", "CONTINUE").replace(
                    "###SPEC###\nnone",
                    "###SPEC###\nSimulate 50000 draws with default_rng(1); print the "
                    "estimate, its MC standard error, and whether df exists.")
        elif agent == "Executor":
            text = ("```python\nimport numpy as np\nprint('###RESULTS_START###')\n"
                    "print('df_exists:', 'df' in globals())\n"
                    "rng = np.random.default_rng(1)\n"
                    "p = float((rng.integers(1, 7, size=(50000, 2)).sum(axis=1) >= 9).mean())\n"
                    "print(f'p={p:.4f}'); print(f'mc_se={(p*(1-p)/50000)**0.5:.4f}')\n"
                    "print('###RESULTS_END###')\n```")
        elif agent == "Synthesizer":
            text = ("###GATES###\nUNCERTAINTY: pass - MC SE reported.\n"
                    "CONVERGENCE: pass - matches 10/36.\nVALIDITY: pass - fair dice.\n"
                    "###VERDICT###\nFINAL\n###BRIEFING###\n" + AUDIT_BRIEFING)
        elif agent == "Reconciler":
            text = ("## Summary\nReconciled: claim 1 confirmed.\n"
                    "## Verification record\n1. confirmed\n" + "z" * 220)
        else:
            text = ""
        meta = {"truncated": False, "output_tokens": 40, "max_tokens": max_tokens}
        return (text, meta) if return_meta else text


vf = _VerifyFlowMock()
llm.LLMClient = lambda **kw: vf                   # main() builds LLMClient(...)
audit = _tmpdir("vc_audit")
err = _run_main(["--verify", prior, "--output", audit, "--iterations", "5"])
assert err is None, f"compute verify failed: {err!r}"

# the audit ran in compute mode: df=None -> telemetry reports 0x0 (mode auto-detected)
assert _newest_telemetry(audit)["run"]["dataset"] == {"rows": 0, "cols": 0}, \
    "the audit did not run in compute mode (telemetry not 0x0)"
# the four-document set is present, with the original carried across verbatim
for name in ["briefing.md", "briefing_original.md", "briefing_audit.md", "claims.md"]:
    assert os.path.exists(os.path.join(audit, name)), f"missing {name}"
with open(os.path.join(audit, "briefing_original.md"), encoding="utf-8") as f:
    assert f.read() == ORIG_BRIEFING, "original briefing not preserved"
with open(os.path.join(audit, "briefing.md"), encoding="utf-8") as f:
    assert "Reconciled" in f.read(), "briefing.md is not the reconciled document"
# the compute verify templates were the ones actually used
assert "computational briefing" in vf.seen["ClaimExtractor"]
assert "What the computation shows" in vf.seen["Reconciler"]
assert ("converges under refinement" in vf.seen["Investigator"]
        or "independent implementation" in vf.seen["Investigator"]), \
    "the audit seed handed to the Investigator did not carry the compute battery"
assert ROOT_Q in vf.seen["Investigator"], \
    "the audit seed did not carry the root problem of the extended run (only the last seed)"
print("B) compute verify end-to-end: mode auto-detected, df=None, four docs, "
      "compute templates used: OK")


# === Part C: --compute against a dataset prior is rejected ======================
dprior = _tmpdir("vc_dprior")
with open(os.path.join(dprior, "briefing.md"), "w", encoding="utf-8") as f:
    f.write("## Summary\nx\n")
with open(os.path.join(dprior, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"compute": False}, f)
err = _run_main(["--verify", dprior, "--compute", "--output", _tmpdir("vc_o"),
                 "--iterations", "5"])
assert isinstance(err, SystemExit), \
    f"--compute on a dataset prior must be rejected, got {err!r}"
print("C) --compute on a dataset prior's audit: rejected: OK")

print("VERIFY COMPUTE (audit in compute mode, compute templates, mode from run_meta) WORKS")

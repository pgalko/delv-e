"""Tests for the serial verification feature (--verify).

Standalone: python3 tests/test_verify.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import verify
import prompts
from nav_state import NavState

# ---------- claim extraction parsing (malformation shapes, per the standing lesson) ----------

class ExtractMock:
    def __init__(s, text, truncated=False):
        s.text, s.truncated = text, truncated
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if return_meta:
            return s.text, {"output_tokens": max_tokens if s.truncated else 10,
                            "max_tokens": max_tokens, "truncated": s.truncated}
        return s.text

clean = "1. The threshold sits at -2.3 with CI [-2.6, -2.0].\n2. The trend is null (p=0.97).\n"
r = verify.extract_claims(ExtractMock(clean), "m:x", "ignored")
assert r == ["The threshold sits at -2.3 with CI [-2.6, -2.0].",
             "The trend is null (p=0.97)."], r
print("claims: clean numbered list: OK")

messy = ("Here are the decisive claims:\n\n"
         "1) **The gate is at -2.3** with a clustered CI\n"
         "   spanning [-2.6, -2.0].\n"
         "2] Peak values fell from 108.6\nto 78.8 (p=0.012).\n"
         "## stray header\n"
         "3. __Attribution__ rests on p=0.035.\n")
r = verify.extract_claims(ExtractMock(messy), "m:x", "ignored")
assert len(r) == 3, r
assert r[0] == "The gate is at -2.3 with a clustered CI spanning [-2.6, -2.0].", r[0]
assert r[1] == "Peak values fell from 108.6 to 78.8 (p=0.012).", r[1]
assert "Attribution rests on p=0.035." == r[2], r[2]
print("claims: bold markers, mixed numbering, continuation folding, header noise: OK")

r = verify.extract_claims(ExtractMock("No list here, only chatter."), "m:x", "x")
assert r == [], r
print("claims: garbage yields empty list: OK")

many = "\n".join(f"{i}. claim {i}" for i in range(1, 15))
r = verify.extract_claims(ExtractMock(many), "m:x", "x")
assert len(r) == verify.MAX_CLAIMS, len(r)
print("claims: capped at MAX_CLAIMS: OK")

# ---------- audit seed composition ----------

seedtxt = verify.compose_audit_seed("Original question?",
                                    ["A is 1.", "B is 2."])
for fingerprint in ["Original question?", "(1) A is 1.", "(2) B is 2.",
                    "alternative reasonable definitions",
                    "respect how the observations group",
                    "coverage and missingness",
                    "reference representative",
                    "failed to examine"]:
    assert fingerprint in seedtxt, fingerprint
print("audit seed: original question, claims, and all four stress axes present: OK")

fb = verify.compose_audit_seed("Q?", [], original_briefing="X" * 9000)
assert "Their report follows" in fb and fb.count("X") == verify._FALLBACK_EXCERPT_CHARS
print("audit seed: empty-claims fallback embeds a capped excerpt: OK")

# ---------- original question spans the whole audited chain ----------
# A fresh run's single seed is the whole question; an extended run's question is the
# root problem plus its extension instructions, not just the last seed (which carries
# none of the model). Consecutive duplicates collapse.
assert verify.original_question(["only seed"]) == "only seed"
assert verify.original_question(["root problem", "now test 24"]) == "root problem\n\nnow test 24"
assert verify.original_question(["root", "root", "now test 24"]) == "root\n\nnow test 24"
assert verify.original_question([]) == ""
print("original_question: fresh seed, extended chain, consecutive-dup collapse: OK")

assert "The audit can itself be wrong" in prompts.RECONCILIATION_PROMPT
assert "same level of analysis" in prompts.RECONCILIATION_PROMPT
print("reconciliation prompt: pre-committed decisiveness clause pinned: OK")

assert "what the original computed and what the audit computed" in prompts.RECONCILIATION_PROMPT
assert "incomplete without the specification" in prompts.CLAIM_EXTRACTION_PROMPT
print("prompts: escalation specification rule and extractor fidelity pinned: OK")

rec_t = verify.reconcile(ExtractMock("## Summary\n" + "x" * 300, truncated=True),
                         "m:x", "Q?", "orig", "audit")
assert rec_t.startswith("## Summary")
print("reconcile: truncated meta path returns text and warns: OK")

# ---------- directory guard ----------

tmp_prior = tempfile.mkdtemp(prefix="delve_vprior_")
tmp_out = tempfile.mkdtemp(prefix="delve_vout_")
try:
    verify.check_dirs(tmp_prior, tmp_prior)
    raise AssertionError("same-dir must be rejected")
except SystemExit as e:
    assert "different" in str(e)
try:
    verify.check_dirs(tmp_prior, tmp_out)
    raise AssertionError("missing briefing.md must be rejected")
except SystemExit as e:
    assert "briefing.md" in str(e)
with open(os.path.join(tmp_prior, "briefing.md"), "w") as f:
    f.write("## Summary\nprior text\n")
p = verify.check_dirs(tmp_prior, tmp_out)
assert p.endswith("briefing.md")
print("check_dirs: same-dir and missing-briefing guards: OK")

# ---------- reconcile: marker-remnant tolerance ----------

rec = verify.reconcile(ExtractMock("###BRIEFING##\n## Summary\nmerged.\n"),
                       "m:x", "Q?", "orig", "audit")
assert rec.startswith("## Summary"), rec[:40]
print("reconcile: stray leading briefing marker stripped: OK")

# ---------- finalize: normal path and the never-empty net ----------

reconciled_text = "## Summary\nmerged briefing with statuses.\n" + "y" * 300
path, fallback = verify.finalize_verify_outputs(
    tmp_out, "ORIGINAL DOC", "AUDIT DOC", reconciled_text, ["A is 1."])
assert not fallback
assert open(os.path.join(tmp_out, "briefing.md")).read() == reconciled_text
assert open(os.path.join(tmp_out, "briefing_original.md")).read() == "ORIGINAL DOC"
assert open(os.path.join(tmp_out, "briefing_audit.md")).read() == "AUDIT DOC"
assert "1. A is 1." in open(os.path.join(tmp_out, "claims.md")).read()
print("finalize: four artifacts, reconciled briefing is the deliverable: OK")

path, fallback = verify.finalize_verify_outputs(
    tmp_out, "ORIGINAL DOC", "AUDIT DOC", "  ", [])
assert fallback
assert open(os.path.join(tmp_out, "briefing.md")).read() == "AUDIT DOC"
print("finalize: empty reconciliation falls back to the audit briefing: OK")

# ---------- last-run pointer resolution ----------

assert verify.LAST == "@last"  # must match run_core's argparse const
assert verify.resolve_prior_dir("some/dir") == "some/dir"
ptr = os.path.join(tmp_out, "ptr")
with open(ptr, "w") as f:
    f.write("runs/my_last\n")
assert verify.resolve_prior_dir(verify.LAST, pointer_path=ptr) == "runs/my_last"
try:
    verify.resolve_prior_dir(verify.LAST, pointer_path=os.path.join(tmp_out, "absent"))
    raise AssertionError("missing pointer must be rejected")
except SystemExit as e:
    assert "PRIOR_RUN_DIR" in str(e)
verify.write_last_run_pointer("output", pointer_path=ptr)
assert open(ptr).read() == "output"
verify.write_last_run_pointer("x", pointer_path=os.path.join(tmp_out, "no_dir", "p"))
print("last-run pointer: sentinel, resolution, guards, best-effort write: OK")

# ---------- end-to-end chain mirroring main's verify flow ----------

from kernel import PersistentKernel
from investigation import run_investigation
from llm import RunStats

AUDIT_BRIEFING = ("## Summary\nClaim (1) refuted: proper metric shows no trend.\n"
                  "## Method notes\n- grouped uncertainty used.\n")

class VerifyFlowMock:
    """Plays ClaimExtractor, Investigator, Executor, Synthesizer, Reconciler."""
    def __init__(s):
        s.agents = []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        s.agents.append(agent)
        if agent == "Reconciler" and return_meta:
            return s.call(m, model, max_tokens, temperature, agent), {
                "output_tokens": 50, "max_tokens": max_tokens, "truncated": False}
        if agent == "ClaimExtractor":
            return "1. The trend is -0.010 per season (p=0.07).\n2. The boundary is -3.3."
        if agent == "Investigator":
            text = ("###THINKING###\nt\n###STATUS###\nSYNTHESIZE\n###SPEC###\nnone\n"
                    "###LEDGER###\nFRONTIER | claims | tested | steps: 1\n"
                    "REGIME | definition | examined | steps: 1\n")
            if not getattr(s, "_step_done", False):
                s._step_done = True
                text = text.replace("SYNTHESIZE", "CONTINUE").replace(
                    "###SPEC###\nnone", "###SPEC###\nmedian v by g; print.")
            if return_meta:
                return text, {"output_tokens": 40, "max_tokens": max_tokens,
                              "truncated": False}
            return text
        if agent == "Executor":
            return ("```python\nprint('###RESULTS_START###')\n"
                    "print(df.groupby('g')['v'].median().to_string())\n"
                    "print('###RESULTS_END###')\n```")
        if agent == "Synthesizer":
            return f"###VERDICT###\nFINAL\n###BRIEFING###\n{AUDIT_BRIEFING}"
        if agent == "Reconciler":
            return ("## Summary\nReconciled: claim 1 refuted, claim 2 attenuated.\n"
                    "## Verification record\n1. refuted\n2. attenuated\n" + "z" * 220)
        return ""

mk = VerifyFlowMock()
claims = verify.extract_claims(mk, "m:x", "## Summary\nprior\n")
assert len(claims) == 2
audit_seed = verify.compose_audit_seed("Original Q?", claims)
dfv = pd.DataFrame({"g": ["a", "b", "a", "b"], "v": [1.0, 2.0, 1.1, 2.2]})
outdir = tempfile.mkdtemp(prefix="delve_vflow_")
kk = PersistentKernel(df=dfv)
log, _, nav, audit_briefing = run_investigation(
    seed=audit_seed, df=dfv, client=mk, investigator_model="m:p",
    executor_model="m:c", schema_text="(s)", max_steps=5, output_dir=outdir,
    kernel=kk, nav=NavState(), stats=RunStats())
kk.cleanup()
assert audit_briefing.startswith("## Summary\nClaim (1) refuted")
reconciled = verify.reconcile(mk, "m:x", "Original Q?", "## Summary\nprior\n",
                              audit_briefing)
_, fb = verify.finalize_verify_outputs(outdir, "## Summary\nprior\n",
                                       audit_briefing, reconciled, claims)
assert not fb
final = open(os.path.join(outdir, "briefing.md")).read()
assert "Verification record" in final and "refuted" in final
assert open(os.path.join(outdir, "briefing_audit.md")).read() == audit_briefing
assert set(["ClaimExtractor", "Investigator", "Executor", "Synthesizer",
            "Reconciler"]) <= set(mk.agents)
print("end-to-end: extract -> audit run -> reconcile -> single briefing: OK")


# ---------- main()-level smokes (wiring, not functions: the level a chimera hides at) ----------

import subprocess
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=os.path.join(REPO, "tests", "stubs"))
def smoke(args, cwd):
    r = subprocess.run([sys.executable, os.path.join(REPO, "run_core.py"),
                        "dummy.csv"] + args, cwd=cwd, env=ENV,
                       capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip().splitlines()[-1]

smoke_dir = tempfile.mkdtemp(prefix="delve_smoke_")
assert "no briefing.md found" in smoke(["--verify", "/nonexistent_dir"], smoke_dir)
assert "no prior run recorded" in smoke(["--verify"], smoke_dir)
os.makedirs(os.path.join(smoke_dir, "priorrun"))
with open(os.path.join(smoke_dir, "priorrun", "briefing.md"), "w") as f:
    f.write("## Summary\nx\n")
with open(os.path.join(smoke_dir, ".delve_last_run"), "w") as f:
    f.write("priorrun")
assert "no saved question found in priorrun" in smoke(["--verify"], smoke_dir)
print("main() smokes: explicit bad dir, bare without pointer, pointer without question: OK")

print("OK: serial verification feature, all paths")

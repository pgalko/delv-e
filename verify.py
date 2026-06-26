"""Serial verification: audit a prior run's briefing with a fresh, independent pass.

Validated manually before being built (handover, ninth pass): a fresh delv-e run
whose seed carries the prior briefing's decisive claims plus a fixed audit
mandate caught 2.5 of 3 known artifacts in the prototype. This module automates
the three manual parts: claim extraction, audit-seed composition, and the
reconciliation of the two briefings into one. Independence is the design
invariant: the audit pass gets a clean kernel and clean context, never the prior
run's evidence chain (which is why this is not built on --extend).
"""

import logging
import os
import re

import prompts
from llm import call_with_ladder

logger = logging.getLogger("verify")

MAX_CLAIMS = 10
_FALLBACK_EXCERPT_CHARS = 4000
_MIN_RECONCILED_CHARS = 200

# Bare --verify (no directory) resolves through the last-run pointer file,
# written by every completed PRIMARY run. Verify runs never write it, so a
# bare --verify cannot self-select a previous audit.
LAST = "@last"
LAST_RUN_POINTER = ".delve_last_run"


def resolve_prior_dir(explicit, pointer_path=LAST_RUN_POINTER):
    """Resolve the directory to audit: an explicit path passes through; the
    bare-flag sentinel reads the last-run pointer."""
    if explicit != LAST:
        return explicit
    try:
        with open(pointer_path, encoding="utf-8") as f:
            prior = f.read().strip()
    except OSError:
        prior = ""
    if not prior:
        raise SystemExit("--verify: no prior run recorded "
                         f"({pointer_path} missing or empty); pass "
                         "PRIOR_RUN_DIR explicitly.")
    return prior


def write_last_run_pointer(output_dir, pointer_path=LAST_RUN_POINTER):
    """Record the most recent completed primary run. Best effort: a pointer
    failure must never fail a run."""
    try:
        with open(pointer_path, "w", encoding="utf-8") as f:
            f.write(output_dir)
    except OSError:
        logger.warning("Could not write the last-run pointer %s.", pointer_path)


def extract_claims(client, model, briefing_text, compute=False):
    """One cheap call that distills the briefing into numbered decisive claims.

    Returns a list of claim strings (possibly empty: the caller falls back to
    an excerpt of the briefing itself). Parsing is malformation-tolerant per the
    standing lesson: numbered lines in any of the forms "1." "2)" "3]" with or
    without markdown bold, and unnumbered continuation lines fold into the
    previous claim.
    """
    template = (prompts.CLAIM_EXTRACTION_PROMPT_COMPUTE if compute
                else prompts.CLAIM_EXTRACTION_PROMPT)
    out, _meta = call_with_ladder(
        client,
        [{"role": "user",
          "content": template.format(briefing=briefing_text)}],
        model=model, agent="ClaimExtractor")
    claims = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(?:\*\*|__)?\s*\d+\s*[.)\]]\s*(.+)$", line)
        if m:
            claims.append(m.group(1).strip())
        elif claims and not re.match(r"^#{1,6}\s", line):
            claims[-1] = claims[-1] + " " + line
    claims = [re.sub(r"(\*\*|__)", "", c).strip() for c in claims if c.strip()]
    return claims[:MAX_CLAIMS]


def claims_blob(claims, original_briefing):
    """The claims section of the audit seed, with a fallback when extraction
    yielded nothing: an excerpt of the report itself, so the audit can still
    adjudicate rather than the run dying on a parse failure."""
    if claims:
        return "\n".join(f"({i}) {c}" for i, c in enumerate(claims, 1))
    logger.warning("Claim extraction yielded no claims; auditing against a "
                   "briefing excerpt instead.")
    return ("Their report follows; adjudicate its decisive findings.\n\n"
            + original_briefing[:_FALLBACK_EXCERPT_CHARS])


def original_question(seeds):
    """The question an audit adjudicates is the full chain the audited briefing
    answered: the root problem first, then any extension instructions. Consecutive
    duplicates (an extend re-run with the same question) collapse. For a fresh run
    this is the single seed; for an extended run it restores the root problem that a
    bare last-seed lookup would miss, where the last seed is only an extension
    instruction like "now also test 24 mornings" and carries none of the model.
    """
    chain = []
    for s in seeds:
        if s and (not chain or s != chain[-1]):
            chain.append(s)
    return "\n\n".join(chain)


def compose_audit_seed(original_seed, claims, original_briefing="", compute=False):
    template = (prompts.AUDIT_SEED_TEMPLATE_COMPUTE if compute
                else prompts.AUDIT_SEED_TEMPLATE)
    return template.format(
        original_seed=original_seed.strip(),
        claims=claims_blob(claims, original_briefing))


def reconcile(client, model, original_seed, original_briefing, audit_briefing,
              compute=False):
    """One synthesis-grade call that merges the two documents into the single
    corrected briefing answering the original question."""
    template = (prompts.RECONCILIATION_PROMPT_COMPUTE if compute
                else prompts.RECONCILIATION_PROMPT)
    out, meta = call_with_ladder(
        client,
        [{"role": "user",
          "content": template.format(
              seed=original_seed.strip(), original=original_briefing,
              audit=audit_briefing)}],
        model=model, agent="Reconciler")
    if meta.get("truncated") or (meta.get("max_tokens")
                                 and meta.get("output_tokens") == meta.get("max_tokens")):
        logger.warning("Reconciliation hit its token cap; briefing.md may be "
                       "incomplete (briefing_audit.md is complete alongside).")
    text = out.strip()
    # Marker-remnant tolerance: strip a stray leading briefing marker if the
    # model echoes one (same failure family as the synthesis incidents).
    text = re.sub(r"^#{2,}\s*BRIEFING[^\n]*\n", "", text).strip()
    return text


def check_dirs(prior_dir, output_dir):
    """The audit must not write into the run it is auditing."""
    if os.path.abspath(prior_dir) == os.path.abspath(output_dir):
        raise SystemExit("--verify: choose an --output different from the "
                         "prior run directory (the audit must not overwrite "
                         "the run it audits).")
    briefing_path = os.path.join(prior_dir, "briefing.md")
    if not os.path.exists(briefing_path):
        raise SystemExit(f"--verify: no briefing.md found in {prior_dir}.")
    return briefing_path


def finalize_verify_outputs(output_dir, original_briefing, audit_briefing,
                            reconciled, claims):
    """Write the verification artifact set. briefing.md is the reconciled
    document; the inputs are preserved alongside for the audit trail. If
    reconciliation produced nothing usable, the audit briefing stands in (never
    end empty-handed) and the fallback is reported."""
    fallback = not reconciled or len(reconciled.strip()) < _MIN_RECONCILED_CHARS
    final_text = audit_briefing if fallback else reconciled
    if fallback:
        logger.warning("Reconciliation produced no usable briefing; the audit "
                       "briefing stands in as briefing.md.")
    paths = {
        "briefing.md": final_text,
        "briefing_original.md": original_briefing,
        "briefing_audit.md": audit_briefing,
        "claims.md": "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
                     or "(claim extraction fell back to a briefing excerpt)",
    }
    for name, text in paths.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
    return os.path.join(output_dir, "briefing.md"), fallback
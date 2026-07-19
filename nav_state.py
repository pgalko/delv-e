"""
Navigational state for delv-e's inverted-core loop (step 3, spec §7).

THE ONE RULE: this is a navigation aid, not the answer. It tracks coverage and
the frontier so the Investigator knows what to probe next and the Synthesizer
knows what evidence exists to reason over. The answer is NEVER read from here —
it is always re-derived holistically over the RAW step outputs.

How it resists decaying into an answer-digest (the original failure mode where a
cheap agent re-authored a multi-section prose "research model" every turn and
downstream reasoning treated it as ground truth):

  - It is a STRUCTURED ledger of short handles + status enums + STEP POINTERS,
    not prose conclusions. The numbers live in the referenced steps; the ledger
    only says *that* estimand X was tested by steps [3,5], not what it concluded.
  - The only free text allowed is a terse reason on a BREAKDOWN line (why a
    regime is blocked/thin) — a reason, never a numeric verdict.
  - The evidence index is built MECHANICALLY from the log; no model authors it.
  - Nothing here is a cached "established findings" blob. A current-best-answer
    is not stored; it is produced by the Synthesizer from raw evidence.

The four investigator-maintained ledgers:
  FRONTIER  — candidate framings/estimands and whether they've been tried.
              Crucially includes UNTESTED ones: the adjacent possible to probe.
  REGIME    — STRATIFICATION / effect-modifier axes: has the main effect been
              estimated WITHIN each level of the axis (testing whether the
              effect VARIES across levels)? This is NOT a list of confounders
              you controlled. Restricting to asphalt to remove a surface
              confound is NOT examining the surface regime; estimating the
              effect separately at each intensity level IS examining the
              intensity regime. This distinction is what makes G1 bite — delve
              controlled surface/heat/taper as confounders yet never estimated
              the effect within an intensity axis, and returned a false null.
  RISK      — threats to the current reading not yet tested.
  BREAKDOWN — where the answer holds / is thin / blocked / unrecoverable + why.

Wire format (the Investigator emits these lines in a ###LEDGER### block, under a section
header per kind). This is the SAME shape render_for_investigator shows the model, so a model
that simply echoes the map it is shown round-trips cleanly. apply_ledger_block also still
accepts the legacy pipe shape `KIND | label | status | steps` for back-compat.
  FRONTIER:
    <handle> [untested|in_progress|tested|foreclosed] steps:1,3
  REGIME:
    <axis> [not_examined|partial|examined] steps:4,5
  RISK:
    <handle> [open|resolved] steps:-
  BREAKDOWN:
    <where> [holds|thin|blocked|unrecoverable] steps:7
  (a BREAKDOWN entry may append an optional terse reason after its steps; the parser
   reads it and the render shows it back)
"""

import re

from logger_config import get_logger

logger = get_logger(__name__)

FRONTIER_STATUS = {"untested", "in_progress", "tested", "foreclosed"}
REGIME_STATUS = {"not_examined", "partial", "examined"}
RISK_STATUS = {"open", "resolved"}
BREAKDOWN_STATUS = {"holds", "thin", "blocked", "unrecoverable"}

# Statuses that mean a thread is still LIVE. An entry in one of these that
# disappears from a re-emitted ledger — rather than passing through a closing
# status (tested/foreclosed/examined/resolved) — is the silent-drift signal the
# per-section merge warns about. BREAKDOWN entries are terminal descriptors, so
# their removal is logged but never warned.
_LIVE_STATUS = {"frontier": {"untested", "in_progress"},
                "regime": {"not_examined", "partial"},
                "risk": {"open"}}

# Loose synonym mapping so minor model drift in status words still parses.
_STATUS_SYNONYMS = {
    "in progress": "in_progress", "inprogress": "in_progress",
    "done": "tested", "complete": "tested", "completed": "tested",
    "closed": "foreclosed", "dead": "foreclosed", "ruled_out": "foreclosed",
    "not examined": "not_examined", "unexamined": "not_examined", "none": "not_examined",
    "partially": "partial", "partial_examined": "partial",
    "examined_within": "examined", "checked": "examined",
    "unresolved": "open", "mitigated": "resolved",
    "stable": "holds", "ok": "holds",
    "sparse": "thin", "weak": "thin",
    "unidentifiable": "unrecoverable", "not_recoverable": "unrecoverable",
    "not_identifiable": "unrecoverable", "blocked_": "blocked",
}


def _norm_status(raw, valid, default):
    s = (raw or "").strip().lower()
    s = _STATUS_SYNONYMS.get(s, s)
    return s if s in valid else default


def _parse_steps(raw):
    """Parse a 'steps:' field like '1,3,5' or '-' or 'none' into a list[int]."""
    if not raw:
        return []
    raw = raw.split(":", 1)[-1] if ":" in raw else raw
    out = []
    for tok in raw.replace(" ", "").split(","):
        if not tok or tok in ("-", "none", "na", "n/a"):
            continue
        # tolerate 'step3' / '#3' / '3'
        digits = "".join(ch for ch in tok if ch.isdigit())
        if digits:
            out.append(int(digits))
    return sorted(set(out))


# Generic markers that the analysis estimated something WITHIN subgroups/levels
# rather than only pooled — the observable footprint of a G1 stratification. These
# are domain-agnostic pandas/stats idioms; no column names or use-case specifics.
_STRATIFY_PATTERNS = [
    r"\.groupby\s*\(",            # df.groupby(axis)...
    r"\.pivot_table\s*\(",        # pivot_table(..., index/columns=axis)
    r"\.pivot\s*\(",
    r"\bcrosstab\s*\(",           # pd.crosstab(a, b)
    r"\.(?:cut|qcut)\s*\(",       # binning a continuous axis into levels
    r"\bcut\s*\(", r"\bqcut\s*\(",
    r"\.resample\s*\(",           # stratify over time
    r"for\s+\w+\s+in\s+[^\n:]*\.unique\s*\(\)",   # per-level loop
    r"for\s+\w+\s*,\s*\w+\s+in\s+[^\n:]*\.groupby", # for k, g in df.groupby(...)
    r"C\([^)]+\)\s*[:*]\s*",      # interaction term in a model formula: C(x):z
    r"[:*]\s*C\([^)]+\)",
    r"~[^#\n]*[:*][^#\n]*",       # any '*' or ':' interaction inside a formula RHS
]
_STRATIFY_RE = re.compile("|".join(_STRATIFY_PATTERNS))


def code_shows_stratification(log):
    """Has any executed step actually stratified the analysis (grouped, binned,
    looped per level, or fit an interaction)? Generic backstop for G1 so a model
    that does the work but mis-marks its ledger isn't falsely gated. Scans only
    real executed code, never prose, and is intentionally permissive: it is used
    only to UPGRADE an unmarked ledger, so a false positive can at worst let a
    synthesis proceed (which still re-derives over raw evidence)."""
    if not log:
        return False
    for e in log:
        if e.get("terminal"):
            continue
        code = e.get("code") or ""
        if code and _STRATIFY_RE.search(code):
            return True
    return False


class Entry:
    __slots__ = ("kind", "label", "status", "steps", "why")

    def __init__(self, kind, label, status, steps=None, why=""):
        self.kind = kind
        self.label = label
        self.status = status
        self.steps = steps or []
        self.why = why

    def to_dict(self):
        d = {"label": self.label, "status": self.status, "steps": self.steps}
        if self.kind == "breakdown":
            d["why"] = self.why
        return d


class NavState:
    """Structured pointer/status ledger. Re-emitted in full by the Investigator
    each turn and merged PER SECTION from its ###LEDGER### block (a section is
    replaced only when it parsed; see apply_ledger_block), constrained by format
    to handles + statuses + step pointers, so 're-emission' can't smuggle in a
    prose answer and a partially garbled turn can't wipe the map."""

    def __init__(self):
        self.frontier = []
        self.regimes = []
        self.risks = []
        self.breakdown = []
        # The specific contrast/quantity the seed question asks for, named once by
        # the Investigator and pinned (shown back every turn, never silently drifting
        # into a proxy). Empty until the Investigator states it. See G3 (synthesis).
        self.target_estimand = ""

    # ---- ingest the Investigator's ledger block -------------------------

    def apply_ledger_block(self, text):
        """Parse a ###LEDGER### block and MERGE it into the ledgers, per section.

        Accepts the CANONICAL shape the Investigator is shown and asked to emit:
        a section header (FRONTIER / REGIME / RISK / BREAKDOWN, bare or decorated
        like 'REGIME LEDGER:' or 'OPEN RISKS:') followed by indented entry lines
        `label [status] steps:ids`, with BREAKDOWN optionally adding ` — why: ...`.
        Also still accepts the legacy pipe shape `KIND | label | status | steps`
        so older behaviour never breaks. Individual malformed lines are skipped.

        REPLACEMENT IS PER SECTION, never wholesale. A bucket is replaced only
        when its section yielded at least one parsed entry this turn; a section
        that is absent, explicitly "(none)", or whose every line was malformed
        keeps its prior contents. Under the old wholesale rule, one parseable
        line anywhere replaced ALL FOUR buckets, so a turn with garbled
        FRONTIER/REGIME lines and one good RISK line silently wiped the frontier
        and regimes — and an emptied REGIME bucket disarms the G1 gate
        (open_regimes() drives both the loop's pre-synthesis pushback and the
        Synthesizer's backstop). Entries are closed by STATUS
        (tested/foreclosed/resolved), never by omission, so keeping a prior
        section on a bad parse matches the ledger's own rules. A totally
        empty/garbled block still leaves the whole prior state intact.

        Every applied block is DIFFED against the prior state: additions and
        removals are logged, and a removed entry whose prior status was still
        LIVE (frontier untested/in_progress, regime not_examined/partial, risk
        open) is logged as a WARNING — the silent-drift signal section 7 of the
        handover says to keep watching for, now mechanical. Returns the diff
        dict ({kind: {"added": [...], "removed": [...], "dropped_live": [...]}})
        for replaced sections, or None when nothing was applied.

        Render and parser share this shape, so a model that simply echoes the
        map it is shown round-trips cleanly.
        """
        if not text or not text.strip():
            return None
        buckets = {"frontier": [], "regime": [], "risk": [], "breakdown": []}
        defaults = {"frontier": ("untested", FRONTIER_STATUS),
                    "regime": ("not_examined", REGIME_STATUS),
                    "risk": ("open", RISK_STATUS),
                    "breakdown": ("thin", BREAKDOWN_STATUS)}
        seen_sections = set()      # section headers encountered this block
        explicit_empty = set()     # sections whose body was a literal "(none)"
        current = None  # section in force while reading header+entry lines

        def _section_of(line):
            """'frontier'/'regime'/'risk'/'breakdown' if this bracket-free,
            pipe-free line is that section's header; '' for a non-ledger header
            (e.g. EVIDENCE INDEX) that should stop bucketing; None if not a header."""
            if "[" in line or "|" in line:
                return None
            low = line.lower()
            for kw in ("frontier", "regime", "risk", "breakdown"):
                if kw in low:
                    return kw
            if "evidence" in low or "index" in low:
                return ""
            return None

        entry_re = re.compile(r"^(?P<label>.+?)\s*\[(?P<status>[^\]]+)\]\s*(?P<rest>.*)$")

        def _steps_why_from_fields(fields):
            steps, why = [], ""
            for f in fields:
                fl = f.lower()
                if fl.startswith("why"):
                    why = f.split(":", 1)[-1].strip()
                elif "step" in fl or re.fullmatch(r"[0-9,\s-]+", f or ""):
                    steps = _parse_steps(f)
            return steps, why

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            # A header line (bare or decorated) sets the active section.
            sec = _section_of(line)
            if sec is not None:
                current = sec or None
                if current:
                    seen_sections.add(current)
                continue

            # A literal "(none)" body is the render's empty-section marker; a
            # model echoing it is stating "no entries", not garbling a line.
            if current and re.fullmatch(r"\(?\s*(?:none|n/?a|-)\s*\)?", line,
                                        re.IGNORECASE):
                explicit_empty.add(current)
                continue

            kind = label = status_raw = None
            steps, why, lenient = [], "", False

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if parts[0].lower() in buckets and len(parts) >= 3:
                    # legacy: KIND | label | status | tail...
                    kind, label, status_raw = parts[0].lower(), parts[1], parts[2]
                    steps, why = _steps_why_from_fields(parts[3:])
                    lenient = True  # keep old forgiving behaviour for this shape
                elif current and len(parts) >= 2:
                    # section header above + pipe fields without a kind prefix
                    kind, label, status_raw = current, parts[0], parts[1]
                    steps, why = _steps_why_from_fields(parts[2:])
                else:
                    continue
            else:
                # canonical bracket entry under the active section
                m = entry_re.match(line)
                if not (m and current):
                    continue
                kind, label, status_raw = current, m.group("label"), m.group("status")
                rest = m.group("rest")
                sm = re.search(r"steps?\s*:?\s*(-|[0-9][0-9,\s]*)", rest)
                if sm:
                    steps = _parse_steps(sm.group(1))
                wm = re.search(r"why\s*:\s*(.+)$", rest, re.IGNORECASE)
                if wm:
                    why = wm.group(1).strip()

            default_status, valid = defaults[kind]
            status = _norm_status(status_raw, valid, default_status if lenient else None)
            if status is None:
                continue  # bracketed/again token is not a real status -> skip the line
            label = (label or "").strip().lstrip("-").strip()
            if not label:
                continue
            buckets[kind].append(Entry(kind, label, status, steps, why))

        # ---- per-section merge + mechanical drift diff ----
        prior = {"frontier": self.frontier, "regime": self.regimes,
                 "risk": self.risks, "breakdown": self.breakdown}
        replaced = {k for k, v in buckets.items() if v}
        if not replaced:
            return None  # nothing parsed anywhere: whole prior map intact, as before

        # A section whose header (or explicit "(none)") appeared but whose lines
        # all failed to parse is the wipe-hazard tell: say so, keep the prior.
        for kind in ("frontier", "regime", "risk", "breakdown"):
            if kind in replaced or not prior[kind]:
                continue
            if kind in explicit_empty:
                logger.warning("Ledger: %s emitted as '(none)' while %d prior "
                               "entr%s exist; entries close by status, never by "
                               "omission — prior kept.", kind, len(prior[kind]),
                               "y" if len(prior[kind]) == 1 else "ies")
            elif kind in seen_sections:
                logger.warning("Ledger: %s section present but no line parsed; "
                               "prior entries kept.", kind)

        diff = {}
        for kind in replaced:
            old = {e.label: e for e in prior[kind]}
            new = {e.label for e in buckets[kind]}
            added = [l for l in new if l not in old]
            removed = [l for l in old if l not in new]
            live = _LIVE_STATUS.get(kind, set())
            dropped_live = [l for l in removed if old[l].status in live]
            diff[kind] = {"added": added, "removed": removed,
                          "dropped_live": dropped_live}
            if added or removed:
                logger.info("Ledger diff (%s): +%s -%s", kind,
                            added or "[]", removed or "[]")
            if dropped_live:
                logger.warning("Ledger: %s entr%s %s disappeared while still "
                               "LIVE (never passed through a closing status) — "
                               "renamed or dropped by the model.", kind,
                               "y" if len(dropped_live) == 1 else "ies",
                               dropped_live)

        if "frontier" in replaced:
            self.frontier = buckets["frontier"]
        if "regime" in replaced:
            self.regimes = buckets["regime"]
        if "risk" in replaced:
            self.risks = buckets["risk"]
        if "breakdown" in replaced:
            self.breakdown = buckets["breakdown"]
        return diff

    # ---- guardrail support ---------------------------------------------

    def g1_satisfied(self, log=None):
        """True once the main effect has been examined within >=1 regime axis.
        Used to flag a premature null/uniform/synthesize (spec §9 G1).

        Primary signal is the model's own ledger (a regime marked examined/partial).
        When `log` is supplied, a CODE-GROUNDED backstop also counts G1 as satisfied
        if any executed step actually stratified the analysis (groupby / pivot / bins
        / per-level loop / interaction term). This makes the gate robust to a model
        that does the stratification but keeps a sloppy ledger: it can only ever
        UPGRADE not-marked to satisfied on observable evidence, never the reverse."""
        if any(r.status in ("examined", "partial") for r in self.regimes):
            return True
        if log is not None and code_shows_stratification(log):
            return True
        return False

    def untested_frontier(self):
        return [e.label for e in self.frontier if e.status == "untested"]

    def open_regimes(self):
        return [e.label for e in self.regimes if e.status == "not_examined"]

    def load_bearing_steps(self):
        """Step ids referenced by any frontier/regime/breakdown entry. These are
        the steps whose RAW output must never be trimmed when context is tight —
        they are the evidence the current map rests on."""
        ids = set()
        for coll in (self.frontier, self.regimes, self.breakdown):
            for e in coll:
                for s in getattr(e, "steps", []) or []:
                    ids.add(s)
        return ids

    def protected_steps(self, max_protected=6):
        """Steps whose full raw must stay resident because they feed a LIVE thread.
        Tightened so history compaction actually fires: for each live thread we keep
        only its MOST RECENT step (older steps of a long-running thread collapse to a
        headline and are rehydratable), and the whole set is capped to the most
        recent `max_protected`. Steps feeding only tested/foreclosed/resolved threads
        are never protected."""
        keep = set()
        for f in self.frontier:
            if f.status not in ("tested", "foreclosed") and f.steps:
                keep.add(max(f.steps))          # latest step of a live frontier item
        for r in self.risks:
            if r.status == "open" and r.steps:
                keep.add(max(r.steps))          # latest step of an open risk
        if len(keep) > max_protected:
            keep = set(sorted(keep)[-max_protected:])
        return keep

    # ---- rendering ------------------------------------------------------

    @staticmethod
    def evidence_index(log, spec_chars=90):
        """MECHANICAL view of the log: step -> spec handle + has-output/error.
        Lets any reader resolve a step pointer back to what was run. Not stored
        in NavState; rebuilt from the log on demand (and after --resume/--extend)."""
        lines = []
        for e in log:
            if e.get("terminal"):
                continue
            spec = (e.get("spec") or "").replace("\n", " ")
            if len(spec) > spec_chars:
                spec = spec[:spec_chars] + "…"
            tag = "ERROR" if e.get("error") else "ok"
            lines.append(f"  step {e['step']}: [{tag}] {spec}")
        return "\n".join(lines) if lines else "  (no steps yet)"

    def render_for_investigator(self, log=None):
        """The structured view passed back to the Investigator each turn.

        Bare section headers by design (audit 4.2): the sections' meanings,
        status vocabularies, and G1's rule are taught ONCE in the cached system
        prompt, so re-teaching them here re-billed ~600 uncached chars every
        turn and made the legend a third leg of the render/legend/parser
        contract. The header KEYWORDS (frontier/regime/risk/breakdown) are
        load-bearing — apply_ledger_block's section detection keys on them and
        the model echoes this shape back — so change them only with the parser.
        Pass `log` to append the EVIDENCE INDEX (Synthesizer payload); the
        Investigator's call site omits it, since each step's spec already sits
        in its history region."""
        def fmt(entries, with_why=False):
            if not entries:
                return "  (none)"
            out = []
            for e in entries:
                steps = ",".join(map(str, e.steps)) if e.steps else "-"
                base = f"  {e.label} [{e.status}] steps:{steps}"
                if with_why and e.why:
                    base += f" — why: {e.why}"
                out.append(base)
            return "\n".join(out)

        sections = []
        if self.target_estimand:
            sections += [
                "TARGET ESTIMAND (pinned — aim the primary answer at this):",
                f"  {self.target_estimand}",
                "",
            ]
        sections += [
            "FRONTIER:",
            fmt(self.frontier),
            "REGIME LEDGER:",
            fmt(self.regimes),
            "OPEN RISKS:",
            fmt(self.risks),
            "BREAKDOWN MAP:",
            fmt(self.breakdown, with_why=True),
        ]
        if log is not None:
            sections += ["EVIDENCE INDEX (step pointers — raw results are in the log above):",
                         self.evidence_index(log)]
        return "\n".join(sections)

    def to_markdown(self):
        """For artifacts/dashboard. Still pointer-based — not an answer."""
        return self.render_for_investigator(log=None)

    # ---- persistence (for --resume/--extend / dashboard) ----------------

    def to_dict(self):
        return {
            "target_estimand": self.target_estimand,
            "frontier": [e.to_dict() for e in self.frontier],
            "regimes": [e.to_dict() for e in self.regimes],
            "risks": [e.to_dict() for e in self.risks],
            "breakdown": [e.to_dict() for e in self.breakdown],
        }

    @classmethod
    def from_dict(cls, d):
        ns = cls()
        d = d or {}
        ns.target_estimand = d.get("target_estimand", "")
        ns.frontier = [Entry("frontier", x["label"], x["status"], x.get("steps", []))
                       for x in d.get("frontier", [])]
        ns.regimes = [Entry("regime", x["label"], x["status"], x.get("steps", []))
                      for x in d.get("regimes", [])]
        ns.risks = [Entry("risk", x["label"], x["status"], x.get("steps", []))
                    for x in d.get("risks", [])]
        ns.breakdown = [Entry("breakdown", x["label"], x["status"], x.get("steps", []),
                              x.get("why", ""))
                        for x in d.get("breakdown", [])]
        return ns
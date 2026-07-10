# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
def _require_dataset():
    p = os.environ.get("EEDR_DATASET", os.path.join(_ROOT, "datasets", "EEDR_sessions_laps_enriched.csv"))
    if not os.path.exists(p):
        print(f"SKIP: EEDR dataset not found at {p} (set EEDR_DATASET or place the CSV in datasets/)")
        sys.exit(0)
    return p
# --- end bootstrap ---

from investigation import _decision_from_status, _parse_investigator

# 1) bare verbs (the instructed form)
assert _decision_from_status("CONTINUE") == "CONTINUE"
assert _decision_from_status("SYNTHESIZE") == "SYNTHESIZE"
assert _decision_from_status("SEARCH") == "SEARCH"
assert _decision_from_status("continue") == "CONTINUE"          # case-insensitive
assert _decision_from_status("Synthesize") == "SYNTHESIZE"
print("bare verbs: OK")

# 2) the dangerous misfires the old substring parser got WRONG
#    a CONTINUE that merely MENTIONS synthesize/search must stay CONTINUE.
assert _decision_from_status("CONTINUE, not ready to synthesize yet") == "CONTINUE"
assert _decision_from_status("CONTINUE - no search needed") == "CONTINUE"
assert _decision_from_status("CONTINUE, still need to search the literature later") == "CONTINUE"
assert _decision_from_status("CONTINUE then SYNTHESIZE once the regime check is done") == "CONTINUE"
print("prose-wrapped CONTINUE never misfires to SYNTHESIZE/SEARCH: OK")

# 3) leading verb wins even with trailing prose
assert _decision_from_status("SYNTHESIZE now, the evidence is sufficient") == "SYNTHESIZE"
assert _decision_from_status("SEARCH the literature first, then continue") == "SEARCH"
print("leading verb decides: OK")

# 4) verb-less but decisive (no CONTINUE present) falls back sensibly
assert _decision_from_status("ready to synthesize") == "SYNTHESIZE"
assert _decision_from_status("let's search external sources") == "SEARCH"
print("verb-less fallback: OK")

# 5) empty / whitespace / junk -> safe default CONTINUE (never finalize on nothing)
assert _decision_from_status("") == "CONTINUE"
assert _decision_from_status("   ") == "CONTINUE"
assert _decision_from_status("...") == "CONTINUE"
print("empty/junk -> CONTINUE: OK")

# 6) end-to-end through the real parser, with a full ###STATUS### block
def _status_via_parser(status_text, body="###SPEC###\nrun something\n"):
    text = f"###THINKING###\nreasoning here\n###STATUS###\n{status_text}\n{body}"
    return _parse_investigator(text)["status"]

assert _status_via_parser("CONTINUE, not ready to synthesize") == "CONTINUE"
assert _status_via_parser("SYNTHESIZE") == "SYNTHESIZE"
assert _status_via_parser("SEARCH") == "SEARCH"
# the markerless fallback (no THINKING, no SPEC) still finalizes, unchanged by this work
assert _parse_investigator("just some prose with no markers at all")["status"] == "SYNTHESIZE"
print("end-to-end via _parse_investigator: OK")

print("\nALL STATUS-HARDENING ASSERTIONS PASSED")

# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Briefing charts, v2 contract (reworked after the first live glm run): the
# Synthesizer emits CHART/SECTION/CAPTION/SPEC entries and writes NO image
# links; the harness renders each spec through the Executor against the live
# kernel namespace and INSERTS each produced chart at the end of the section
# its SECTION field names. Placement moved to the harness because the live run
# dumped every image link after the last header despite instructions, and the
# annotation ban moved to spec level (plus a directive override) because the
# live specs asked for fifteen point labels. Pins: legend v2 in both modes,
# the directive's override precedence and sizing, dict parsing with optional
# fields, deterministic placement (fuzzy header match, same-section ordering,
# missing-section fallback), stripping of model-authored image lines (the live
# failure verbatim), failed charts simply absent, render-loop isolation and
# manifest, and byte-identical zero-charts behavior.

import inspect
import json
import tempfile

import prompts as P
import synthesis as S
import investigation as I
from synthesis import (_parse_synth, _parse_charts, sanitize_chart_name,
                       apply_chart_results, MAX_CHARTS)

# ── 1) Legend v2 in both modes; directive v2 carries sizing + override ──
src = inspect.getsource(P)
assert src.count("###CHARTS###") >= 2
assert src.count("Do NOT put image links in the BRIEFING") == 2
assert src.count("Never ask for text annotations") == 2
assert src.count("values from separate model fits share no scale") == 2, \
    "the encoding-faithfulness rule must exist in both modes"
assert src.count("chart a quantity that carries it") == 2
assert src.count("HEADLINE claim visible at a glance") == 2
d = P.CHART_STYLE_DIRECTIVE.format(name="x.png")
for phrase in ('fig.savefig("x.png", dpi=115)', "figsize=(8, 4.5)",
               "EVEN IF THE SPEC ASKS FOR THEM", "frameon=False",
               "horizontal bars sorted by value", "small grey points",
               "do not recompute the analysis", "SAVED x.png"):
    assert phrase in d, f"directive lost: {phrase!r}"
print("legend v2 + directive v2: OK")

# ── 2) Parse: four fields, optional middles, sanitize, dedupe, cap ──
block = ("CHART: Joint Scatter.PNG\n"
         "SECTION: ## What the data can answer\n"
         "CAPTION: Verstappen is the joint outlier (r=0.87)\n"
         "SPEC: scatter of joint_abilities_n30, x ability_finish, y ability_grid.\n"
         "Second spec line.\n"
         "CHART: era_leaders.png\n"
         "SPEC: horizontal bars of era_ability_summary top entity per era")
cs = _parse_charts(block)
assert [c["name"] for c in cs] == ["joint_scatter.png", "era_leaders.png"]
assert cs[0]["section"] == "## What the data can answer"
assert cs[0]["caption"].startswith("Verstappen")
assert cs[0]["spec"].endswith("Second spec line.")
assert cs[1]["section"] == "" and cs[1]["caption"] == ""
many = "\n".join(f"CHART: c{i}.png\nSPEC: s{i}" for i in range(5))
assert len(_parse_charts(many)) == MAX_CHARTS == 3
assert sanitize_chart_name("../Evil Name.PNG") == "evil_name.png"
t = ("###VERDICT###\nFINAL\n###BRIEFING###\nBody\n###CHARTS###\n"
     "CHART: a.png\nSECTION: Summary\nCAPTION: c\nSPEC: s")
r = _parse_synth(t)
assert r["briefing"] == "Body" and r["charts"][0]["name"] == "a.png"
assert _parse_synth("###VERDICT###\nFINAL\n###BRIEFING###\nB")["charts"] == []
print("parse v2: OK")

# ── 3) Placement: the live failure as a regression ──
briefing = ("## Summary\nS text.\n\n## What the data can answer\nW text.\n"
            "More W.\n\n## Method notes\nM text.\n\n"
            "![model-dumped one](charts/joint_scatter.png)\n"
            "![model-dumped two](charts/era_leaders.png)")
charts = [
    {"name": "joint_scatter.png", "section": "What the data can answer",
     "caption": "Joint outlier", "spec": "s1"},
    {"name": "era_leaders.png", "section": "## summary",
     "caption": "No single era leader", "spec": "s2"},
    {"name": "gone.png", "section": "Summary", "caption": "x", "spec": "s3"},
]
out = apply_chart_results(briefing, charts, {"joint_scatter.png", "era_leaders.png"})
lines = out.split("\n")
assert "model-dumped" not in out, "model-authored image lines are stripped"
i_sum, i_what, i_meth = (lines.index("## Summary"),
                         lines.index("## What the data can answer"),
                         lines.index("## Method notes"))
i_era = lines.index("![No single era leader](charts/era_leaders.png)")
i_joint = lines.index("![Joint outlier](charts/joint_scatter.png)")
assert i_sum < i_era < i_what, "era chart lands inside Summary (fuzzy '## summary' match)"
assert i_what < i_joint < i_meth, "joint chart lands inside its named section"
assert "gone.png" not in out, "a failed chart simply does not appear"
# same-section ordering + missing-section fallback
two = [{"name": "a.png", "section": "Summary", "caption": "A", "spec": "s"},
       {"name": "b.png", "section": "Summary", "caption": "B", "spec": "s"},
       {"name": "c.png", "section": "No Such Header", "caption": "C", "spec": "s"}]
out2 = apply_chart_results("## Summary\ntext\n\n## End\nz", two, {"a.png", "b.png", "c.png"})
l2 = out2.split("\n")
assert l2.index("![A](charts/a.png)") < l2.index("![B](charts/b.png)") < l2.index("## End")
assert out2.rstrip().endswith("![C](charts/c.png)"), "missing section falls back to the end"
assert apply_chart_results("plain text", [], set()) == "plain text"
print("placement: live regression + ordering + fallback: OK")

# ── 4) Render loop: dict charts, isolation, manifest with section/caption ──
class FakeKernel:
    def describe_namespace(self, **kw):
        return "- joint_abilities_n30: DataFrame"


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def run(self, spec, kernel, registry_text, analysis_dir=None):
        self.calls.append(spec)
        name = spec.split('fig.savefig("')[1].split('"')[0]
        if name == "ok.png":
            os.makedirs(analysis_dir, exist_ok=True)
            open(os.path.join(analysis_dir, name), "wb").write(b"png")
            return {"code": "c", "stdout": f"SAVED {name}", "error": None, "attempts": 1}
        return {"code": "c", "stdout": "", "error": "boom", "attempts": 3}


with tempfile.TemporaryDirectory() as out_dir:
    ex = FakeExecutor()
    produced = I._render_charts(
        ex, FakeKernel(),
        [{"name": "ok.png", "section": "Summary", "caption": "cap", "spec": "plot it"},
         {"name": "no.png", "section": "", "caption": "", "spec": "plot other"}],
        out_dir)
    assert produced == {"ok.png"}
    assert ex.calls[0].startswith("plot it\n\nCHART STYLE") and 'dpi=115' in ex.calls[0]
    man = json.load(open(os.path.join(out_dir, "charts", "manifest.json")))
    assert man[0]["section"] == "Summary" and man[0]["caption"] == "cap"
    assert man[1]["produced"] is False and man[1]["error"] == "boom"
print("render loop v2: OK")

print("test_charts: OK")
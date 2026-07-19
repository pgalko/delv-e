# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# The chart contract: what the technical pass may ask for, and how the harness
# renders it. Each rule here was written after a live chart failed:
#   v1 dumped every image link at the end of the document, so the harness owns
#     placement and the model only says WHERE (a [[CHART:Fn]] marker);
#   v1 charts were unreadable with annotations, callouts and floating stats, so
#     identifiers belong in the data (tick labels, legend, colour), never as text;
#   a v2 chart drew a y=x line between two quantities the method notes said shared
#     no scale, and bars comparing magnitudes across separate fits, so a chart may
#     not encode a comparison the analysis itself disclaims;
#   and the chart that would have shown the headline was never drawn, so the
#     headline claim is what a chart should make visible at a glance.
# Charts now key on a FINDING rather than a section header, because the editor
# chooses its own headings and a header is no longer a stable anchor.

import inspect
import re

import prompts as P
from synthesis import (MAX_CHARTS, _parse_charts, charts_for_editor,
                       render_chart_markers, sanitize_chart_name)

src = inspect.getsource(P)

# ── 1) The legend exists in both modes and keys on findings ──
assert src.count("###CHARTS###") >= 2
assert src.count("FINDING: <the id of the finding") == 2
assert "SECTION:" not in src, "a section header is not a stable anchor for the editor"
assert src.count("Never ask for text annotations") == 2
assert src.count("values from separate model fits share no scale") == 2, \
    "the encoding-faithfulness rule must exist in both modes"
assert src.count("chart a quantity that carries it") == 2
assert src.count("HEADLINE claim visible at a glance") == 2
print("legend: both modes, finding-keyed, faithfulness + headline rules: OK")

# ── 2) Parsing a chart block ──
block = ("CHART: Era Leaders!.png\n"
         "FINDING: f2\n"
         "CAPTION: the leader in each era\n"
         "SPEC: horizontal bars from era_summary, sorted by value\n"
         "\n"
         "CHART: second.png\n"
         "FINDING: F3\n"
         "CAPTION: the second one\n"
         "SPEC: a line over time\n")
charts = _parse_charts(block)
assert len(charts) == 2
assert charts[0]["name"] == sanitize_chart_name("Era Leaders!.png")
assert charts[0]["finding"] == "F2", "finding ids normalise to upper case"
assert charts[1]["finding"] == "F3"
assert "sorted by value" in charts[0]["spec"]
print("parsing: names sanitised, findings normalised: OK")

# ── 3) The cap holds ──
many = "".join(f"CHART: c{i}.png\nFINDING: F{i}\nCAPTION: c\nSPEC: s\n\n" for i in range(6))
assert len(_parse_charts(many)) == MAX_CHARTS == 3
print(f"cap: at most {MAX_CHARTS} charts: OK")

# ── 4) Rendering: the model says where, the harness says what ──
produced = {"c0.png", "c1.png"}
cs = _parse_charts("".join(f"CHART: c{i}.png\nFINDING: F{i}\nCAPTION: cap{i}\nSPEC: s\n\n"
                          for i in range(2)))
out = render_chart_markers("A [[CHART:F0]]\n\nB\n\n[[CHART:F1]]", cs, produced)
assert out.count("![") == 2 and "[[CHART" not in out
assert "![cap0](charts/c0.png)" in out and "![cap1](charts/c1.png)" in out
# an image link is never emitted for a chart that failed to render
assert "![" not in render_chart_markers("[[CHART:F0]][[CHART:F1]]", cs, set())
# and a rendered chart the editor forgot is appended rather than lost
tail = render_chart_markers("No markers at all.", cs, produced)
assert tail.count("![") == 2
print("rendering: no broken links, no lost charts: OK")

# ── 5) What the editor is told about the charts ──
listing = charts_for_editor(cs, produced)
assert "[[CHART:F0]] -> c0.png: cap0" in listing
assert charts_for_editor(cs, set()) == "(no charts were produced)"
print("manifest for the editor: OK")

# ── 6) The editor is told to place markers, never links ──
assert "[[CHART:F3]]" in P.EDITOR_SYSTEM
assert "Never write an image link yourself" in P.EDITOR_SYSTEM
print("editor: markers only: OK")

print("test_charts: OK")

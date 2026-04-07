"""
Output manager for delv-e.

Handles all content rendering and file writing:
- Terminal display (print_wrapper, display_*)
- Analysis markdown (write_analysis_md)
- Iteration summaries (write_iteration_summary)
- Final outputs: research model, synthesis report, cost summary (write_final_outputs)
- Synthesis HTML rendering with chart embedding and analysis hyperlinks
"""

import glob
import os
import re
import sys

import style
from logger_config import get_logger
logger = get_logger(__name__)


class OutputManager:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.silent_mode = False
        self._captured_output = []

    # ──────────────────────────────────────────────
    # Terminal display
    # ──────────────────────────────────────────────

    def print_wrapper(self, message, end="\n", flush=False, chain_id=None, thought=False):
        message = str(message)
        if self.silent_mode:
            self._captured_output.append(message)
            if end:
                self._captured_output.append(end)
            return
        print(message, end=end, flush=flush)

    def display_system_messages(self, message, chain_id=None):
        if self.silent_mode:
            return
        print(f"  {style.DIM}{message}{style.RESET}")

    def display_tool_start(self, agent_name, model, chain_id=None):
        if self.silent_mode:
            return
        print(style.agent(agent_name, model))

    def display_results(self, chain_id=None, **kwargs):
        if self.silent_mode:
            return
        if 'answer' in kwargs and kwargs['answer']:
            print(kwargs['answer'])

    def display_error(self, error, chain_id=None):
        if self.silent_mode:
            return
        print(style.error_msg(str(error)[:200]), file=sys.stderr)

    def display_tool_info(self, action, action_input, chain_id=None):
        if self.silent_mode:
            return

    def set_silent(self, silent):
        self.silent_mode = silent

    # ──────────────────────────────────────────────
    # Analysis file writing
    # ──────────────────────────────────────────────

    def write_analysis_md(self, analysis_dir, question, code, results, error, plots,
                          iteration=0, max_iterations=0, chain_id=0):
        """Write analysis.md with code, output, and embedded plots."""
        md_parts = [f"# {question}\n"]
        md_parts.append(
            f"| Field | Value |\n|-------|-------|\n"
            f"| Iteration | {iteration} of {max_iterations} |\n"
            f"| Chain ID | {chain_id} |\n"
        )

        md_parts.append("\n## Code\n")
        md_parts.append(f"```python\n{code}\n```\n")

        md_parts.append("\n## Output\n")
        if results:
            md_parts.append(f"```\n{results}\n```\n")
        elif error:
            md_parts.append(f"```\nEXECUTION ERROR:\n{error}\n```\n")
        else:
            md_parts.append("```\nNo output produced.\n```\n")

        if plots:
            md_parts.append("\n## Plots\n")
            for plot_path in plots:
                fname = os.path.basename(plot_path)
                md_parts.append(f"![{fname}]({fname})\n")

        md_path = os.path.join(analysis_dir, "analysis.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_parts))

    def write_iteration_summary(self, iteration, solutions_data, scores,
                                 selected_index, model_impact, contradiction,
                                 arc_exhausted,
                                 new_questions=None, selected_questions=None):
        """Write _summary.md for an iteration. Called from auto_explore.run()."""
        iter_dir = f"{iteration:02d}"
        summary_dir = os.path.join(self.output_dir, "exploration", iter_dir)
        os.makedirs(summary_dir, exist_ok=True)

        parts = [f"# Iteration {iteration}\n"]

        parts.append("## Solutions Evaluated\n")
        parts.append("| # | Chain ID | Question | Score |")
        parts.append("|---|----------|----------|-------|")
        for i, sol in enumerate(solutions_data):
            s = scores[i] if i < len(scores) else "?"
            marker = " ✓" if i == selected_index else ""
            cid = sol.get('chain_id', '?')
            parts.append(f"| {i+1}{marker} | [{cid}]({cid}/analysis.md) | {sol['question'][:80]} | {s}/10 |")
        parts.append("")

        parts.append("## Research Model Update\n")
        parts.append(f"- **Model Impact:** {model_impact}")
        parts.append(f"- **Contradiction:** {'Yes' if contradiction else 'No'}")
        parts.append(f"- **Arc Exhausted:** {'Yes' if arc_exhausted else 'No'}")
        parts.append("")

        if new_questions:
            parts.append("## Questions Generated\n")
            for i, q in enumerate(new_questions, 1):
                parts.append(f"{i}. {q[:120]}")
            parts.append("")

        if selected_questions:
            parts.append("**Selected for next iteration:** " + ", ".join(
                q[:60] + "..." for q in selected_questions
            ))

        path = os.path.join(summary_dir, "_summary.md")
        with open(path, "w") as f:
            f.write("\n".join(parts))

    # ──────────────────────────────────────────────
    # Final output writing
    # ──────────────────────────────────────────────

    def write_final_outputs(self, research_model, synthesis_text=None,
                            cost_tracker=None, run_logger=None):
        """Write final research model, synthesis report, and cost summary."""
        model_path = os.path.join(self.output_dir, "research_model.md")
        with open(model_path, "w") as f:
            f.write("# Final Research Model\n\n")
            f.write(research_model or "(empty)")

        if synthesis_text:
            synth_path = os.path.join(self.output_dir, "synthesis_report.md")
            with open(synth_path, "w") as f:
                f.write(synthesis_text)

            try:
                self._write_synthesis_html(synthesis_text)
            except Exception as e:
                logger.warning(f"Synthesis HTML generation failed: {e}")

        if cost_tracker:
            cost_path = os.path.join(self.output_dir, "cost.txt")
            with open(cost_path, "w") as f:
                f.write(cost_tracker.report() + "\n")
                if run_logger:
                    agent_summary = run_logger.summary()
                    if agent_summary:
                        f.write("\n" + agent_summary + "\n")

    # ──────────────────────────────────────────────
    # Synthesis HTML rendering
    # ──────────────────────────────────────────────

    def _write_synthesis_html(self, synthesis_text):
        """Render synthesis markdown as a styled, self-contained HTML report."""
        cited_ids = set(re.findall(r'\[\[(\d+)\]\]', synthesis_text))
        analysis_links = {}

        for cid in cited_ids:
            pattern = os.path.join(self.output_dir, "exploration", "*", str(cid), "analysis.md")
            matches = glob.glob(pattern)
            if not matches:
                continue
            md_path = matches[0]
            analysis_dir = os.path.dirname(md_path)
            rel_dir = os.path.relpath(analysis_dir, self.output_dir)
            html_rel = os.path.join(rel_dir, "analysis.html")
            analysis_links[cid] = html_rel

            try:
                with open(md_path) as f:
                    analysis_md = f.read()
                analysis_html = self._md_to_html(analysis_md, is_analysis=True)

                img_html = ""
                for ext in ('*.png', '*.jpg', '*.svg'):
                    for img_path in glob.glob(os.path.join(analysis_dir, ext)):
                        fname = os.path.basename(img_path)
                        img_html += f'<div class="plot"><img src="{fname}" alt="{fname}"></div>\n'

                wrapper = self._analysis_html_template(analysis_html, img_html, cid)
                html_path = os.path.join(analysis_dir, "analysis.html")
                with open(html_path, 'w') as f:
                    f.write(wrapper)
            except Exception as e:
                logger.debug(f"Analysis HTML failed for {cid}: {e}")

        html_body = synthesis_text
        for cid, href in analysis_links.items():
            html_body = html_body.replace(
                f'[[{cid}]]',
                f'<a href="{href}" class="cite-link" title="View analysis {cid}">[{cid[-4:]}]</a>'
            )
        html_body = re.sub(r'\[\[(\d+)\]\]', r'<span class="cite-dead">[\1]</span>', html_body)

        html_body = self._md_to_html(html_body)

        html_body = re.sub(
            r'<img src="(synthesis_charts/[^"]+)" alt="([^"]*)"',
            r'<img src="\1" alt="\2" class="chart-img"',
            html_body
        )

        full_html = self._synthesis_html_template(html_body)
        out_path = os.path.join(self.output_dir, "synthesis_report.html")
        with open(out_path, 'w') as f:
            f.write(full_html)
        logger.info(f"Synthesis HTML written to {out_path}")

    # ──────────────────────────────────────────────
    # Markdown → HTML converter
    # ──────────────────────────────────────────────

    @staticmethod
    def _md_to_html(md_text, is_analysis=False):
        """Lightweight markdown to HTML conversion. No external dependencies."""
        lines = md_text.split('\n')
        html_parts = []
        in_code = False
        in_table = False
        in_list = False
        paragraph_lines = []

        def flush_paragraph():
            if paragraph_lines:
                text = ' '.join(paragraph_lines)
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
                html_parts.append(f'<p>{text}</p>')
                paragraph_lines.clear()

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('```'):
                if in_code:
                    html_parts.append('</code></pre>')
                    in_code = False
                else:
                    flush_paragraph()
                    lang = stripped[3:].strip()
                    html_parts.append(f'<pre><code class="lang-{lang}">')
                    in_code = True
                continue
            if in_code:
                html_parts.append(line.replace('<', '&lt;').replace('>', '&gt;'))
                continue

            if stripped == '---':
                flush_paragraph()
                html_parts.append('<hr>')
                continue

            header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if header_match:
                flush_paragraph()
                level = len(header_match.group(1))
                text = header_match.group(2)
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                tag = f'h{level}'
                html_parts.append(f'<{tag}>{text}</{tag}>')
                continue

            img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
            if img_match:
                flush_paragraph()
                alt, src = img_match.groups()
                html_parts.append(f'<div class="chart-container"><img src="{src}" alt="{alt}"></div>')
                continue

            if '|' in stripped and stripped.startswith('|'):
                if not in_table:
                    flush_paragraph()
                    html_parts.append('<table>')
                    in_table = True
                if re.match(r'^\|[\s\-:|]+\|$', stripped):
                    continue
                cells = [c.strip() for c in stripped.strip('|').split('|')]
                if html_parts[-1] == '<table>':
                    tag = 'th'
                else:
                    tag = 'td'
                styled_cells = [re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', c) for c in cells]
                row = ''.join(f'<{tag}>{c}</{tag}>' for c in styled_cells)
                html_parts.append(f'<tr>{row}</tr>')
                continue
            elif in_table:
                html_parts.append('</table>')
                in_table = False

            if stripped.startswith('- ') or stripped.startswith('* '):
                if not in_list:
                    flush_paragraph()
                    html_parts.append('<ul>')
                    in_list = True
                item = stripped[2:]
                item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
                item = re.sub(r'\*(.+?)\*', r'<em>\1</em>', item)
                html_parts.append(f'<li>{item}</li>')
                continue
            elif in_list and stripped:
                html_parts.append('</ul>')
                in_list = False

            if not stripped:
                flush_paragraph()
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                continue

            paragraph_lines.append(stripped)

        flush_paragraph()
        if in_table:
            html_parts.append('</table>')
        if in_list:
            html_parts.append('</ul>')

        return '\n'.join(html_parts)

    # ──────────────────────────────────────────────
    # HTML templates
    # ──────────────────────────────────────────────

    @staticmethod
    def _synthesis_html_template(body_html):
        """Full HTML page template for the synthesis report."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>delv-e Synthesis Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');
  :root {{
    --bg: #FAFAF8; --bg2: #F1F0EC; --fg: #1A1A18; --fg2: #6B6A66; --fg3: #9C9B96;
    --border: #E0DFDB; --blue: #2563EB; --green: #16A34A; --red: #DC2626;
    --font: 'IBM Plex Sans', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #161615; --bg2: #1E1E1C; --fg: #E8E7E3; --fg2: #9C9B96; --fg3: #6B6A66;
      --border: #2E2E2B; --blue: #60A5FA; --green: #4ADE80; --red: #F87171;
    }}
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 16px; line-height: 1.7; }}
  .container {{ max-width: 820px; margin: 0 auto; padding: 40px 24px 80px; }}
  .back-link {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--fg3); text-decoration: none; margin-bottom: 24px; font-family: var(--mono); }}
  .back-link:hover {{ color: var(--fg2); }}
  h1 {{ font-size: 28px; font-weight: 600; line-height: 1.3; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; font-weight: 600; margin-top: 48px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 17px; font-weight: 600; margin-top: 32px; margin-bottom: 12px; }}
  p {{ margin-bottom: 16px; color: var(--fg); }}
  strong {{ font-weight: 600; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 40px 0; }}
  ul {{ margin: 0 0 16px 24px; }}
  li {{ margin-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 14px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ font-weight: 600; font-size: 13px; color: var(--fg2); text-transform: uppercase; letter-spacing: 0.3px; }}
  pre {{ background: var(--bg2); border-radius: 8px; padding: 16px; overflow-x: auto; margin: 16px 0; font-size: 13px; }}
  code {{ font-family: var(--mono); font-size: 13px; }}
  .chart-container {{ margin: 24px 0; text-align: center; }}
  .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .cite-link {{ color: var(--blue); text-decoration: none; font-family: var(--mono); font-size: 13px; font-weight: 500; }}
  .cite-link:hover {{ text-decoration: underline; }}
  .cite-dead {{ color: var(--fg3); font-family: var(--mono); font-size: 13px; }}
  .footer {{ text-align: center; margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 12px; color: var(--fg3); }}
  .pdf-btn {{ display: inline-flex; align-items: center; gap: 5px; font-size: 13px; font-weight: 500; padding: 8px 20px; border-radius: 20px; background: var(--bg2); color: var(--fg2); border: 1px solid var(--border); cursor: pointer; font-family: var(--mono); transition: opacity 0.15s; }}
  .pdf-btn:hover {{ opacity: 0.7; }}
  .pdf-wrap {{ text-align: center; margin-top: 48px; }}
  @media print {{
    body {{ background: white; color: #1A1A18; font-size: 11pt; }}
    .container {{ max-width: 100%; padding: 0; }}
    .back-link, .pdf-wrap, .footer {{ display: none; }}
    a.cite-link {{ color: #2563EB; text-decoration: none; }}
    h1 {{ font-size: 22pt; }}
    h2 {{ font-size: 16pt; page-break-after: avoid; margin-top: 28pt; }}
    h3 {{ font-size: 13pt; page-break-after: avoid; }}
    p, li {{ orphans: 3; widows: 3; }}
    table {{ page-break-inside: avoid; }}
    .chart-container {{ page-break-inside: avoid; }}
    .chart-container img {{ max-width: 100%; box-shadow: none; border-radius: 0; }}
    pre {{ page-break-inside: avoid; border: 1px solid #E0DFDB; }}
    hr {{ border-top: 1px solid #CCC; }}
  }}
</style>
</head>
<body>
<div class="container">
  <a href="dashboard.html" class="back-link">← Dashboard</a>
  {body_html}
  <div class="pdf-wrap"><button class="pdf-btn" onclick="window.print()">⬇ Export PDF</button></div>
  <div class="footer">Generated by delv-e · Deep Exploratory Learning &amp; Visualization Engine</div>
</div>
</body>
</html>'''

    @staticmethod
    def _analysis_html_template(body_html, img_html, chain_id):
        """HTML wrapper for individual analysis pages."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analysis {chain_id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');
  :root {{
    --bg: #FAFAF8; --bg2: #F1F0EC; --fg: #1A1A18; --fg2: #6B6A66; --fg3: #9C9B96;
    --border: #E0DFDB;
    --font: 'IBM Plex Sans', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #161615; --bg2: #1E1E1C; --fg: #E8E7E3; --fg2: #9C9B96; --fg3: #6B6A66;
      --border: #2E2E2B;
    }}
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 15px; line-height: 1.6; }}
  .container {{ max-width: 820px; margin: 0 auto; padding: 40px 24px 80px; }}
  .back-link {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--fg3); text-decoration: none; margin-bottom: 24px; font-family: var(--mono); }}
  .back-link:hover {{ color: var(--fg2); }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 16px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin-top: 32px; margin-bottom: 12px; }}
  p {{ margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ font-weight: 600; color: var(--fg2); }}
  pre {{ background: var(--bg2); border-radius: 8px; padding: 16px; overflow-x: auto; margin: 16px 0; font-size: 13px; }}
  code {{ font-family: var(--mono); font-size: 13px; }}
  .plot {{ margin: 24px 0; text-align: center; }}
  .plot img {{ max-width: 100%; height: auto; border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">
  <a href="../../../synthesis_report.html" class="back-link">← Back to Report</a>
  {body_html}
  {img_html}
</div>
</body>
</html>'''
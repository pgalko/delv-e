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
import threading

import style
from logger_config import get_logger
logger = get_logger(__name__)


# Per-thread output buffer. When a thread calls OutputManager.begin_buffer(),
# its `print_wrapper` writes accumulate here instead of going to stdout. The
# main thread later retrieves the accumulated text via end_buffer() and flushes
# the buffers in deterministic order. Used by the parallel question loop in
# auto_explore.AutoExplorer.run to keep N concurrent workers' terminal output
# from interleaving on stdout.
#
# Distinct from silent_mode (a per-OutputManager flag used elsewhere to mute
# LLM token streaming during cheap-agent calls) — silent_mode is global to
# the manager, this is per-thread. The two mechanisms coexist: silent_mode
# wins if both are active, since silent_mode means "don't show this at all"
# whereas buffering means "show this later, in order".
_per_thread_buffer = threading.local()


def _active_buffer():
    """Return the calling thread's active output buffer (list), or None."""
    return getattr(_per_thread_buffer, 'buffer', None)


class OutputManager:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.silent_mode = False
        self._captured_output = []

    # ──────────────────────────────────────────────
    # Per-thread output buffering (parallel question processing)
    # ──────────────────────────────────────────────

    def begin_buffer(self):
        """Activate per-thread output buffering for the calling thread.

        After this call, every print_wrapper invocation on this thread writes
        to a thread-local list instead of stdout. The accumulated output is
        retrieved via end_buffer(). Also sets style quiet mode on the thread,
        so style.spinner becomes a no-op and avoids fighting other workers
        over terminal cursor positioning.

        Idempotent in practice — calling twice on the same thread without an
        intervening end_buffer simply re-initialises the buffer (any
        previously-buffered content is dropped). The two-step
        begin_buffer / end_buffer pattern is intended to bracket exactly one
        unit of work.
        """
        _per_thread_buffer.buffer = []
        style.set_quiet_mode(True)

    def end_buffer(self):
        """Deactivate buffering on the calling thread and return what was buffered.

        Returns the accumulated output as a single string. After this call,
        print_wrapper resumes writing to stdout for this thread. Always
        pair with a preceding begin_buffer() — calling end_buffer() without
        an active buffer returns the empty string.
        """
        buf = getattr(_per_thread_buffer, 'buffer', None)
        _per_thread_buffer.buffer = None
        style.set_quiet_mode(False)
        return ''.join(buf) if buf else ''

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
        # Per-thread buffering: parallel workers route their output here.
        # The main thread flushes the buffers in deterministic order after
        # all workers complete.
        buf = _active_buffer()
        if buf is not None:
            buf.append(message)
            if end:
                buf.append(end)
            return
        print(message, end=end, flush=flush)

    def display_system_messages(self, message, chain_id=None):
        if self.silent_mode:
            return
        line = f"  {style.DIM}{message}{style.RESET}"
        buf = _active_buffer()
        if buf is not None:
            buf.append(line)
            buf.append("\n")
            return
        print(line)

    def display_tool_start(self, agent_name, model, chain_id=None):
        if self.silent_mode:
            return
        line = style.agent(agent_name, model)
        buf = _active_buffer()
        if buf is not None:
            buf.append(line)
            buf.append("\n")
            return
        print(line)

    def display_results(self, chain_id=None, **kwargs):
        if self.silent_mode:
            return
        if 'answer' in kwargs and kwargs['answer']:
            buf = _active_buffer()
            if buf is not None:
                buf.append(str(kwargs['answer']))
                buf.append("\n")
                return
            print(kwargs['answer'])

    def display_error(self, error, chain_id=None):
        if self.silent_mode:
            return
        line = style.error_msg(str(error)[:200])
        # In buffered mode the error message belongs inside this question's
        # output block (so the user sees it next to the question that
        # produced it, when the buffer flushes). Outside buffered mode it
        # still goes to stderr as before.
        buf = _active_buffer()
        if buf is not None:
            buf.append(line)
            buf.append("\n")
            return
        print(line, file=sys.stderr)

    def display_tool_info(self, action, action_input, chain_id=None):
        if self.silent_mode:
            return

    def get_captured_output(self):
        """Return captured output from silent mode and clear the buffer."""
        result = ''.join(self._captured_output)
        self._captured_output.clear()
        return result

    def set_silent(self, silent):
        if silent:
            self._captured_output.clear()
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

    def write_final_outputs(self, research_model, briefing_text=None,
                            insight_tree=None, cost_tracker=None,
                            run_logger=None,
                            # Backward-compat for any caller still passing
                            # the old kwarg name:
                            synthesis_text=None):
        """Write final research model, briefing, and supporting artefacts.

        Writes these artefacts to the output directory:
          - research_model.md
          - briefing.md + briefing.html        (replaces synthesis_report.*)
          - findings_index.md + findings_index.html  (MD written to disk
              by AutoExplorer before this call; HTML rendered here)
          - structural_map.md + structural_map.html  (same — MD on disk
              already, HTML rendered here)
          - cost.txt

        Each HTML artefact has its [[chain_id]] citations rewritten as
        clickable links to per-analysis HTML pages, which are generated
        once up front (shared across all three artefacts).
        """
        # Honour backward-compat alias
        if briefing_text is None and synthesis_text is not None:
            briefing_text = synthesis_text

        # 1. Research model (unchanged)
        model_path = os.path.join(self.output_dir, "research_model.md")
        with open(model_path, "w") as f:
            f.write("# Final Research Model\n\n")
            f.write(research_model or "(empty)")

        # 2. Briefing + companion artefacts
        if briefing_text:
            briefing_md_path = os.path.join(self.output_dir, "briefing.md")
            with open(briefing_md_path, "w") as f:
                f.write(briefing_text)

            # HTML rendering: briefing + the two companion artefacts.
            # The companion MDs were written by AutoExplorer
            # (_write_briefing_artefacts). We read them here for HTML.
            try:
                self._write_briefing_html_artefacts(briefing_text)
            except Exception as e:
                logger.warning(f"Briefing HTML generation failed: {e}")

        # 3. Cost summary (unchanged)
        if cost_tracker:
            cost_path = os.path.join(self.output_dir, "cost.txt")
            with open(cost_path, "w") as f:
                f.write(cost_tracker.report() + "\n")
                if run_logger:
                    agent_summary = run_logger.summary()
                    if agent_summary:
                        f.write("\n" + agent_summary + "\n")

    # ──────────────────────────────────────────────
    # Briefing + companion artefacts HTML rendering
    # ──────────────────────────────────────────────

    def _write_briefing_html_artefacts(self, briefing_text):
        """Render briefing + companion artefacts as clickable HTML.

        Produces three HTML files:
          - briefing.html         (top nav links to dashboard + companions)
          - findings_index.html   (nav back to briefing)
          - structural_map.html   (nav back to briefing)

        Per-analysis HTML pages are generated once from the union of all
        [[chain_id]] citations found across the three artefact MDs, so they
        are shared and rendered only once regardless of how many artefacts
        cite the same analysis.
        """
        # 1. Gather all chain_ids cited across all three artefacts so the
        #    per-analysis pages cover every citation in every artefact.
        all_cited_ids = set(re.findall(r'\[\[(\d+)\]\]', briefing_text))

        findings_index_path = os.path.join(self.output_dir, "findings_index.md")
        structural_map_path = os.path.join(self.output_dir, "structural_map.md")

        findings_index_md = ""
        structural_map_md = ""
        if os.path.exists(findings_index_path):
            try:
                with open(findings_index_path) as f:
                    findings_index_md = f.read()
                all_cited_ids.update(re.findall(r'\[\[(\d+)\]\]', findings_index_md))
            except Exception as e:
                logger.debug(f"findings_index.md unreadable: {e}")

        if os.path.exists(structural_map_path):
            try:
                with open(structural_map_path) as f:
                    structural_map_md = f.read()
                all_cited_ids.update(re.findall(r'\[\[(\d+)\]\]', structural_map_md))
            except Exception as e:
                logger.debug(f"structural_map.md unreadable: {e}")

        # 2. Build per-analysis HTML pages once, shared by all artefacts.
        analysis_links = self._build_analysis_pages(all_cited_ids)

        # 3. Render each artefact MD -> HTML with citations rewritten to
        #    clickable links where the analysis page exists.
        self._render_artefact_html(
            md_text=briefing_text,
            title="Investigation Briefing",
            output_filename="briefing.html",
            analysis_links=analysis_links,
            is_briefing=True,
        )

        if findings_index_md:
            self._render_artefact_html(
                md_text=findings_index_md,
                title="Findings Index",
                output_filename="findings_index.html",
                analysis_links=analysis_links,
                is_briefing=False,
            )

        if structural_map_md:
            self._render_artefact_html(
                md_text=structural_map_md,
                title="Structural Landscape",
                output_filename="structural_map.html",
                analysis_links=analysis_links,
                is_briefing=False,
            )

        logger.info(f"Briefing + companion artefacts written to {self.output_dir}")

    def _build_analysis_pages(self, cited_ids):
        """Render per-analysis HTML pages for a set of cited chain_ids.

        For each chain_id that has an analysis.md on disk, render its
        matching analysis.html (with embedded plots) in the same directory.
        Returns a dict {chain_id_str: relative_href} so callers can rewrite
        [[chain_id]] citations into clickable links.

        Analysis HTML pages have a "← Back to Briefing" link at the top.
        """
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

        return analysis_links

    def _render_artefact_html(self, md_text, title, output_filename,
                              analysis_links, is_briefing=False):
        """Render a markdown artefact to styled HTML with clickable citations.

        Args:
            md_text: markdown source text
            title: page title (shown in <title> and header)
            output_filename: destination filename (within self.output_dir)
            analysis_links: dict {chain_id_str: relative_href} for rewriting
            is_briefing: if True, top nav shows companion-artefact links +
                PDF export; if False, top nav links back to the briefing.
        """
        html_body = md_text

        # Rewrite [[chain_id]] citations to clickable links where the
        # analysis page exists; remaining [[N]] become styled dead refs.
        for cid, href in analysis_links.items():
            html_body = html_body.replace(
                f'[[{cid}]]',
                f'<a href="{href}" class="cite-link" title="View analysis {cid}">[{cid[-4:]}]</a>'
            )
        html_body = re.sub(r'\[\[(\d+)\]\]', r'<span class="cite-dead">[\1]</span>', html_body)

        # Convert markdown -> HTML
        html_body = self._md_to_html(html_body)

        # Chart image class (applies to briefing only but harmless elsewhere)
        html_body = re.sub(
            r'<img src="(synthesis_charts/[^"]+)" alt="([^"]*)"',
            r'<img src="\1" alt="\2" class="chart-img"',
            html_body
        )

        full_html = self._artefact_html_template(
            body_html=html_body,
            title=title,
            is_briefing=is_briefing,
        )

        out_path = os.path.join(self.output_dir, output_filename)
        with open(out_path, 'w') as f:
            f.write(full_html)

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
    def _artefact_html_template(body_html, title, is_briefing=False):
        """Full HTML page template for briefing + companion artefacts.

        One template serves all three artefacts (briefing, findings index,
        structural map). The small amount of branching is in the <header>
        navigation:
          - Briefing page: links to dashboard, the two companion artefacts,
            and PDF export
          - Companion pages: link back to the briefing
        The CSS block is identical across all three.
        """
        if is_briefing:
            nav_html = (
                '  <a href="dashboard.html" class="back-link">← Dashboard</a>\n'
                '  <span class="nav-sep">·</span>\n'
                '  <a href="findings_index.html" class="back-link">Findings Index</a>\n'
                '  <span class="nav-sep">·</span>\n'
                '  <a href="structural_map.html" class="back-link">Structural Map</a>'
            )
            footer_pdf = (
                '  <div class="pdf-wrap"><button class="pdf-btn" onclick="window.print()">'
                '⬇ Export PDF</button></div>'
            )
        else:
            nav_html = '  <a href="briefing.html" class="back-link">← Back to Briefing</a>'
            footer_pdf = ''

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>delv-e — {title}</title>
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
  .nav-bar {{ display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin-bottom: 24px; font-family: var(--mono); font-size: 13px; }}
  .back-link {{ display: inline-flex; align-items: center; gap: 6px; color: var(--fg3); text-decoration: none; }}
  .back-link:hover {{ color: var(--fg2); }}
  .nav-sep {{ color: var(--fg3); }}
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
    .nav-bar, .pdf-wrap, .footer {{ display: none; }}
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
  <div class="nav-bar">
{nav_html}
  </div>
  {body_html}
{footer_pdf}
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
  <a href="../../../briefing.html" class="back-link">← Back to Briefing</a>
  {body_html}
  {img_html}
</div>
</body>
</html>'''
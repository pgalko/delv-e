"""
Dashboard generator for delv-e.

Writes a self-contained HTML dashboard to output/dashboard.html.
Auto-refreshes in the browser every 30 seconds via meta tag.
Called after each iteration checkpoint — zero dependencies beyond stdlib.
"""

import json
import os
import time
import re

from logger_config import get_logger
logger = get_logger(__name__)


def write_dashboard(output_dir, explorer, engine, iteration, max_iterations):
    """Write dashboard.html from live exploration state.

    Args:
        output_dir: path to output directory
        explorer: AutoExplorer instance
        engine: ExplorationEngine instance
        iteration: current iteration number
        max_iterations: total iterations planned
    """
    try:
        data = _extract_data(explorer, engine, iteration, max_iterations, output_dir)
        html = _render_html(data)
        path = os.path.join(output_dir, "dashboard.html")
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            f.write(html)
        os.replace(tmp, path)
    except Exception as e:
        import traceback
        logger.warning(
            f"Dashboard write failed: {e}\n"
            f"{traceback.format_exc()}"
        )


def _extract_data(explorer, engine, iteration, max_iterations, output_dir):
    """Extract all dashboard data from explorer and engine state."""

    # --- Iteration data from insight tree ---
    nodes = []
    for nid, node in explorer.insight_tree.items():
        parts = nid.split('_')
        counter = int(parts[-1])
        nodes.append({
            'counter': counter,
            'score': node.get('quality_score', 0),
            'status': node.get('status', ''),
            'summary': node.get('finding_summary', ''),
            'iteration': node.get('iteration_added'),
        })
    nodes.sort(key=lambda x: x['counter'])

    # Group by actual iteration if available, otherwise fall back to pairs
    from collections import defaultdict
    has_iteration_data = any(n['iteration'] is not None for n in nodes)

    iter_data = []
    if has_iteration_data:
        iter_groups = defaultdict(list)
        for n in nodes:
            iter_groups[n['iteration'] or 0].append(n)
        for it_num in sorted(iter_groups.keys()):
            group = iter_groups[it_num]
            winner = max(group, key=lambda x: x['score'])
            loser = min(group, key=lambda x: x['score']) if len(group) > 1 else winner
            iter_data.append({
                'winner_score': winner['score'],
                'loser_score': loser['score'],
                'summary': winner['summary'][:140],
            })
    else:
        # Fallback: pair by counter order
        num_parallel = 2
        for i in range(0, len(nodes), num_parallel):
            pair = nodes[i:i + num_parallel]
            winner = max(pair, key=lambda x: x['score'])
            loser = min(pair, key=lambda x: x['score']) if len(pair) > 1 else winner
            iter_data.append({
                'winner_score': winner['score'],
                'loser_score': loser['score'],
                'summary': winner['summary'][:140],
            })

    winner_scores = [d['winner_score'] for d in iter_data]
    loser_scores = [d['loser_score'] for d in iter_data]

    # --- Commitment map (from strategic review history) ---
    commitment_at = {}
    commitment_history = getattr(explorer, 'commitment_history', [])
    for it, action in commitment_history:
        commitment_at[it + 1] = action  # commitment at iteration N governs iteration N+1

    # Hold bands (contiguous HOLD sequences — analogous to old PURSUING bands)
    hold_bands = []
    band_start = None
    for i in range(1, iteration + 1):
        if commitment_at.get(i) == 'HOLD' and band_start is None:
            band_start = i
        if commitment_at.get(i) != 'HOLD' and band_start is not None:
            hold_bands.append([band_start, i - 1])
            band_start = None
    if band_start is not None:
        hold_bands.append([band_start, iteration])

    # --- Pivot/abandon iterations (strategic inflection points) ---
    pivot_iters = [it + 1 for it, action in commitment_history if action in ('PIVOT', 'ABANDON')]

    # --- Parse research model ---
    rm = explorer.research_model or ''
    established = _parse_section(rm, '## Established Findings')
    health_lines = _parse_section(rm, '## Exploration Health')
    landscape_lines = _parse_section(rm, '## Structural Landscape')

    # Parse health metrics (new minimal shape: Breadth + Unexplored)
    breadth = 'UNKNOWN'
    for line in health_lines:
        if 'Breadth:' in line:
            breadth = line.split('Breadth:')[-1].strip()
    # topics is legacy but kept in output dict for template compatibility
    topics = len(established)

    # STATUS tag counts from Established Findings
    status_counts = {'ESTABLISHED': 0, 'PROVISIONAL': 0, 'SHRINKS': 0, 'CONTRADICTED': 0}
    for line in established:
        for tag in status_counts:
            if f'[{tag}]' in line:
                status_counts[tag] += 1
                break

    # Structural landscape summary — compact preview for the dashboard card
    if landscape_lines and not any('<<< DO NOT MODIFY' in l for l in landscape_lines):
        landscape_preview = ' '.join(landscape_lines)[:200]
    else:
        landscape_preview = 'Not yet populated'

    # --- Agent call counts ---
    agent_counts = {}
    if hasattr(engine, 'run_logger') and engine.run_logger:
        for entry in engine.run_logger.entries:
            a = entry.get('agent', 'unknown')
            agent_counts[a] = agent_counts.get(a, 0) + 1

    # --- Cost ---
    total_cost = 0.0
    total_calls = 0
    if hasattr(engine, 'cost_tracker'):
        total_cost = engine.cost_tracker.total_cost
        total_calls = engine.cost_tracker.calls

    # --- Models ---
    agent_model = engine.models.agent_model
    code_model = engine.models.code_model
    premium_model = getattr(explorer, 'premium_model', None)

    # --- Dataset ---
    if engine.df is not None:
        dataset_shape = f"{engine.df.shape[0]:,} rows x {engine.df.shape[1]} cols"
    else:
        dataset_shape = "computation mode"

    # --- Mean score ---
    mean_score = round(sum(winner_scores) / max(len(winner_scores), 1), 1)
    significant = sum(1 for s in winner_scores if s >= 7)

    # --- Status ---
    is_complete = iteration >= max_iterations

    # --- Heatmap: derive arc spans from arc_history ---
    arc_history = getattr(explorer, 'arc_history', [])
    heatmap_arcs = []
    for idx, (start_iter, label) in enumerate(arc_history):
        end_iter = arc_history[idx + 1][0] - 1 if idx + 1 < len(arc_history) else iteration
        iters = list(range(start_iter, end_iter + 1))
        if iters:
            heatmap_arcs.append({'name': label, 'iters': iters})

    # Probe iterations
    probe_iters = {pit: pr[:80] for pit, pr in getattr(explorer, 'probe_history', [])}

    # Rotation iterations
    rot_iters = {it for it, _, _ in getattr(explorer, 'rotation_history', [])}

    # --- Run geometry (embedding-based observability) ---
    # Compute on the fly from active-node embeddings. None entries propagate
    # as coverage gaps. The panel shows from the first embedded node — Hz
    # and clusters degrade gracefully (display as "—") until enough points
    # accumulate. Hidden entirely when no embeddings exist.
    geometry = None
    try:
        from embeddings import rolling_dispersion, z_score, compute_lineage
        active_nodes_geo = sorted(
            [n for n in explorer.insight_tree.values()
             if n.get('status') == 'active'],
            key=lambda n: n['chain_id'],
        )
        geo_vectors = [n.get('embedding') for n in active_nodes_geo]
        geo_embedded = sum(1 for v in geo_vectors if v is not None)

        if geo_embedded >= 1:
            # Hz needs ≥2 points in the window to produce values; will be
            # all-None for a single-embedding trajectory.
            H = rolling_dispersion(geo_vectors, window=5)
            H_z = z_score(H)
            # Conceptual lineage: each iter's nearest prior in embedding space.
            # Deterministic; no clustering algorithm. depths is the same length
            # as geo_vectors with None for unembedded.
            parents, par_dists, depths = compute_lineage(geo_vectors)
            geo_iters = [
                n.get('iteration_added', i + 1)
                for i, n in enumerate(active_nodes_geo)
            ]
            geo_scores = [n.get('quality_score', 0) for n in active_nodes_geo]

            # Current Hz (last embedded node) + trend (last 3 vs prior 3)
            cur_hz = next((z for z in reversed(H_z) if z is not None), None)
            valid_zs = [z for z in H_z if z is not None]
            trend = 0.0
            if len(valid_zs) >= 6:
                trend = (sum(valid_zs[-3:]) / 3.0
                         - sum(valid_zs[-6:-3]) / 3.0)

            valid_depths = [d for d in depths if d is not None]
            max_depth = max(valid_depths) if valid_depths else 0

            # Truncated finding summaries for tooltip hover
            geo_summaries = [
                (n.get('finding_summary') or n.get('result_digest')
                 or n.get('question') or '').strip()[:200]
                for n in active_nodes_geo
            ]

            geometry = {
                'iters': geo_iters,
                'scores': geo_scores,
                'H_z': H_z,
                'depths': depths,
                'parents': parents,
                'par_dists': par_dists,
                'max_depth': max_depth,
                'embedded': geo_embedded,
                'total': len(active_nodes_geo),
                'current_hz': cur_hz,
                'trend': trend,
                'summaries': geo_summaries,
            }
    except Exception as e:
        logger.debug(f"Geometry computation skipped: {e}")
        geometry = None

    return {
        'iteration': iteration,
        'max_iterations': max_iterations,
        'is_complete': is_complete,
        'winner_scores': winner_scores,
        'loser_scores': loser_scores,
        'iter_data': iter_data,
        'commitment_at': commitment_at,
        'hold_bands': hold_bands,
        'pivot_iters': pivot_iters,
        'established': established,
        'status_counts': status_counts,
        'breadth': breadth,
        'topics': topics,
        'landscape_preview': landscape_preview,
        'n_arcs': len(heatmap_arcs),
        'agent_counts': agent_counts,
        'total_cost': total_cost,
        'total_calls': total_calls,
        'agent_model': agent_model,
        'code_model': code_model,
        'premium_model': premium_model,
        'seed_question': explorer.seed_question,
        'dataset_shape': dataset_shape,
        'mean_score': mean_score,
        'significant': significant,
        'probe_history': getattr(explorer, 'probe_history', []),
        'rotation_history': getattr(explorer, 'rotation_history', []),
        'search_history': getattr(explorer, 'search_history', []),
        'heatmap_arcs': heatmap_arcs,
        'probe_iters': probe_iters,
        'rot_iters': rot_iters,
        'output_dir': output_dir,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'geometry': geometry,
    }


def _parse_section(rm, header):
    """Extract bullet lines from a research model section."""
    lines = []
    in_section = False
    for line in rm.split('\n'):
        if header in line:
            in_section = True
            continue
        if line.startswith('## ') and in_section:
            break
        if in_section and line.strip().startswith('- '):
            text = line.strip()[2:]
            # Skip STATUS-only lines from [PUBLISHED] entries
            if text.startswith('STATUS:'):
                continue
            lines.append(text)
        elif in_section and line.strip() and not line.strip().startswith('#'):
            # Skip standalone STATUS lines without bullet prefix too
            if line.strip().startswith('STATUS:'):
                continue
            lines.append(line.strip())
    return lines


def _short_model(m):
    """Shorten model name for display."""
    if not m:
        return ''
    return m.split("-202")[0] if "-202" in m else m


def _escape(text):
    """Escape HTML special characters."""
    return (text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _render_html(d):
    """Render the full dashboard HTML."""

    # Status pill
    if d['is_complete']:
        # Check which artefacts actually exist before linking to them —
        # avoids 404s when e.g. structural_map.md couldn't be extracted.
        output_dir = d.get('output_dir', '')
        artefact_buttons = ['<a href="briefing.html" class="report-btn">View Briefing</a>']
        if output_dir and os.path.exists(os.path.join(output_dir, 'findings_index.html')):
            artefact_buttons.append(
                '<a href="findings_index.html" class="report-btn report-btn-alt">Findings Index</a>'
            )
        if output_dir and os.path.exists(os.path.join(output_dir, 'structural_map.html')):
            artefact_buttons.append(
                '<a href="structural_map.html" class="report-btn report-btn-alt">Structural Map</a>'
            )
        status_html = (
            '<span class="status-pill status-complete">Complete</span> '
            + ' '.join(artefact_buttons)
        )
    else:
        status_html = '<span class="status-pill status-running">Running</span>'

    # Metric cards
    # Status-tag breakdown from Established Findings (replaces old maturity note)
    sc = d.get('status_counts', {})
    total_tagged = sum(sc.values())
    if total_tagged > 0:
        bits = []
        if sc.get('ESTABLISHED'): bits.append(f"{sc['ESTABLISHED']} est.")
        if sc.get('PROVISIONAL'): bits.append(f"{sc['PROVISIONAL']} prov.")
        if sc.get('SHRINKS'): bits.append(f"{sc['SHRINKS']} shrinks")
        if sc.get('CONTRADICTED'): bits.append(f"{sc['CONTRADICTED']} contr.")
        maturity_note = ' · '.join(bits)
    else:
        maturity_note = f'{d["significant"]} scored 7+'

    # Established findings HTML
    findings_html = ''
    for f in d['established']:
        text = _escape(f[:120])
        findings_html += f'''
        <div class="finding-item">
          <span class="finding-badge badge-complete">EST</span>
          <span>{text}</span>
        </div>'''
    if not findings_html:
        findings_html = '<div style="color: var(--fg3); font-size: 13px; padding: 8px 0;">No established findings yet</div>'

    # Recent iterations HTML (last 15, reversed)
    recent_html = ''
    recent_start = max(0, len(d['iter_data']) - 15)
    for idx in range(len(d['iter_data']) - 1, recent_start - 1, -1):
        it = d['iter_data'][idx]
        iter_num = idx + 1
        score = it['winner_score']
        if score >= 7:
            score_class = 'score-high'
        elif score >= 5:
            score_class = 'score-med'
        else:
            score_class = 'score-fail'
        summary = _escape(it['summary'])
        recent_html += f'''
        <div class="iter-row">
          <span class="iter-num">#{iter_num}</span>
          <span class="iter-score {score_class}">{score}</span>
          <span class="iter-summary">{summary}</span>
        </div>'''

    # Agent breakdown HTML
    agent_order = [
        'Code Generator', 'Question Generator', 'Research Interpreter',
        'Result Evaluator', 'Question Selector', 'Error Corrector',
        'Strategic Review', 'Reframing Probe', 'Perspective Rotation',
        'Literature Search', 'Literature Integration',
        'Seed Decomposition', 'Briefing Generator', 'Synthesis Chart'
    ]
    max_calls = max(d['agent_counts'].values()) if d['agent_counts'] else 1
    agent_html = ''
    for agent in agent_order:
        count = d['agent_counts'].get(agent, 0)
        if count == 0:
            continue
        pct = round(count / max_calls * 100)
        label = agent.lower().replace('_', ' ')
        is_premium = agent in ('Strategic Review', 'Reframing Probe', 'Perspective Rotation', 'Seed Decomposition', 'Briefing Generator', 'Synthesis Chart')
        is_search = agent in ('Literature Search', 'Literature Integration')
        bar_color = ('var(--cyan)' if is_search else
                     'var(--purple)' if is_premium else
                     ('var(--amber)' if agent == 'Error Corrector' else 'var(--blue)'))
        tag = (' <span style="font-weight: 400; color: var(--fg3);">search</span>' if is_search else
               ' <span style="font-weight: 400; color: var(--fg3);">premium</span>' if is_premium else '')
        agent_html += f'''
        <div class="health-row">
          <span class="health-label">{label}</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div class="bar-wrap" style="width: 180px;">
              <div class="bar-fill" style="width: {max(pct, 1)}%; background: {bar_color};"></div>
            </div>
            <span class="health-val">{count}{tag}</span>
          </div>
        </div>'''

    # Breadth bar
    breadth_pct = {'HIGH': 92, 'MEDIUM': 55, 'LOW': 20}.get(d['breadth'].upper(), 50)
    breadth_color = {'HIGH': 'bar-green', 'MEDIUM': 'bar-amber', 'LOW': 'bar-red'}.get(d['breadth'].upper(), 'bar-blue')

    # Probe history HTML
    probe_html = ''
    if d['probe_history']:
        probe_items = ''
        for it, result in d['probe_history']:
            is_null = 'null' in result.lower()
            badge = 'badge-dormant' if is_null else 'badge-complete'
            label = 'NULL' if is_null else 'HIT'
            probe_items += f'''
            <div class="finding-item">
              <span class="finding-badge {badge}">{label}</span>
              <span style="color: var(--fg2);">iter {it}</span>
              <span style="margin-left: 4px;">{_escape(result[:120])}</span>
            </div>'''
        probe_html = f'''
  <div style="margin-top: 16px;">
    <div class="section-title">Reframing probes ({len(d['probe_history'])})</div>
    <div class="card">{probe_items}
    </div>
  </div>'''

    # Search history HTML
    search_html = ''
    if d['search_history']:
        search_items = ''
        for entry in d['search_history']:
            if len(entry) == 3:
                it, query, summary = entry
            else:
                it, query = entry[0], entry[1] if len(entry) > 1 else ''
                summary = ''
            search_items += f'''
            <div class="finding-item">
              <span class="finding-badge badge-quantified">LIT</span>
              <span style="color: var(--fg2);">iter {it}</span>
              <span style="margin-left: 4px;" title="{_escape(query)}">{_escape(summary[:120] or query[:120])}</span>
            </div>'''
        search_html = f'''
  <div style="margin-top: 16px;">
    <div class="section-title">Literature searches ({len(d['search_history'])})</div>
    <div class="card">{search_items}
    </div>
  </div>'''

    # Rotation history HTML
    rotation_html = ''
    if d['rotation_history']:
        rotation_items = ''
        for it, parent, perspectives in d['rotation_history']:
            parent_short = _escape(parent[:50])
            persp_names = ', '.join(_escape(p['name']) for p in perspectives)
            rotation_items += f'''
            <div class="finding-item" style="flex-direction: column; align-items: flex-start; gap: 4px;">
              <div>
                <span class="finding-badge badge-complete">ROT</span>
                <span style="color: var(--fg2);">iter {it}</span>
                <span style="margin-left: 4px; font-weight: 500;">{parent_short}</span>
              </div>
              <div style="margin-left: 28px; font-size: 12px; color: var(--fg2);">'''
            for p in perspectives:
                rotation_items += f'''
                <div style="margin-top: 2px;">&#8627; <strong>{_escape(p["name"])}</strong>: {_escape(p.get("question","")[:100])}</div>'''
            rotation_items += '''
              </div>
            </div>'''
        rotation_html = f'''
  <div style="margin-top: 16px;">
    <div class="section-title">Perspective rotations ({len(d['rotation_history'])})</div>
    <div class="card">{rotation_items}
    </div>
  </div>'''

    # --- Heatmap HTML ---
    heatmap_html = ''
    cell_h = 18
    if d['heatmap_arcs']:
        max_iter = d['iteration']
        probe_iters = d['probe_iters']
        rot_iters = d['rot_iters']
        winner_scores = d['winner_scores']
        score_colors = {1:'#F09595',2:'#F09595',3:'#F09595',4:'#F09595',
                        5:'#FAC775',6:'#FAC775',
                        7:'#C0DD97',8:'#5DCAA5',9:'#5DCAA5',10:'#5DCAA5'}

        arc_rows = ''
        for arc in d['heatmap_arcs']:
            cells = ''
            for i in range(1, max_iter + 1):
                if i in arc['iters'] and i - 1 < len(winner_scores):
                    sc = winner_scores[i - 1]
                    fill = score_colors.get(sc, '#C0DD97')
                    dot = ''
                    if i in probe_iters:
                        dot = '<div class="hm-ldot" style="background:#D85A30;"></div>'
                    elif i in rot_iters:
                        dot = '<div class="hm-ldot" style="background:#378ADD;"></div>'
                    tip = f'Iter {i} &middot; {_escape(arc["name"])} &middot; Score {sc}'
                    cells += (f'<div class="hm-c" style="background:{fill};" '
                              f'data-hmtip="{tip}">{dot}</div>')
                else:
                    cells += '<div class="hm-c hm-empty"></div>'
            arc_rows += f'<div class="hm-row"><div class="hm-label" title="{_escape(arc["name"])}">{_escape(arc["name"])}</div><div class="hm-cells">{cells}</div></div>\n'

        num_step = 5 if max_iter <= 50 else 10
        iter_nums = ''.join(f'<div class="hm-n">{i if i % num_step == 0 else ""}</div>' for i in range(1, max_iter + 1))

        heatmap_html = f'''
  <div class="chart-section">
    <div class="section-title">Exploration trajectory</div>
    <div class="card" style="background: var(--bg2); border-radius: var(--radius-lg); padding: 12px 16px;">
      {arc_rows}
      <div class="hm-nums">{iter_nums}</div>
      <div class="hm-legend">
        <span><span class="hm-lsw" style="background:#F09595;"></span>1-4</span>
        <span><span class="hm-lsw" style="background:#FAC775;"></span>5-6</span>
        <span><span class="hm-lsw" style="background:#C0DD97;"></span>7</span>
        <span><span class="hm-lsw" style="background:#5DCAA5;"></span>8-10</span>
        <span style="margin-left:8px;"><span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#D85A30;vertical-align:middle;margin-right:4px;"></span>probe</span>
        <span><span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#378ADD;vertical-align:middle;margin-right:4px;"></span>rotation</span>
      </div>
    </div>
  </div>
<div id="hmTip" class="hm-tip"></div>'''

    # Premium model display
    premium_meta = ''
    if d['premium_model']:
        premium_meta = f'<span><span class="meta-label">Premium</span> <span class="meta-val">{_short_model(d["premium_model"])}</span></span>'

    # --- Run geometry panel ---
    geometry_html = ''
    geo = d.get('geometry')
    if geo:
        # Internal coordinate system stays at 880×60 / 880×12 (wide enough that
        # at typical dashboard widths the SVG renders close to 1:1, keeping
        # score-9 dots nearly circular). width="100%" + preserveAspectRatio=
        # "none" makes the SVG fill the panel column at any viewport.
        sw, sh, pad = 880, 60, 6
        n = len(geo['iters'])
        if n > 1:
            xs = [pad + i * (sw - 2 * pad) / (n - 1) for i in range(n)]
        else:
            xs = [sw / 2]

        def _y_of(z):
            if z is None:
                return None
            z = max(-2.5, min(2.5, z))
            return pad + (sh - 2 * pad) * (1 - (z + 2.5) / 5.0)

        line_pts = []
        for x, z in zip(xs, geo['H_z']):
            yy = _y_of(z)
            if yy is not None:
                line_pts.append(f"{x:.1f},{yy:.1f}")

        dots_svg = ''
        for x, z, sc in zip(xs, geo['H_z'], geo['scores']):
            yy = _y_of(z)
            if yy is None or sc < 9:
                continue
            dots_svg += (
                f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="3" '
                f'fill="#E8C547" stroke="#1a1a1a" stroke-width="0.5"/>'
            )

        baseline_y = _y_of(0)
        sparkline_svg = (
            f'<svg width="100%" height="{sh}" viewBox="0 0 {sw} {sh}" '
            f'preserveAspectRatio="none" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
            f'<line x1="{pad}" y1="{baseline_y:.1f}" '
            f'x2="{sw - pad}" y2="{baseline_y:.1f}" '
            f'stroke="var(--border)" stroke-width="1" stroke-dasharray="2,2"/>'
            f'<polyline points="{" ".join(line_pts)}" fill="none" '
            f'stroke="#D87C5A" stroke-width="1.8"/>'
            f'{dots_svg}</svg>'
        )

        # Depth-encoded strip cells. Each cell colored by its lineage depth
        # — pale = shallow (near iter 1), dark = deeply built on prior work.
        # Iter 1 and any unembedded nodes show neutral grey.
        c_low = (228, 233, 240)   # pale slate
        c_high = (28, 52, 89)     # deep navy
        max_depth = max(geo['max_depth'], 1)
        strip_cells = ''
        for i, depth in enumerate(geo['depths']):
            if depth is None:
                color = '#cccccc'
                depth_str = '—'
            else:
                t = depth / max_depth
                rgb = tuple(int(c_low[k] + t * (c_high[k] - c_low[k]))
                            for k in range(3))
                color = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
                depth_str = str(depth)
            it_num = geo['iters'][i] if i < len(geo['iters']) else (i + 1)
            sc = geo['scores'][i] if i < len(geo['scores']) else 0
            hz = geo['H_z'][i] if i < len(geo['H_z']) else None
            hz_str = f"{hz:+.2f}" if hz is not None else "—"
            parent = geo['parents'][i] if i < len(geo['parents']) else None
            par_d = geo['par_dists'][i] if i < len(geo['par_dists']) else None
            if parent is None:
                parent_str = '— (root)'
            else:
                parent_iter = (geo['iters'][parent]
                               if parent < len(geo['iters']) else parent)
                par_d_str = f"{par_d:.3f}" if par_d is not None else "—"
                parent_str = f"iter {parent_iter} (d={par_d_str})"
            summary = (geo['summaries'][i] if i < len(geo['summaries']) else '') \
                      or '(no summary)'
            tip = (f"Iter {it_num} · score {sc} · depth {depth_str}\n"
                   f"Parent: {parent_str}\n"
                   f"Hz {hz_str} (z-scored)\n\n"
                   f"{_escape(summary)}")
            strip_cells += (
                f'<div class="geo-cell" style="background:{color};" '
                f'data-hmtip="{tip}"></div>'
            )
        strip_svg = (
            f'<div class="geo-strip-cells">{strip_cells}</div>'
        )

        if geo['current_hz'] is None:
            hz_str = '—'
        else:
            hz_str = f"{geo['current_hz']:+.2f}"
        if geo['trend'] > 0.1:
            trend_arrow = '↑'
        elif geo['trend'] < -0.1:
            trend_arrow = '↓'
        else:
            trend_arrow = '→'
        depth_str = (str(geo['max_depth'])
                     if geo['max_depth'] > 0 else '—')

        geometry_html = f'''
  <div class="chart-section">
    <div class="section-title">Run geometry</div>
    <div class="geo-card">
      <div class="geo-stats">
        <div class="geo-stat">
          <div class="geo-label">Dispersion (Hz)</div>
          <div class="geo-value">{hz_str} <span class="geo-trend">{trend_arrow}</span></div>
        </div>
        <div class="geo-stat">
          <div class="geo-label">Coverage</div>
          <div class="geo-value">{geo['embedded']}/{geo['total']}</div>
        </div>
        <div class="geo-stat">
          <div class="geo-label">Max depth</div>
          <div class="geo-value">{depth_str}</div>
        </div>
      </div>
      <div class="geo-spark">
        <div class="geo-sublabel">Hz · window 5 · z-scored<span style="float:right">−2σ ⟷ +2σ</span></div>
        {sparkline_svg}
      </div>
      <div class="geo-strip">
        <div class="geo-sublabel">Trajectory · iter 1 → {n}<span>color = lineage depth</span></div>
        {strip_svg}
      </div>
      <div class="geo-caption">Hz &lt; 0: deepening · Hz &gt; 0: spreading · ● score-9</div>
    </div>
  </div>'''

    # JSON data for charts
    winners_json = json.dumps(d['winner_scores'])
    losers_json = json.dumps(d['loser_scores'])
    hold_json = json.dumps(d['hold_bands'])
    pivot_json = json.dumps(d['pivot_iters'])
    commitments_json = json.dumps(d['commitment_at'])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>delv-e dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --bg: #FAFAF8; --bg2: #F1F0EC; --bg3: #E8E7E3;
    --fg: #1A1A18; --fg2: #6B6A66; --fg3: #9C9B96;
    --border: #E0DFDB; --border2: #CCCBC6;
    --blue: #2563EB; --blue-bg: #EFF6FF; --blue-fg: #1E40AF;
    --green: #16A34A; --green-bg: #F0FDF4; --green-fg: #166534;
    --amber: #D97706; --amber-bg: #FFFBEB; --amber-fg: #92400E;
    --red: #DC2626; --red-bg: #FEF2F2; --red-fg: #991B1B;
    --purple: #7C3AED; --purple-bg: #F5F3FF; --purple-fg: #5B21B6;
    --radius: 8px; --radius-lg: 12px;
    --font: 'IBM Plex Sans', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #161615; --bg2: #1E1E1C; --bg3: #282826;
      --fg: #E8E7E3; --fg2: #9C9B96; --fg3: #6B6A66;
      --border: #2E2E2B; --border2: #3A3A37;
      --blue: #60A5FA; --blue-bg: #1E293B; --blue-fg: #93C5FD;
      --green: #4ADE80; --green-bg: #14231A; --green-fg: #86EFAC;
      --amber: #FBBF24; --amber-bg: #231D0F; --amber-fg: #FCD34D;
      --red: #F87171; --red-bg: #231414; --red-fg: #FCA5A5;
      --purple: #A78BFA; --purple-bg: #1C1730; --purple-fg: #C4B5FD;
    }}
  }}
  body {{ background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 14px; line-height: 1.5; }}
  .header {{ padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
  .logo {{ font-family: var(--mono); font-weight: 500; font-size: 15px; letter-spacing: -0.5px; }}
  .logo span {{ color: var(--fg3); font-weight: 400; }}
  .status-pill {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; padding: 4px 12px; border-radius: 20px; }}
  .status-running {{ background: var(--green-bg); color: var(--green-fg); }}
  .status-running::before {{ content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
  .status-complete {{ background: var(--blue-bg); color: var(--blue-fg); }}
  .report-btn {{ display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 500; padding: 4px 14px; border-radius: 20px; background: var(--green-bg); color: var(--green-fg); text-decoration: none; margin-left: 8px; transition: opacity 0.15s; }}
  .report-btn:hover {{ opacity: 0.8; }}
  .report-btn-alt {{ background: transparent; border: 1px solid var(--border); color: var(--fg2); font-size: 11px; padding: 3px 10px; }}
  .report-btn-alt:hover {{ background: var(--bg2); color: var(--fg); opacity: 1; }}
  .meta-bar {{ padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; gap: 24px; flex-wrap: wrap; font-size: 12px; color: var(--fg2); }}
  .meta-bar span {{ display: flex; align-items: center; gap: 4px; }}
  .meta-label {{ color: var(--fg3); }}
  .meta-val {{ color: var(--fg); font-family: var(--mono); font-size: 12px; }}
  .content {{ padding: 24px; max-width: 1200px; margin: 0 auto; }}
  .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .metric {{ background: var(--bg2); border-radius: var(--radius); padding: 16px; }}
  .metric-label {{ font-size: 12px; color: var(--fg3); margin-bottom: 4px; }}
  .metric-value {{ font-size: 22px; font-weight: 500; font-family: var(--mono); }}
  .metric-sub {{ font-size: 11px; color: var(--fg3); margin-top: 2px; font-family: var(--mono); }}
  .chart-section {{ margin-bottom: 24px; }}
  .section-title {{ font-size: 13px; font-weight: 500; color: var(--fg2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
  .chart-card {{ background: var(--bg2); border-radius: var(--radius-lg); padding: 20px; }}
  .chart-legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 12px; font-size: 11px; color: var(--fg2); }}
  .chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}
  .legend-line {{ width: 12px; height: 2px; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  .geo-card {{ background: var(--bg2); border-radius: var(--radius-lg); padding: 16px 20px; }}
  .geo-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }}
  .geo-stat {{ display: flex; flex-direction: column; }}
  .geo-label {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; color: var(--fg3); text-transform: uppercase; margin-bottom: 3px; }}
  .geo-value {{ font-family: var(--mono); font-size: 20px; color: var(--fg); }}
  .geo-trend {{ color: #D87C5A; font-size: 14px; margin-left: 4px; }}
  .geo-sublabel {{ font-family: var(--mono); font-size: 10px; color: var(--fg3); margin-bottom: 4px; display: flex; justify-content: space-between; }}
  .geo-spark {{ margin-bottom: 10px; }}
  .geo-strip {{ margin-bottom: 8px; }}
  .geo-strip-cells {{ display: flex; height: 14px; border-radius: 1px; overflow: hidden; }}
  .geo-cell {{ flex: 1 1 0; min-width: 0; height: 14px; border-right: 1px solid var(--bg2); cursor: default; transition: opacity 0.1s; }}
  .geo-cell:last-child {{ border-right: none; }}
  .geo-cell:hover {{ outline: 1.5px solid var(--fg); outline-offset: 0; z-index: 2; position: relative; }}
  .geo-caption {{ font-family: var(--mono); font-size: 10px; color: var(--fg3); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(3, 1fr); }} }}
  .findings-card {{ background: var(--bg2); border-radius: var(--radius-lg); padding: 20px; }}
  .finding-item {{ display: flex; align-items: flex-start; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
  .finding-item:last-child {{ border-bottom: none; }}
  .finding-badge {{ flex-shrink: 0; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; font-family: var(--mono); }}
  .badge-complete {{ background: var(--green-bg); color: var(--green-fg); }}
  .iterations-card {{ background: var(--bg2); border-radius: var(--radius-lg); padding: 20px; max-height: 480px; overflow-y: auto; }}
  .iter-row {{ display: grid; grid-template-columns: 44px 36px 1fr; gap: 8px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 12px; }}
  .iter-row:last-child {{ border-bottom: none; }}
  .iter-num {{ font-family: var(--mono); color: var(--fg3); font-size: 11px; }}
  .iter-score {{ font-family: var(--mono); font-weight: 500; text-align: center; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .score-high {{ background: var(--green-bg); color: var(--green-fg); }}
  .score-med {{ background: var(--amber-bg); color: var(--amber-fg); }}
  .score-fail {{ background: var(--red-bg); color: var(--red-fg); }}
  .iter-summary {{ color: var(--fg2); line-height: 1.4; }}
  .health-card {{ background: var(--bg2); border-radius: var(--radius-lg); padding: 20px; }}
  .health-row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
  .health-row:last-child {{ border-bottom: none; }}
  .health-label {{ color: var(--fg2); }}
  .health-val {{ font-family: var(--mono); font-weight: 500; font-size: 12px; }}
  .bar-wrap {{ width: 120px; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 3px; }}
  .bar-green {{ background: var(--green); }}
  .bar-amber {{ background: var(--amber); }}
  .bar-red {{ background: var(--red); }}
  .footer {{ padding: 12px 24px; border-top: 1px solid var(--border); font-size: 11px; color: var(--fg3); text-align: center; }}
  .hm-row {{ display: flex; align-items: center; }}
  .hm-label {{ width: 140px; min-width: 140px; box-sizing: border-box; font-size: 11px; color: var(--fg2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 8px; text-align: right; height: {cell_h}px; line-height: {cell_h}px; }}
  .hm-cells {{ display: flex; flex: 1; }}
  .hm-c {{ flex: 1 1 0; min-width: 0; height: {cell_h}px; border-radius: 1px; cursor: default; position: relative; border-right: 1px solid var(--bg2); }}
  .hm-c:last-child {{ border-right: none; }}
  .hm-c:hover {{ outline: 1.5px solid var(--fg); outline-offset: 0; z-index: 2; }}
  .hm-empty {{ background: var(--bg3); opacity: 0.2; }}
  .hm-nums {{ display: flex; flex: 1; margin-left: 140px; }}
  .hm-n {{ flex: 1 1 0; min-width: 0; font-size: 8px; color: var(--fg3); text-align: center; }}
  .hm-legend {{ display: flex; gap: 14px; align-items: center; margin-top: 10px; font-size: 11px; color: var(--fg2); flex-wrap: wrap; }}
  .hm-lsw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin-right: 3px; }}
  .hm-ldot {{ display: inline-block; width: 4px; height: 4px; border-radius: 50%; position: absolute; bottom: 1px; right: 1px; }}
  .hm-tip {{ position: fixed; background: var(--bg); border: 1px solid var(--border2); border-radius: var(--radius); padding: 8px 12px; font-size: 11px; color: var(--fg); pointer-events: none; z-index: 200; display: none; max-width: 320px; line-height: 1.45; font-family: var(--mono); white-space: pre-line; }}
</style>
</head>
<body>

<div class="header">
  <div style="display: flex; align-items: center; gap: 16px;">
    <div class="logo">delv-e <span>dashboard</span></div>
    {status_html}
  </div>
  <div style="font-size: 12px; color: var(--fg3); font-family: var(--mono);">{d['timestamp']}</div>
</div>

<div class="meta-bar">
  <span><span class="meta-label">Dataset</span> <span class="meta-val">{_escape(d['dataset_shape'])}</span></span>
  <span><span class="meta-label">Code</span> <span class="meta-val">{_short_model(d['code_model'])}</span></span>
  <span><span class="meta-label">Agents</span> <span class="meta-val">{_short_model(d['agent_model'])}</span></span>
  {premium_meta}
</div>
<div style="padding: 10px 24px; border-bottom: 1px solid var(--border); font-size: 12px;">
  <span style="color: var(--fg3);">Question</span> <span style="color: var(--fg); font-family: var(--mono); font-size: 12px;">{_escape(d['seed_question'])}</span>
</div>

<div class="content">

  <div class="metrics">
    <div class="metric">
      <div class="metric-label">Progress</div>
      <div class="metric-value">{d['iteration']}<span style="font-size: 14px; color: var(--fg3);"> / {d['max_iterations']}</span></div>
      <div class="metric-sub">iterations</div>
    </div>
    <div class="metric">
      <div class="metric-label">Established findings</div>
      <div class="metric-value">{len(d['established'])}</div>
      <div class="metric-sub">{maturity_note}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Structural Landscape</div>
      <div class="metric-value">{sum(d.get('status_counts', {}).values())}</div>
      <div class="metric-sub">tagged findings</div>
    </div>
    <div class="metric">
      <div class="metric-label">Mean score</div>
      <div class="metric-value">{d['mean_score']}</div>
      <div class="metric-sub">{d['significant']} scored 7+</div>
    </div>
    <div class="metric">
      <div class="metric-label">Cost</div>
      <div class="metric-value">${d['total_cost']:.2f}</div>
      <div class="metric-sub">{d['total_calls']} LLM calls</div>
    </div>
  </div>

  <div class="chart-section">
    <div class="section-title">Score timeline</div>
    <div class="chart-card">
      <div class="chart-legend">
        <span><span class="legend-dot" style="background: #16A34A;"></span> Score 7-10</span>
        <span><span class="legend-dot" style="background: #D97706;"></span> Score 5-6</span>
        <span><span class="legend-dot" style="background: #DC2626;"></span> Score 1-4</span>
        <span><span class="legend-dot" style="background: #D1D5DB;"></span> Runner-up</span>
        <span><span class="legend-dot" style="background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);"></span> HOLD (depth)</span>
        <span><span class="legend-line" style="background: #7C3AED; border-top: 1px dashed #7C3AED; height: 0;"></span> Pivot / abandon</span>
      </div>
      <div class="chart-wrap">
        <canvas id="scoreChart"></canvas>
      </div>
    </div>
  </div>

  <div class="chart-section">
    <div class="section-title">Cumulative score (significant findings only, 7+)</div>
    <div class="chart-card">
      <div class="chart-legend">
        <span><span class="legend-dot" style="background: #16A34A;"></span> Cumulative (scores 7+ only)</span>
        <span><span class="legend-dot" style="background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);"></span> HOLD (depth)</span>
        <span><span class="legend-line" style="background: #7C3AED; border-top: 1px dashed #7C3AED; height: 0;"></span> Pivot / abandon</span>
      </div>
      <div class="chart-wrap">
        <canvas id="cumChart"></canvas>
      </div>
    </div>
  </div>

{geometry_html}

{heatmap_html}

  <div class="two-col">
    <div>
      <div class="section-title">Established findings</div>
      <div class="findings-card" style="max-height: 480px; overflow-y: auto;">
        {findings_html}
      </div>
    </div>
    <div>
      <div class="section-title">Recent iterations</div>
      <div class="iterations-card">
        <div class="iter-row" style="color: var(--fg3); font-weight: 500; font-size: 11px; border-bottom: 1px solid var(--border2);">
          <span>Iter</span><span style="text-align: center;">Score</span><span>Finding</span>
        </div>
        {recent_html}
      </div>
    </div>
  </div>

  <div class="two-col">
    <div>
      <div class="section-title">Exploration health</div>
      <div class="health-card">
        <div class="health-row">
          <span class="health-label">Arcs explored</span>
          <span class="health-val">{d['n_arcs']}</span>
        </div>
        <div class="health-row">
          <span class="health-label">Breadth</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div class="bar-wrap"><div class="bar-fill {breadth_color}" style="width: {breadth_pct}%;"></div></div>
            <span class="health-val">{d['breadth']}</span>
          </div>
        </div>
        <div class="health-row">
          <span class="health-label">Findings</span>
          <span class="health-val">{d['topics']} established</span>
        </div>
        <div class="health-row">
          <span class="health-label">Landscape</span>
          <span class="health-val" style="max-width: 280px; text-align: right; font-size: 11px; line-height: 1.3; font-weight: 400; color: var(--fg2);">{_escape(d['landscape_preview'][:150])}</span>
        </div>
      </div>
    </div>
    <div>
      <div class="section-title">Agent calls breakdown</div>
      <div class="health-card">
        {agent_html}
      </div>
    </div>
  </div>

{probe_html}

{search_html}

{rotation_html}

</div>

<div class="footer">
  delv-e dashboard &middot; auto-refreshes every 30s &middot; last updated {d['timestamp']}
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const winners = {winners_json};
const losers = {losers_json};
const holdBands = {hold_json};
const pivotIters = {pivot_json};
const commitments = {commitments_json};
const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
const labels = Array.from({{length: winners.length}}, (_, i) => i + 1);

function makeHoldPlugin(id) {{
  return {{
    id: id,
    beforeDraw(chart) {{
      const {{ctx, chartArea: {{left,right,top,bottom}}, scales: {{x}}}} = chart;
      ctx.save();
      ctx.fillStyle = isDark ? 'rgba(124,58,237,0.10)' : 'rgba(124,58,237,0.06)';
      holdBands.forEach(([s,e]) => {{
        const x1 = x.getPixelForValue(s - 1.5);
        const x2 = x.getPixelForValue(e - 0.5);
        ctx.fillRect(x1, top, x2 - x1, bottom - top);
      }});
      ctx.restore();
    }}
  }};
}}

function makePivotPlugin(id) {{
  return {{
    id: id,
    afterDraw(chart) {{
      const {{ctx, chartArea: {{top,bottom}}, scales: {{x}}}} = chart;
      ctx.save();
      ctx.setLineDash([4,4]);
      ctx.strokeStyle = isDark ? '#A78BFA' : '#7C3AED';
      ctx.lineWidth = 0.8;
      pivotIters.forEach(i => {{
        const px = x.getPixelForValue(i - 1);
        ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
      }});
      ctx.restore();
    }}
  }};
}}

const n = winners.length;
const tickStep = n <= 10 ? 1 : n <= 30 ? 5 : 10;
const sharedXScale = {{
  ticks: {{
    callback: (v, i) => (i + 1) % tickStep === 0 || i === 0 ? i + 1 : '',
    color: isDark ? '#6B6A66' : '#9C9B96',
    font: {{ family: "'IBM Plex Mono', monospace", size: 10 }}
  }},
  grid: {{ display: false }},
  title: {{ display: true, text: 'Iteration', color: isDark ? '#6B6A66' : '#9C9B96', font: {{ family: "'IBM Plex Sans', sans-serif", size: 11 }} }}
}};

const sharedTooltip = {{
  backgroundColor: isDark ? '#282826' : '#fff',
  titleColor: isDark ? '#E8E7E3' : '#1A1A18',
  bodyColor: isDark ? '#9C9B96' : '#6B6A66',
  borderColor: isDark ? '#3A3A37' : '#E0DFDB',
  borderWidth: 1,
  padding: 10,
  titleFont: {{ family: "'IBM Plex Mono', monospace", size: 12 }},
  bodyFont: {{ family: "'IBM Plex Sans', sans-serif", size: 12 }},
}};

new Chart(document.getElementById('scoreChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Winner',
        data: winners,
        backgroundColor: winners.map(s => {{
          if (s >= 7) return isDark ? 'rgba(74,222,128,0.7)' : 'rgba(22,163,74,0.75)';
          if (s >= 5) return isDark ? 'rgba(251,191,36,0.6)' : 'rgba(217,119,6,0.6)';
          return isDark ? 'rgba(248,113,113,0.6)' : 'rgba(220,38,38,0.55)';
        }}),
        borderRadius: 2, barPercentage: 0.85, categoryPercentage: 0.9, order: 1
      }},
      {{
        label: 'Runner-up',
        data: losers,
        backgroundColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
        borderRadius: 2, barPercentage: 0.85, categoryPercentage: 0.9, order: 2
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index' }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: Object.assign({{}}, sharedTooltip, {{
        callbacks: {{
          title: (items) => {{
            const i = items[0].dataIndex + 1;
            return 'Iteration ' + i + (commitments[i] ? ' \\u00b7 ' + commitments[i] : '');
          }},
          label: (item) => item.dataset.label + ': ' + item.raw
        }}
      }})
    }},
    scales: {{
      x: sharedXScale,
      y: {{
        min: 0, max: 10,
        ticks: {{ stepSize: 2, color: isDark ? '#6B6A66' : '#9C9B96', font: {{ family: "'IBM Plex Mono', monospace", size: 10 }} }},
        grid: {{ color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }},
        title: {{ display: true, text: 'Score', color: isDark ? '#6B6A66' : '#9C9B96', font: {{ family: "'IBM Plex Sans', sans-serif", size: 11 }} }}
      }}
    }}
  }},
  plugins: [makeHoldPlugin('h1'), makePivotPlugin('p1')]
}});

const cumScores = [];
let cumTotal = 0;
winners.forEach(s => {{ cumTotal += (s >= 7 ? s : 0); cumScores.push(cumTotal); }});

new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [{{
      label: 'Cumulative',
      data: cumScores,
      borderColor: isDark ? '#4ADE80' : '#16A34A',
      backgroundColor: isDark ? 'rgba(74,222,128,0.08)' : 'rgba(22,163,74,0.06)',
      fill: true, borderWidth: 2, pointRadius: 0,
      pointHoverRadius: 4, pointHoverBackgroundColor: isDark ? '#4ADE80' : '#16A34A',
      tension: 0
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: Object.assign({{}}, sharedTooltip, {{
        callbacks: {{
          title: (items) => {{
            const i = items[0].dataIndex + 1;
            return 'Iteration ' + i + (commitments[i] ? ' \\u00b7 ' + commitments[i] : '');
          }},
          label: (item) => {{
            const i = item.dataIndex;
            const delta = winners[i] >= 7 ? winners[i] : 0;
            return 'Cumulative: ' + cumScores[i] + '  (+' + delta + ', score ' + winners[i] + ')';
          }}
        }}
      }})
    }},
    scales: {{
      x: sharedXScale,
      y: {{
        beginAtZero: true,
        ticks: {{ color: isDark ? '#6B6A66' : '#9C9B96', font: {{ family: "'IBM Plex Mono', monospace", size: 10 }} }},
        grid: {{ color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }},
        title: {{ display: true, text: 'Cumulative score', color: isDark ? '#6B6A66' : '#9C9B96', font: {{ family: "'IBM Plex Sans', sans-serif", size: 11 }} }}
      }}
    }}
  }},
  plugins: [makeHoldPlugin('h2'), makePivotPlugin('p2')]
}});

(function() {{
  const tip = document.getElementById('hmTip');
  if (!tip) return;
  document.addEventListener('mousemove', function(e) {{
    const cell = e.target.closest('[data-hmtip]');
    if (!cell) {{ tip.style.display = 'none'; return; }}
    tip.textContent = cell.getAttribute('data-hmtip');
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 320) + 'px';
    tip.style.top = (e.clientY - 36) + 'px';
  }});
}})();

</script>
</body>
</html>'''
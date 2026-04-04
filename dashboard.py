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
        data = _extract_data(explorer, engine, iteration, max_iterations)
        html = _render_html(data)
        path = os.path.join(output_dir, "dashboard.html")
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            f.write(html)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"Dashboard write failed: {e}")


def _extract_data(explorer, engine, iteration, max_iterations):
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
            'iteration': node.get('iteration_added', 0),
        })
    nodes.sort(key=lambda x: x['counter'])

    # Group by actual iteration (handles variable node counts)
    from collections import defaultdict
    iter_groups = defaultdict(list)
    for n in nodes:
        iter_groups[n['iteration']].append(n)

    iter_data = []
    for it_num in sorted(iter_groups.keys()):
        group = iter_groups[it_num]
        winner = max(group, key=lambda x: x['score'])
        loser = min(group, key=lambda x: x['score']) if len(group) > 1 else winner
        iter_data.append({
            'winner_score': winner['score'],
            'loser_score': loser['score'],
            'summary': winner['summary'][:140],
        })

    winner_scores = [d['winner_score'] for d in iter_data]
    loser_scores = [d['loser_score'] for d in iter_data]

    # --- Phase map ---
    phase_at = {}
    current_phase = 'MAPPING'
    transitions = {p[0]: p[2] for p in explorer.phase_history}
    for i in range(1, iteration + 1):
        if i in transitions:
            current_phase = transitions[i]
        phase_at[i] = current_phase

    # Pursuing bands
    pursuing_bands = []
    band_start = None
    for i in range(1, iteration + 1):
        if phase_at.get(i) == 'PURSUING' and band_start is None:
            band_start = i
        if phase_at.get(i) != 'PURSUING' and band_start is not None:
            pursuing_bands.append([band_start, i - 1])
            band_start = None
    if band_start is not None:
        pursuing_bands.append([band_start, iteration])

    # --- Phase transition iterations (strategic inflection points) ---
    transition_iters = [t[0] + 1 for t in explorer.phase_history] if explorer.phase_history else []

    # --- Parse research model ---
    rm = explorer.research_model or ''
    established = _parse_section(rm, '## Established Findings')
    connections = _parse_section(rm, '## Cross-Finding Connections')
    maturity_lines = _parse_section(rm, '## Finding Maturity')
    health_lines = _parse_section(rm, '## Exploration Health')
    gap_lines = _parse_section(rm, '## Biggest Gap')

    # Parse health metrics
    breadth = 'UNKNOWN'
    topics = 0
    for line in health_lines:
        if 'Breadth:' in line:
            breadth = line.split('Breadth:')[-1].strip()
        m = re.search(r'Topics investigated:\s*(\d+)', line)
        if m:
            topics = int(m.group(1))

    biggest_gap = ' '.join(gap_lines)[:200] if gap_lines else 'None identified'

    # Count confirmed connections
    confirmed = sum(1 for c in connections if 'confirmed' in c.lower() or 'tested' in c.lower())

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
    dataset_shape = f"{engine.df.shape[0]:,} rows x {engine.df.shape[1]} cols"

    # --- Mean score ---
    mean_score = round(sum(winner_scores) / max(len(winner_scores), 1), 1)
    significant = sum(1 for s in winner_scores if s >= 7)

    # --- Status ---
    is_complete = iteration >= max_iterations

    return {
        'iteration': iteration,
        'max_iterations': max_iterations,
        'is_complete': is_complete,
        'winner_scores': winner_scores,
        'loser_scores': loser_scores,
        'iter_data': iter_data,
        'phase_at': phase_at,
        'pursuing_bands': pursuing_bands,
        'transition_iters': transition_iters,
        'established': established,
        'connections': connections,
        'confirmed_connections': confirmed,
        'maturity_lines': maturity_lines,
        'breadth': breadth,
        'topics': topics,
        'biggest_gap': biggest_gap,
        'current_phase': explorer.current_phase,
        'phase_transitions': len(explorer.phase_history),
        'stagnation_count': explorer.stagnation_count,
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
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
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
            lines.append(line.strip()[2:])
        elif in_section and line.strip() and not line.strip().startswith('#'):
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
        status_html = '<span class="status-pill status-complete">Complete</span>'
    else:
        status_html = '<span class="status-pill status-running">Running</span>'

    # Metric cards
    maturity_note = ''
    if d['maturity_lines']:
        if any('COMPLETE' in l.upper() for l in d['maturity_lines']):
            maturity_note = 'all COMPLETE maturity'
        else:
            active = sum(1 for l in d['maturity_lines'] if l.strip())
            maturity_note = f'{active} in progress'
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
        'Strategic Review', 'Reframing Probe', 'Seed Decomposition', 'Synthesis Generator'
    ]
    max_calls = max(d['agent_counts'].values()) if d['agent_counts'] else 1
    agent_html = ''
    for agent in agent_order:
        count = d['agent_counts'].get(agent, 0)
        if count == 0:
            continue
        pct = round(count / max_calls * 100)
        label = agent.lower().replace('_', ' ')
        is_premium = agent in ('Strategic Review', 'Reframing Probe', 'Seed Decomposition', 'Synthesis Generator')
        bar_color = 'var(--purple)' if is_premium else ('var(--amber)' if agent == 'Error Corrector' else 'var(--blue)')
        premium_tag = ' <span style="font-weight: 400; color: var(--fg3);">premium</span>' if is_premium else ''
        agent_html += f'''
        <div class="health-row">
          <span class="health-label">{label}</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div class="bar-wrap" style="width: 180px;">
              <div class="bar-fill" style="width: {max(pct, 1)}%; background: {bar_color};"></div>
            </div>
            <span class="health-val">{count}{premium_tag}</span>
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

    # Phase pill
    phase_class = 'phase-pursuing' if d['current_phase'] == 'PURSUING' else 'phase-mapping'

    # Premium model display
    premium_meta = ''
    if d['premium_model']:
        premium_meta = f'<span><span class="meta-label">Premium</span> <span class="meta-val">{_short_model(d["premium_model"])}</span></span>'

    # JSON data for charts
    winners_json = json.dumps(d['winner_scores'])
    losers_json = json.dumps(d['loser_scores'])
    pursuing_json = json.dumps(d['pursuing_bands'])
    conn_json = json.dumps(d['transition_iters'])
    phases_json = json.dumps(d['phase_at'])

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
  .phase-pill {{ font-size: 11px; font-weight: 500; padding: 2px 10px; border-radius: 10px; font-family: var(--mono); }}
  .phase-mapping {{ background: var(--blue-bg); color: var(--blue-fg); }}
  .phase-pursuing {{ background: var(--purple-bg); color: var(--purple-fg); }}
  .footer {{ padding: 12px 24px; border-top: 1px solid var(--border); font-size: 11px; color: var(--fg3); text-align: center; }}
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
      <div class="metric-label">Connections tested</div>
      <div class="metric-value">{len(d['connections'])}</div>
      <div class="metric-sub">{d['confirmed_connections']} confirmed</div>
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
        <span><span class="legend-dot" style="background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);"></span> PURSUING</span>
        <span><span class="legend-line" style="background: #7C3AED; border-top: 1px dashed #7C3AED; height: 0;"></span> Phase transition</span>
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
        <span><span class="legend-dot" style="background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);"></span> PURSUING</span>
        <span><span class="legend-line" style="background: #7C3AED; border-top: 1px dashed #7C3AED; height: 0;"></span> Phase transition</span>
      </div>
      <div class="chart-wrap">
        <canvas id="cumChart"></canvas>
      </div>
    </div>
  </div>

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
          <span class="health-label">Current phase</span>
          <span class="phase-pill {phase_class}">{d['current_phase']}</span>
        </div>
        <div class="health-row">
          <span class="health-label">Breadth</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div class="bar-wrap"><div class="bar-fill {breadth_color}" style="width: {breadth_pct}%;"></div></div>
            <span class="health-val">{d['breadth']}</span>
          </div>
        </div>
        <div class="health-row">
          <span class="health-label">Topics investigated</span>
          <span class="health-val">{d['topics']}</span>
        </div>
        <div class="health-row">
          <span class="health-label">Phase transitions</span>
          <span class="health-val">{d['phase_transitions']}</span>
        </div>
        <div class="health-row">
          <span class="health-label">Stagnation count</span>
          <span class="health-val">{d['stagnation_count']}</span>
        </div>
        <div class="health-row">
          <span class="health-label">Biggest gap</span>
          <span class="health-val" style="max-width: 280px; text-align: right; font-size: 11px; line-height: 1.3; font-weight: 400; color: var(--fg2);">{_escape(d['biggest_gap'][:150])}</span>
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

</div>

<div class="footer">
  delv-e dashboard &middot; auto-refreshes every 30s &middot; last updated {d['timestamp']}
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const winners = {winners_json};
const losers = {losers_json};
const pursuingBands = {pursuing_json};
const connIters = {conn_json};
const phases = {phases_json};
const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
const labels = Array.from({{length: winners.length}}, (_, i) => i + 1);

function makePhasePlugin(id) {{
  return {{
    id: id,
    beforeDraw(chart) {{
      const {{ctx, chartArea: {{left,right,top,bottom}}, scales: {{x}}}} = chart;
      ctx.save();
      ctx.fillStyle = isDark ? 'rgba(124,58,237,0.10)' : 'rgba(124,58,237,0.06)';
      pursuingBands.forEach(([s,e]) => {{
        const x1 = x.getPixelForValue(s - 1.5);
        const x2 = x.getPixelForValue(e - 0.5);
        ctx.fillRect(x1, top, x2 - x1, bottom - top);
      }});
      ctx.restore();
    }}
  }};
}}

function makeConnPlugin(id) {{
  return {{
    id: id,
    afterDraw(chart) {{
      const {{ctx, chartArea: {{top,bottom}}, scales: {{x}}}} = chart;
      ctx.save();
      ctx.setLineDash([4,4]);
      ctx.strokeStyle = isDark ? '#A78BFA' : '#7C3AED';
      ctx.lineWidth = 0.8;
      connIters.forEach(i => {{
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
            return 'Iteration ' + i + ' \\u00b7 ' + (phases[i] || 'MAPPING');
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
  plugins: [makePhasePlugin('p1'), makeConnPlugin('c1')]
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
            return 'Iteration ' + i + ' \\u00b7 ' + (phases[i] || 'MAPPING');
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
  plugins: [makePhasePlugin('p2'), makeConnPlugin('c2')]
}});
</script>
</body>
</html>'''
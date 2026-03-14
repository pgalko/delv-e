"""
Terminal styling with ANSI colors. Claude Code CLI-inspired layout.
"""

import os
import sys
import textwrap
import threading
import time

# ── ANSI codes ──────────────────────────────────────────────

_COLOR = os.environ.get("NO_COLOR") is None

RESET   = "\033[0m"   if _COLOR else ""
BOLD    = "\033[1m"    if _COLOR else ""
DIM     = "\033[2m"    if _COLOR else ""
ITALIC  = "\033[3m"    if _COLOR else ""

CYAN    = "\033[36m"   if _COLOR else ""
GREEN   = "\033[32m"   if _COLOR else ""
YELLOW  = "\033[33m"   if _COLOR else ""
RED     = "\033[31m"   if _COLOR else ""
MAGENTA = "\033[35m"   if _COLOR else ""
BLUE    = "\033[34m"   if _COLOR else ""
WHITE   = "\033[97m"   if _COLOR else ""
GRAY    = "\033[90m"   if _COLOR else ""

BRIGHT_CYAN    = "\033[96m" if _COLOR else ""
BRIGHT_GREEN   = "\033[92m" if _COLOR else ""
BRIGHT_YELLOW  = "\033[93m" if _COLOR else ""
BRIGHT_MAGENTA = "\033[95m" if _COLOR else ""

# ── Text helpers ────────────────────────────────────────────

def bold(t):       return f"{BOLD}{t}{RESET}"
def dim(t):        return f"{DIM}{t}{RESET}"
def italic(t):     return f"{ITALIC}{t}{RESET}"
def cyan(t):       return f"{CYAN}{t}{RESET}"
def green(t):      return f"{GREEN}{t}{RESET}"
def yellow(t):     return f"{YELLOW}{t}{RESET}"
def red(t):        return f"{RED}{t}{RESET}"
def magenta(t):    return f"{MAGENTA}{t}{RESET}"
def blue(t):       return f"{BLUE}{t}{RESET}"
def gray(t):       return f"{GRAY}{t}{RESET}"

# ── Spinner ────────────────────────────────────────────────

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

class spinner:
    """Animated spinner as context manager. Shows progress during silent LLM calls.

    Usage:
        with style.spinner("Evaluating results"):
            result = slow_llm_call()
    """

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = _BRAILLE[i % len(_BRAILLE)]
            sys.stdout.write(f"\r  {DIM}{frame} {self.label}{RESET}  ")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)
        # Clear spinner line
        sys.stdout.write(f"\r  {GREEN}✓{RESET} {DIM}{self.label}{RESET}  \n")
        sys.stdout.flush()

    def __enter__(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        return False

# ── Phase styling ───────────────────────────────────────────

PHASE_COLORS = {
    "MAPPING":    BRIGHT_CYAN,
    "PURSUING":   BRIGHT_YELLOW,
    "CONVERGING": MAGENTA,
    "REFRAMING":  RED,
}

def phase(name):
    color = PHASE_COLORS.get(name, WHITE)
    return f"{BOLD}{color}{name}{RESET}"

def phase_transition(old, new):
    return f"{phase(old)} {DIM}→{RESET} {phase(new)}"

# ── Box drawing ─────────────────────────────────────────────

LOGO = (
    f"\n{WHITE}"
    "    ██████╗ ███████╗██╗    ██╗   ██╗    ███████╗\n"
    "    ██╔══██╗██╔════╝██║    ██║   ██║    ██╔════╝\n"
    "    ██║  ██║█████╗  ██║    ██║   ██║ ██ █████╗  \n"
    "    ██║  ██║██╔══╝  ██║    ╚██╗ ██╔╝    ██╔══╝  \n"
    "    ██████╔╝███████╗███████╗╚████╔╝     ███████╗\n"
    "    ╚═════╝ ╚══════╝╚══════╝ ╚═══╝      ╚══════╝"
    f"{RESET}"
)

VERSION = "0.1.0"
TAGLINE = "Deep Exploratory Learning & Visualization Engine"

def config_lines(df_shape, max_iterations, num_parallel, output_dir, agent_model, code_model):
    """Run config info without logo — used after interactive prompt."""
    def short_model(m):
        return m.split("-202")[0] if "-202" in m else m

    lines = []
    lines.append(f"    {DIM}Loop{RESET}   {WHITE}{max_iterations} iterations{RESET}  {DIM}×{RESET}  {WHITE}{num_parallel} parallel{RESET}")
    lines.append(f"    {DIM}Code{RESET}   {WHITE}{short_model(code_model)}{RESET}")
    lines.append(f"    {DIM}Agents{RESET} {WHITE}{short_model(agent_model)}{RESET}")
    lines.append(f"    {DIM}Output{RESET} {WHITE}{output_dir}/{RESET}")
    return "\n".join(lines)


def splash_header(df_shape, max_iterations, num_parallel, output_dir, agent_model, code_model):
    """Full startup banner with logo and run info — used for inline mode."""
    def short_model(m):
        return m.split("-202")[0] if "-202" in m else m

    lines = [LOGO]
    lines.append(f"    {DIM}{VERSION} — {TAGLINE}{RESET}")
    lines.append("")
    lines.append(f"    {DIM}Data{RESET}   {WHITE}{df_shape}{RESET}")
    lines.append(f"    {DIM}Loop{RESET}   {WHITE}{max_iterations} iterations{RESET}  {DIM}×{RESET}  {WHITE}{num_parallel} parallel{RESET}")
    lines.append(f"    {DIM}Code{RESET}   {WHITE}{short_model(code_model)}{RESET}")
    lines.append(f"    {DIM}Agents{RESET} {WHITE}{short_model(agent_model)}{RESET}")
    lines.append(f"    {DIM}Output{RESET} {WHITE}{output_dir}/{RESET}")
    return "\n".join(lines)


def box_header(lines, width=66):
    """Draw a bordered box around 1-3 lines of text."""
    top    = f"  {CYAN}╭{'─' * (width - 2)}╮{RESET}"
    bottom = f"  {CYAN}╰{'─' * (width - 2)}╯{RESET}"
    rows = []
    rows.append(top)
    for line in lines:
        padding = width - 4 - len(line)
        if padding < 0:
            line = line[:width - 7] + "..."
            padding = 0
        rows.append(f"  {CYAN}│{RESET} {BOLD}{WHITE}{line}{RESET}{' ' * padding} {CYAN}│{RESET}")
    rows.append(bottom)
    return "\n".join(rows)

# ── Iteration bar ───────────────────────────────────────────

def iteration_bar(iteration, max_iter, phase_name, width=66):
    """Full-width bar with iteration number left, phase right."""
    left = f" Iteration {iteration}/{max_iter} "
    right = f" {phase_name} "
    fill = width - len(left) - len(right) - 2
    if fill < 4:
        fill = 4
    color = PHASE_COLORS.get(phase_name, WHITE)
    bar = f"{BOLD}{color}{'━' * 2}{left}{'━' * fill}{right}{'━' * 2}{RESET}"
    return f"\n{bar}"

# ── Question display ────────────────────────────────────────

def clean_question(text):
    """Strip metadata from a generated question, keep only the analytical question."""
    # Cut at common metadata markers (may or may not have newline before them)
    for marker in ["Narrative connection:", "Code execution needed:",
                   "--- ", "This directly addresses the Biggest Gap",
                   "This addresses the Biggest Gap",
                   "This question directly addresses"]:
        idx = text.find(marker)
        if idx > 20:  # only cut if there's meaningful text before the marker
            text = text[:idx]
    # Remove leading bold title if present: **Title** rest
    stripped = text.strip()
    if stripped.startswith("**"):
        end = stripped.find("**", 2)
        if 0 < end < 80:
            stripped = stripped[end+2:].strip()
            if stripped and stripped[0] in ':–—-':
                stripped = stripped[1:].strip()
    return stripped.strip()

def question_display(q_idx, total, category, text, width=66):
    """Format a question as a clean block with wrapping."""
    cleaned = clean_question(text)
    label = f"Q{q_idx}/{total}" if total > 1 else "Q"
    cat = f" {DIM}[{category.upper()}]{RESET}" if category else ""
    header = f"  {BOLD}{CYAN}▸{RESET} {BOLD}{label}{RESET}{cat}"

    # Wrap the question text to fit nicely
    indent = "    "
    wrapped = textwrap.fill(cleaned, width=width - 4, initial_indent="", subsequent_indent="")
    wrapped_lines = wrapped.split("\n")

    lines = [f"{header} {wrapped_lines[0]}"]
    for wl in wrapped_lines[1:]:
        lines.append(f"{indent}{wl}")
    return "\n".join(lines)

# ── Agent & status ──────────────────────────────────────────

def agent(name, model=None):
    if model:
        short = model.split("-202")[0] if "-202" in model else model  # trim date suffix
        return f"    {BLUE}[{name}]{RESET} {DIM}{short}{RESET}"
    return f"    {BLUE}[{name}]{RESET}"

def success(text):
    return f"    {GREEN}✓{RESET} {text}"

def error_msg(text):
    return f"    {RED}✗{RESET} {text}"

def file_ref(path, note=""):
    suffix = f" {DIM}({note}){RESET}" if note else ""
    return f"    {GREEN}📄{RESET} {DIM}{path}{RESET}{suffix}"

def branch_event(text):
    return f"    {YELLOW}⤷ {text}{RESET}"

# ── Result output block ────────────────────────────────────

def result_border():
    return f"    {GRAY}{'┄' * 54}{RESET}"

def result_line(text):
    return f"    {GRAY}│{RESET} {text}"

# ── Score & impact ──────────────────────────────────────────

def score(value, max_val=10):
    if value >= 8:
        return f"{BRIGHT_GREEN}{value}/{max_val}{RESET}"
    elif value >= 5:
        return f"{YELLOW}{value}/{max_val}{RESET}"
    else:
        return f"{RED}{value}/{max_val}{RESET}"

def impact(level):
    if level == "HIGH":
        return f"{BOLD}{BRIGHT_GREEN}{level}{RESET}"
    elif level == "MEDIUM":
        return f"{YELLOW}{level}{RESET}"
    return f"{DIM}{level}{RESET}"

# ── Pipeline summary (evaluate → plan block) ───────────────

def pipeline_summary(selected_q, selected_score, reason,
                     model_impact, old_phase, new_phase,
                     n_questions, phase_mode, n_selected,
                     is_seed=False):
    """Compact summary of the evaluate → interpret → plan steps."""
    lines = []
    lines.append(f"\n  {DIM}{'─' * 56}{RESET}")

    if is_seed:
        lines.append(f"  {DIM}Seed baseline{RESET} │ Score: {score(selected_score)}")
    else:
        reason_text = reason.strip() if reason else ""
        lines.append(f"  {DIM}Winner: Q{selected_q}{RESET} │ Score: {score(selected_score)} │ {DIM}{reason_text}{RESET}")

    # Phase + impact on one line
    if new_phase != old_phase:
        phase_str = phase_transition(old_phase, new_phase)
    else:
        phase_str = f"{phase(new_phase)} {dim('(maintained)')}"
    lines.append(f"  {DIM}Impact:{RESET} {impact(model_impact)} │ {phase_str}")

    # Next step (skip if last iteration)
    if n_questions > 0:
        lines.append(f"  {DIM}Next:{RESET} {n_questions} {phase_mode} questions → selected {n_selected}")
    lines.append(f"  {DIM}{'─' * 56}{RESET}")
    return "\n".join(lines)

# ── Final summary ───────────────────────────────────────────

def final_box(iterations, analyses, avg, dormant, phase_hist, cost_str, output_dir):
    """The completion summary."""
    lines = []
    lines.append(f"\n  {CYAN}╭{'─' * 62}╮{RESET}")
    lines.append(f"  {CYAN}│{RESET} {BOLD}Exploration Complete{RESET}")
    lines.append(f"  {CYAN}│{RESET}")
    lines.append(f"  {CYAN}│{RESET}  {bold(str(iterations))} iterations, {bold(str(analyses))} analyses {dim(f'(avg score: {avg:.1f})')}, {dormant} unexplored branches")

    if phase_hist:
        transitions = " → ".join(phase_transition(o, n) for _, o, n in phase_hist)
        lines.append(f"  {CYAN}│{RESET}  {dim('Phases:')} {transitions}")

    lines.append(f"  {CYAN}│{RESET}  {dim('Cost:')} {cost_str}")
    lines.append(f"  {CYAN}│{RESET}")
    lines.append(f"  {CYAN}│{RESET}  {bold('Output:')}")
    lines.append(f"  {CYAN}│{RESET}    synthesis_report.md")
    lines.append(f"  {CYAN}│{RESET}    research_model.md")
    lines.append(f"  {CYAN}│{RESET}    exploration/")
    lines.append(f"  {CYAN}│{RESET}    cost.txt")
    lines.append(f"  {CYAN}╰{'─' * 62}╯{RESET}")
    return "\n".join(lines)


# ── Exploration tree ────────────────────────────────────────

def exploration_tree(insight_tree, root_node_id):
    """Render the insight tree as an ASCII diagram with colored scores."""
    if not insight_tree or not root_node_id:
        return ""

    STATUS_ORDER = {'active': 0, 'runner_up': 1, 'dormant': 2, 'abandoned': 3}

    def fmt_node(nid):
        n = insight_tree[nid]
        q = clean_question(n['question'])
        if len(q) > 62:
            q = q[:59] + "..."
        sc = n['quality_score']
        status = n['status']

        if status == 'abandoned' or sc == 0:
            return f"{DIM}{q} [✗]{RESET}"
        elif status == 'dormant':
            return f"{DIM}{q}{RESET} {score(sc)} {DIM}(dormant){RESET}"
        elif status == 'runner_up':
            return f"{DIM}{q}{RESET} {score(sc)}"
        else:
            return f"{q} {score(sc)}"

    def walk(nid, prefix, is_last):
        connector = "└── " if is_last else "├── "
        result.append(f"{prefix}{connector}{fmt_node(nid)}")
        children = [c for c in insight_tree[nid].get('children_ids', [])
                     if c in insight_tree]
        children.sort(key=lambda c: STATUS_ORDER.get(insight_tree[c]['status'], 3))
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            walk(child, child_prefix, i == len(children) - 1)

    result = []
    result.append(f"\n  {bold('Exploration Tree')}\n")
    result.append(f"  {fmt_node(root_node_id)}")

    children = [c for c in insight_tree[root_node_id].get('children_ids', [])
                 if c in insight_tree]
    children.sort(key=lambda c: STATUS_ORDER.get(insight_tree[c]['status'], 3))
    for i, child in enumerate(children):
        walk(child, "  ", i == len(children) - 1)

    return "\n".join(result)
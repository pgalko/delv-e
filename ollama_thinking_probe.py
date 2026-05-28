#!/usr/bin/env python3
"""
ollama_thinking_probe.py — probe Ollama Cloud for reasoning-control parameters

Uses the SAME mechanism as delv-e's OllamaProvider in llm.py:
  - OpenAI-compatible SDK
  - base_url from OLLAMA_BASE_URL env var (default http://localhost:11434, suffixed with /v1)
  - api_key="ollama" placeholder (works for local daemons signed into Ollama Cloud
    via `ollama signin`; override with OLLAMA_API_KEY for direct https://ollama.com use)

For each (model, setting) pair, runs the same code-generation prompt N_REPS times
to separate real parameter effects from per-call variance. Captures the actual
HTTP request body sent to Ollama (so we can verify what `extra_body` translates to
on the wire), as well as completion_tokens / elapsed_time / reasoning trace size
per call. Aggregates to mean ± std per cell, then classifies each setting against
a noise floor derived from the baseline cell's own observed variance.

What you're looking for
─────────────────────────────────────────────────────────────────────────────────────
  HONOURED:  mean differs from baseline by more than the baseline cell's noise
             band (default 2σ). The direction (↑ or ↓) is reported. This is a
             real, parameter-driven effect — wire into delv-e's OllamaProvider.

  IGNORED:   mean falls inside the baseline noise band. The parameter is being
             accepted by the API but has no observable effect on output.

  NOISY:     within-cell std is so large that mean is uninformative. Increase
             N_REPS, or accept that this setting's behaviour is unstable on
             this provider.

  REJECTED:  the call returns an error (every rep). The parameter name or value
             type is unsupported on this endpoint.

Verification of what's on the wire
─────────────────────────────────────────────────────────────────────────────────────
An httpx event hook prints the exact JSON body of each outbound request, so you
can confirm that extra_body={"reasoning_effort": "low"} actually surfaces as a
top-level "reasoning_effort": "low" in the request body. The body is also saved
into the results JSON for offline inspection.

Usage
─────────────────────────────────────────────────────────────────────────────────────
  pip install openai           # if not already present
  python ollama_thinking_probe.py

  # against a remote daemon
  OLLAMA_BASE_URL=http://my-host:11434 python ollama_thinking_probe.py

  # against ollama.com directly
  OLLAMA_BASE_URL=https://ollama.com OLLAMA_API_KEY=sk-... python ollama_thinking_probe.py

Runtime
─────────────────────────────────────────────────────────────────────────────────────
  N_REPS × (models × settings) calls. With N_REPS=3 and 2 models × 6 settings,
  that's 36 calls. On kimi-k2.6:cloud each call is 1-5 minutes; on glm-5.1:cloud
  usually under 1 minute. Estimated total: 60-90 minutes. Reduce N_REPS or trim
  MODELS / SETTINGS to narrow scope.
"""

import json
import os
import sys
import time

try:
    from openai import OpenAI
    import httpx
except ImportError as e:
    missing = "openai" if "openai" in str(e) else "httpx"
    print(f"ERROR: {missing} package required. Install with: pip install {missing}",
          file=sys.stderr)
    sys.exit(1)

import statistics


# ─────────────────────────────────────────────────────────────────────
# Configuration — edit MODELS, SETTINGS, or N_REPS to narrow the test surface
# ─────────────────────────────────────────────────────────────────────

# Same base_url resolution as delv-e's OllamaProvider.__init__
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")

# Number of repetitions per (model, setting) cell. Increase if results are
# noisy. N=3 is the minimum to get a meaningful mean ± std; N=5 gives more
# confidence at ~70% more runtime. With N=1 (the original probe) we couldn't
# distinguish parameter effects from per-call variance.
N_REPS = 3

MODELS = [
    "kimi-k2.6:cloud",
    "glm-5.1:cloud",
]

# Each entry: (label, kwargs forwarded to client.chat.completions.create).
# The OpenAI SDK serialises everything in `extra_body` into the JSON request
# body alongside the standard OpenAI fields — this is how non-OpenAI params
# reach the Ollama backend through the /v1/chat/completions endpoint. The
# httpx event hook below prints the actual outbound JSON so we can verify
# that extra_body translates to a top-level field as expected.
#
# We test multiple parameter names because Ollama's OpenAI-compat layer
# accepts different forms for different model families, and the mapping
# isn't fully documented for kimi / glm specifically.
#
# `reasoning=low` (string) was dropped from this set because run-1 confirmed
# it returns a 400 schema error on both kimi and glm — Ollama expects a
# structured object there, not a string. No point burning calls on it.
SETTINGS = [
    ("baseline (no param)",          {}),
    ("reasoning_effort=low",         {"extra_body": {"reasoning_effort": "low"}}),
    ("reasoning_effort=medium",      {"extra_body": {"reasoning_effort": "medium"}}),
    ("reasoning_effort=high",        {"extra_body": {"reasoning_effort": "high"}}),
    ("think=low (string)",           {"extra_body": {"think": "low"}}),
    ("think=false (boolean)",        {"extra_body": {"think": False}}),
]


# Matches what delv-e sends in _call_llm_for_code; the provider caps below this
# for kimi-k2.6 (observed ceiling: 16384), but we send the same number for
# fidelity with the actual delv-e call shape.
MAX_TOKENS = 20000
TEMPERATURE = 0
TIMEOUT_SECONDS = 600  # generous; kimi can take 4-5 min on hard reasoning


# ─────────────────────────────────────────────────────────────────────
# Test prompt — a realistic mid-complexity code-gen task that should
# normally need a few K thinking tokens, not 16K. Similar shape to what
# delv-e's Code Generator agent receives (system + user, code-in-fenced-
# block convention, ###RESULTS_START### markers).
# ─────────────────────────────────────────────────────────────────────

SYSTEM_MSG = """You are an expert data analyst writing Python code to analyse a pandas DataFrame.

The DataFrame `df` has columns: date (datetime), open, high, low, close (floats),
volume (int). It contains daily stock price data over approximately 10 years.

Rules:
- Return code in a single ```python``` block.
- The DataFrame `df` is pre-loaded; do not redefine it.
- Print the results table wrapped in ###RESULTS_START### / ###RESULTS_END### markers.
- Be concise; aim for under 80 lines of code."""

USER_MSG = """For each completed calendar year in the data:
1. Compute the year's total return (close on last trading day / close on first trading day - 1).
2. Compute the year's annualised realised volatility (sqrt(252) * std of daily log returns).
3. Identify the calendar month within that year with the largest absolute return.
4. Compute that best month's return.

Report a table with columns:
  year, total_return_pct, ann_vol_pct, best_month_name, best_month_return_pct

Sort by year ascending. Format percentages to two decimal places."""


# ─────────────────────────────────────────────────────────────────────
# HTTP request body capture
# ─────────────────────────────────────────────────────────────────────
# We install an httpx event hook on the OpenAI client's underlying HTTP
# transport. Each outbound request triggers the hook, which captures the
# raw JSON body into a module-level slot that run_call reads after the
# response returns. This lets us verify EXACTLY what reaches Ollama —
# whether extra_body={"reasoning_effort": "low"} actually surfaces as a
# top-level "reasoning_effort": "low" field in the JSON, or whether the
# SDK nests it / drops it / rewrites it.

_last_request_body = {"raw": None, "parsed": None}


def _capture_request_body(request):
    """httpx event hook: stash the outbound request body so run_call can grab it.

    Called synchronously for each request before it's sent. Replaces any
    previously-captured body so the slot always holds the most recent request.
    """
    try:
        raw = request.content.decode("utf-8") if request.content else ""
    except Exception:
        raw = "<could not decode>"
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
    _last_request_body["raw"] = raw
    _last_request_body["parsed"] = parsed


def make_client():
    """Construct an OpenAI client with the request-body capture hook installed."""
    http_client = httpx.Client(
        timeout=TIMEOUT_SECONDS,
        event_hooks={"request": [_capture_request_body]},
    )
    return OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=API_KEY,
        timeout=TIMEOUT_SECONDS,
        http_client=http_client,
    )


# ─────────────────────────────────────────────────────────────────────
# Test execution
# ─────────────────────────────────────────────────────────────────────

def run_call(client, model, settings_kwargs):
    """Execute one test call. Returns dict with timing, tokens, content metadata.

    Never raises — failures are captured into the returned dict so the test
    loop can continue with subsequent settings/models. Also captures the
    actual JSON request body that was sent (via the httpx hook installed
    by make_client) so we can verify the wire format matches our intent.
    """
    # Clear the capture slot so we know any captured body is from THIS call
    _last_request_body["raw"] = None
    _last_request_body["parsed"] = None

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_MSG},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=TIMEOUT_SECONDS,
            **settings_kwargs,
        )
    except Exception as e:
        return {
            "ok": False,
            "elapsed_s": round(time.time() - t0, 1),
            "error": f"{type(e).__name__}: {str(e)[:240]}",
            "completion_tokens": None,
            "prompt_tokens": None,
            "visible_len": 0,
            "reasoning_len": 0,
            "has_code": False,
            "preview": "",
            "request_body": _last_request_body["parsed"],
        }

    elapsed = time.time() - t0
    msg = response.choices[0].message
    content = msg.content or ""

    # Some thinking-model responses surface the reasoning trace on a separate
    # field (Ollama exposes `thinking` on its native API; some OpenAI-compat
    # wrappers expose `reasoning` or `reasoning_content`). Probe defensively.
    reasoning_text = ""
    for fname in ("thinking", "reasoning", "reasoning_content"):
        try:
            r = getattr(msg, fname, None)
            if r:
                reasoning_text = r if isinstance(r, str) else str(r)
                break
        except Exception:
            continue
    # Also check model_extra (where pydantic stashes unknown fields)
    if not reasoning_text:
        try:
            extra = getattr(msg, "model_extra", None) or {}
            for fname in ("thinking", "reasoning", "reasoning_content"):
                if extra.get(fname):
                    reasoning_text = str(extra[fname])
                    break
        except Exception:
            pass

    usage = response.usage
    return {
        "ok": True,
        "elapsed_s": round(elapsed, 1),
        "error": None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "visible_len": len(content),
        "reasoning_len": len(reasoning_text),
        "has_code": "```" in content,
        "preview": content.strip().replace("\n", " ")[:80],
        "request_body": _last_request_body["parsed"],
    }


def main():
    print("=" * 92)
    print(" Ollama Cloud reasoning-control probe")
    print("=" * 92)
    print(f"  base_url:   {BASE_URL}/v1")
    print(f"  api_key:    {'<set via OLLAMA_API_KEY>' if API_KEY != 'ollama' else '(default placeholder; local daemon mode)'}")
    print(f"  models:     {MODELS}")
    print(f"  settings:   {len(SETTINGS)} variants × {len(MODELS)} models × "
          f"{N_REPS} reps = {len(SETTINGS) * len(MODELS) * N_REPS} calls")
    print(f"  max_tokens: {MAX_TOKENS}  temperature: {TEMPERATURE}  timeout: {TIMEOUT_SECONDS}s")
    print(f"  prompt:     code-gen task (yearly stats over a stock DataFrame)")
    print(f"  capture:    HTTP request body via httpx event hook (verifies wire format)")
    print()
    print(f"  Each kimi call may take 1-5 minutes; glm typically under 1 minute.")
    # Rough estimate: kimi ~2min mean, glm ~50s mean; this is a back-of-envelope.
    est_minutes = N_REPS * (len(SETTINGS) * 2 + len(SETTINGS) * 0.8)
    print(f"  Estimated total runtime: ~{est_minutes:.0f} minutes "
          f"(scales with N_REPS = {N_REPS}; reduce if too long).")
    print()
    try:
        proceed = input("Proceed? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
    if proceed != "y":
        print("aborted")
        sys.exit(0)
    print()

    client = make_client()

    # ─────────────────────────────────────────────────────────────
    # Collection: N_REPS calls per (model, setting) cell
    # ─────────────────────────────────────────────────────────────
    # Results are flat — one dict per call. Aggregation below groups by
    # (model, setting) and computes mean/std across the repetitions.
    results = []
    cell_count = len(MODELS) * len(SETTINGS)
    total_calls = cell_count * N_REPS
    n_done = 0
    body_verified = False  # one-time sanity print

    for model in MODELS:
        print(f"── {model} ".ljust(92, "─"))
        for label, settings_kwargs in SETTINGS:
            for rep in range(1, N_REPS + 1):
                n_done += 1
                tag = f"[{n_done:>3}/{total_calls}] {label:<28} rep {rep}/{N_REPS}"
                print(f"  {tag} … ", end="", flush=True)

                r = run_call(client, model, settings_kwargs)
                r["model"] = model
                r["setting"] = label
                r["rep"] = rep
                results.append(r)

                if r["ok"]:
                    reas_marker = f" reas={r['reasoning_len']}c" if r["reasoning_len"] else ""
                    code_marker = " (no code!)" if not r["has_code"] else ""
                    print(f"OK  {r['elapsed_s']:>6.1f}s  "
                          f"compl={r['completion_tokens']:>5}"
                          f"{reas_marker}{code_marker}")
                else:
                    print(f"FAIL {r['elapsed_s']:>5.1f}s  {r['error'][:80]}")

                # One-time printout of the actual JSON body sent, so the user
                # can verify extra_body did what we expect. Shown for the
                # first call only; subsequent bodies are saved into the JSON
                # dump for offline inspection.
                if not body_verified and r.get("request_body"):
                    body = r["request_body"]
                    keys = sorted(body.keys()) if isinstance(body, dict) else []
                    print(f"        ↳ wire body keys: {keys}")
                    # Show any non-standard fields explicitly — these are what
                    # extra_body translated to. Standard OpenAI fields are
                    # filtered out so the diagnostic value is obvious.
                    standard = {"model", "messages", "max_tokens", "temperature",
                                "stream", "n", "top_p", "frequency_penalty",
                                "presence_penalty", "stop", "user"}
                    extras = {k: v for k, v in (body or {}).items()
                              if k not in standard}
                    if extras:
                        print(f"        ↳ non-standard fields: {extras}")
                    body_verified = True
        print()

    # ─────────────────────────────────────────────────────────────
    # Aggregation: mean ± std per (model, setting) cell
    # ─────────────────────────────────────────────────────────────

    def aggregate_cell(model, setting):
        """Return aggregated stats for one (model, setting) cell.

        Pools successful reps. If 0 successful reps, returns failure info.
        std requires ≥2 successful reps; with N=1 it's reported as None.
        """
        cell = [r for r in results
                if r["model"] == model and r["setting"] == setting]
        ok_cell = [r for r in cell if r["ok"]]
        n_ok = len(ok_cell)
        n_fail = len(cell) - n_ok

        if n_ok == 0:
            errors = list({r["error"] for r in cell if not r["ok"]})
            return {
                "model": model, "setting": setting,
                "n_ok": 0, "n_fail": n_fail,
                "tok_mean": None, "tok_std": None,
                "tok_min": None, "tok_max": None,
                "time_mean": None, "time_std": None,
                "reas_mean": None, "reas_std": None,
                "visible_mean": None,
                "all_have_code": False,
                "errors": errors,
                "reps": cell,
            }

        toks = [r["completion_tokens"] for r in ok_cell if r["completion_tokens"]]
        times = [r["elapsed_s"] for r in ok_cell]
        reasons = [r["reasoning_len"] for r in ok_cell]
        visibles = [r["visible_len"] for r in ok_cell]

        def m(xs): return statistics.mean(xs) if xs else None
        def s(xs): return statistics.stdev(xs) if len(xs) >= 2 else None

        return {
            "model": model, "setting": setting,
            "n_ok": n_ok, "n_fail": n_fail,
            "tok_mean": m(toks), "tok_std": s(toks),
            "tok_min": min(toks) if toks else None,
            "tok_max": max(toks) if toks else None,
            "time_mean": m(times), "time_std": s(times),
            "reas_mean": m(reasons), "reas_std": s(reasons),
            "visible_mean": m(visibles),
            "all_have_code": all(r["has_code"] for r in ok_cell),
            "errors": list({r["error"] for r in cell if not r["ok"]}),
            "reps": cell,
        }

    aggregates = []
    for model in MODELS:
        for label, _ in SETTINGS:
            aggregates.append(aggregate_cell(model, label))

    # ─────────────────────────────────────────────────────────────
    # Aggregated results table
    # ─────────────────────────────────────────────────────────────

    print("=" * 92)
    print(f" AGGREGATED RESULTS  (N_REPS = {N_REPS} per cell)")
    print("=" * 92)
    print(f"{'model':<18} {'setting':<26} {'n':>3} {'tokens (mean±std)':>20} "
          f"{'time s (mean±std)':>20} {'code':>5}")
    print("-" * 92)
    for a in aggregates:
        if a["n_ok"] == 0:
            err = (a["errors"][0] if a["errors"] else "?")[:50]
            print(f"{a['model']:<18} {a['setting']:<26} {a['n_ok']:>3} "
                  f"{'FAILED':>20} {'-':>20} {'-':>5}  {err}")
            continue
        tok_str = (f"{a['tok_mean']:>6.0f} ± {a['tok_std']:>5.0f}"
                   if a['tok_std'] is not None
                   else f"{a['tok_mean']:>6.0f} ± ?")
        time_str = (f"{a['time_mean']:>6.1f} ± {a['time_std']:>5.1f}"
                    if a['time_std'] is not None
                    else f"{a['time_mean']:>6.1f} ± ?")
        code = "yes" if a["all_have_code"] else "MIX"
        suffix = f"  fail={a['n_fail']}" if a["n_fail"] else ""
        print(f"{a['model']:<18} {a['setting']:<26} {a['n_ok']:>3} "
              f"{tok_str:>20} {time_str:>20} {code:>5}{suffix}")
    print()

    # ─────────────────────────────────────────────────────────────
    # Per-model interpretation — noise-floor-aware
    # ─────────────────────────────────────────────────────────────
    # Classification rules (token-mean basis; time would mirror but with
    # more network jitter so token count is more reliable):
    #
    #   REJECTED:  no successful reps in cell
    #   NOISY:     setting's own coeff. of variation (std/mean) > 0.25
    #              → can't trust the mean as a point estimate
    #   IGNORED:   |Δ vs baseline mean| < max(15% of baseline, 1σ baseline)
    #   HONOURED:  Δ exceeds 2σ of baseline (or 25% of baseline if std is
    #              very small) — direction reported
    #
    # The 2σ threshold is a rough analog of a 95% interval; with N=3 reps
    # the std is itself noisy, so don't over-interpret single-cell results.
    # ─────────────────────────────────────────────────────────────

    def classify(setting_agg, baseline_agg):
        """Return ("HONOURED ↓"/"HONOURED ↑"/"IGNORED"/"NOISY"/"REJECTED", explanation)."""
        if setting_agg["n_ok"] == 0:
            err = (setting_agg["errors"][0] if setting_agg["errors"] else "?")[:60]
            return "REJECTED", err
        m_set = setting_agg["tok_mean"]
        s_set = setting_agg["tok_std"]
        m_bl = baseline_agg["tok_mean"]
        s_bl = baseline_agg["tok_std"]
        if m_bl is None or m_set is None:
            return "?", "missing baseline"

        delta = m_set - m_bl
        delta_pct = delta / m_bl * 100
        # Within-cell coefficient of variation
        cv_set = (s_set / m_set) if (s_set and m_set) else 0
        # Baseline noise band: wider of 1σ baseline or 15% baseline
        baseline_band = max(s_bl or 0, 0.15 * m_bl)
        # "Real effect" threshold: wider of 2σ baseline or 25% baseline
        effect_band = max(2 * (s_bl or 0), 0.25 * m_bl)

        if cv_set > 0.25:
            return "NOISY", (f"within-cell std/mean = {cv_set:.0%} — "
                             f"need more reps to draw a conclusion")
        if abs(delta) < baseline_band:
            return "IGNORED", (f"Δ tok {delta_pct:+.0f}% inside baseline "
                               f"noise band (±{baseline_band/m_bl*100:.0f}%)")
        if abs(delta) >= effect_band:
            arrow = "↓" if delta < 0 else "↑"
            note = ("— param reduces generation, candidate for delv-e wire-in"
                    if delta < 0 and setting_agg["all_have_code"]
                    else "— param drives generation in wrong direction"
                    if delta > 0
                    else "— ⚠ output sometimes lacks code")
            return f"HONOURED {arrow}", f"Δ tok {delta_pct:+.0f}% {note}"
        # In-between zone: effect exceeds noise but not "real effect" threshold
        return "weak", (f"Δ tok {delta_pct:+.0f}% — outside noise band "
                       f"but below 2σ threshold; suggestive, not definitive")

    print("=" * 92)
    print(" INTERPRETATION (each setting vs baseline mean, noise-floor aware)")
    print("=" * 92)
    for model in MODELS:
        baseline_agg = next(
            (a for a in aggregates
             if a["model"] == model and "baseline" in a["setting"]),
            None,
        )
        if not baseline_agg or baseline_agg["n_ok"] == 0:
            print(f"\n  {model}: baseline had 0 successful reps — cannot compute deltas.")
            continue
        m_bl = baseline_agg["tok_mean"]
        s_bl = baseline_agg["tok_std"]
        band_pct = (max(s_bl or 0, 0.15 * m_bl) / m_bl) * 100
        std_str = f"±{s_bl:.0f}" if s_bl is not None else "±?"
        print(f"\n  {model}: baseline mean = {m_bl:.0f} tok {std_str}  "
              f"(noise band ±{band_pct:.0f}%, N={baseline_agg['n_ok']})")
        for a in aggregates:
            if a["model"] != model or "baseline" in a["setting"]:
                continue
            verdict, expl = classify(a, baseline_agg)
            print(f"    {a['setting']:<28} {verdict:<14} {expl}")

    # ─────────────────────────────────────────────────────────────
    # JSON dump for offline inspection (raw reps + aggregates + bodies)
    # ─────────────────────────────────────────────────────────────
    out_path = "ollama_thinking_probe_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump({
                "base_url": f"{BASE_URL}/v1",
                "models": MODELS,
                "settings": [s[0] for s in SETTINGS],
                "n_reps": N_REPS,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "aggregates": aggregates,
                "results": results,
            }, f, indent=2, default=str)
        print(f"\n  Full results (raw reps + aggregates + request bodies) written to: {out_path}")
    except OSError as e:
        print(f"\n  (could not write JSON: {e})")

    print()
    print("Next step: for any 'HONOURED ↓' setting, add an extra_body kwarg to")
    print("OllamaProvider.call in llm.py. Example for reasoning_effort=high:")
    print()
    print('    response = self.client.chat.completions.create(')
    print('        model=model, messages=messages,')
    print('        max_tokens=max_tokens, temperature=temperature,')
    print('        extra_body={"reasoning_effort": "high"} if "kimi" in model else None,')
    print('    )')
    print()


if __name__ == "__main__":
    main()
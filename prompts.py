"""
Prompt templates for delv-e.

Architecture:
  - Lean, task-specific agent prompts (no repeated preamble)
  - Finding Completeness: maturity tracking in Research Model guides PURSUING depth
  - Cross-Finding Connections: periodic connection_explorer generates interaction questions
  - Code prompts: rules in system prompt only; user/error inherit

Templates accessed by auto_explore.py:
  - result_evaluator
  - ideas_explorer_auto
  - question_selector
  - research_model_updater
  - connection_explorer       ← NEW: cross-finding interaction questions
  - exploration_synthesis

Templates used by the engine:
  - code_generator_system
  - code_generator_user
  - error_corrector
  - orientation_system
  - orientation_user
"""


class PromptManager:
    """Container for all prompt templates."""

    # ══════════════════════════════════════════════════
    # FINDING MATURITY TEMPLATE
    # Referenced by research_model_updater, evaluator,
    # and question generator during PURSUING phase.
    # ══════════════════════════════════════════════════

    _MATURITY_STAGES = """Finding maturity stages (each finding progresses through these):
  DETECTED    — Signal found (score 7+). Direction and approximate magnitude known.
  QUANTIFIED  — Rate, significance, and effect size precisely established.
  DECOMPOSED  — Tested across subgroups, percentiles, or time periods. Distribution characterised.
  REGIME-TESTED — Checked for structural breaks or temporal instability at candidate breakpoints.
  COMPLETE    — Ready for cross-finding connection testing. Operationally interpretable.

Maturity advances when evidence for a stage arrives — it does NOT require a dedicated analysis
per stage. A single well-designed analysis can advance multiple stages. If a finding is
contradicted at any stage, drop or downgrade it rather than forcing it through remaining stages."""

    # ══════════════════════════════════════════════════
    # AUTO-EXPLORE AGENT PROMPTS
    # ══════════════════════════════════════════════════

    ideas_explorer_auto = """You are generating analytical questions for a data exploration system.

**CURRENT PHASE: {current_phase}**
{phase_instruction}

If the Exploration Health section shows breadth as LOW, at least 3 of your 5 questions
MUST target unexplored territory — regardless of phase.

**When PURSUING:** All 5 questions must target THE SAME finding. Consult Finding Maturity
to identify the least-mature significant finding, then generate 5 different ways to
advance it to its next stage:
  DETECTED → quantify (rate, magnitude, significance)
  QUANTIFIED → decompose (subgroups, percentiles, distribution cuts)
  DECOMPOSED → regime-test (split at temporal breakpoints, rolling windows)
  REGIME-TESTED → connect (test interaction with other established findings)
Do NOT split questions across different findings during PURSUING.

**Question types:**
- EXPLORE: Open new angles, examine untouched variables. Screening analyses valuable.
- EXPLOIT: Deepen a finding — precision, conditions, robustness, disconfirming evidence.

**Requirements:**
1. Answerable by executing code against the available data
2. Connected to the analytical narrative or opens a genuinely new dimension
3. Could produce a surprise — not just confirming the obvious
4. Does NOT duplicate any question in the "All questions investigated" list

**Avoid:**
- Variations on already-investigated questions
- Questions the research model already answers with high confidence
- Methodological monoculture: if recent analyses share a technique, diversify

**Research Model:**
{research_model}

Return exactly 5 questions, one per numbered line. Plain text only.

1. [first question]
2. [second question]
3. [third question]
4. [fourth question]
5. [fifth question]"""


    result_evaluator = """You are the strategic evaluator for a data exploration system.
Score results, select the best, and guide the exploration's direction.

═══ CONTEXT ═══

**Original question:** {seed_question}

{exploration_state}

**Research Model:**
{research_model}

{solutions_block}

═══ TASK 1: SCORE AND SELECT ═══

Rate each solution 1-10:
- 8-10: Genuine discovery — changes understanding, opens new territory
- 5-7: Solid contribution — useful detail, moderate model update
- 3-4: Incremental — confirms what we suspected
- 1-2: Failed or no new information

Confirming what the research model already states is 3-4 at best.

═══ TASK 2: RECOMMEND PHASE ═══

**MAPPING** when:
- Exploration Health shows LOW breadth
- Current thread is confirmed with no open questions
- Recent scores declining (5-6) — thread exhausted
- Large dataset territory untouched

**PURSUING** when:
- Latest result genuinely changed understanding (not just added precision)
- A surprising finding needs verification
- Thread producing 8+ scores and still changing the narrative
- A finding in the Finding Maturity section has not yet reached DECOMPOSED —
  it needs more depth before the system moves on

Key test: would the thread's conclusions change if the last 3 analyses hadn't
been done? If not, the thread is exhausted — recommend MAPPING.
Exception: if the Finding Maturity section shows an active finding below
DECOMPOSED, recommend PURSUING even if recent scores are moderate — the
finding needs its analytical arc completed.

═══ RESPONSE FORMAT (strict) ═══

SCORES: [comma-separated scores 1-10 for each solution]
SUMMARIES: [one-sentence summary per solution, separated by |, max 150 chars each]
SELECTED: [solution number with highest analytical value]
KEEP_DORMANT: [solution number(s) worth revisiting later (score 6+), or NONE]
REASON: [One sentence — why the selected solution is most valuable]
FOLLOW_UP_ANGLE: [Most promising specific direction for next iteration]
PHASE: [MAPPING / PURSUING]
PHASE_REASONING: [One sentence — why this phase fits breadth, trajectory, AND finding maturity]"""


    research_model_updater = """You are the Research Interpreter for a data exploration system.
Maintain a living research model and monitor the exploration's health.

═══ CONTEXT ═══

**Original question:** {seed_question}

**Current Research Model:**
{current_model}

**Latest Result:**
Question: {question}
Quality score: {score}/10
Findings:
{result_summary}

{parallel_results}

═══ TASK 1: ASSESS IMPACT ═══

MODEL_IMPACT: [HIGH / MEDIUM / LOW]
- HIGH: Changes HOW we understand something. New mechanism, contradiction, reframing.
- MEDIUM: Changes WHAT we know in a way that affects next steps.
- LOW: Confirms or adds precision to an already-characterised finding.

If score ≤5, impact should be LOW unless it contradicts an existing finding.
Back-to-back refinements of the same claim are LOW regardless of score.

CONTRADICTION: [YES / NO]
YES only if the result DIRECTLY REVERSES a claimed relationship direction.

THREAD_COMPLETED: [YES / NO]
YES if further investigation would yield diminishing returns.
A finding is NOT complete until checked against alternative outcome measures
(if available) and tested for confounding by group membership.

MATURITY_ADVANCE: [finding name → new stage, or NONE]
If this result provides evidence that advances a finding's maturity, state which
finding and what stage it advances to. A single result can advance multiple stages
if the evidence covers them (e.g., a subgroup analysis that also reveals a breakpoint).

RESULT_DIGEST: [3-5 lines — the key numbers that matter for ongoing analysis]

═══ TASK 2: UPDATE THE RESEARCH MODEL ═══

UPDATED_MODEL:

## Active Hypotheses
Testable claims the next analysis could strengthen or refute. Max 4.
Graduate to Established Findings when confirmed with high confidence.
Format: - [H#] claim | Confidence: low/medium/high | Evidence: brief

## Established Findings
Confirmed discoveries. Max 10 bullet points, each with at least one key number.
When near the limit, consolidate related findings. Do not drop findings that
active hypotheses, threads, or maturity tracking depend on.
Do not duplicate facts already in the data profile.

## Finding Maturity
Track significant findings (score 7+) through their analytical arc.
Format: - finding name | Stage: DETECTED/QUANTIFIED/DECOMPOSED/REGIME-TESTED/COMPLETE | Next: [specific analytical step needed]

Stage definitions:
  DETECTED    — Signal found. Direction and approximate magnitude known.
  QUANTIFIED  — Rate, significance, effect size precisely established.
  DECOMPOSED  — Tested across subgroups, percentiles, or time periods.
  REGIME-TESTED — Checked for structural breaks at candidate breakpoints.
  COMPLETE    — Operationally interpretable. Ready for connection testing.

Rules:
- Add a finding when it first scores 7+ and enters Established Findings.
- Advance the stage when evidence arrives (state what evidence advanced it).
- A single analysis can advance multiple stages.
- If contradicted, remove the finding from tracking (note in Attention Flags).
- Max 5 tracked findings. Graduate COMPLETE findings out of tracking.
- The "Next" field must be a SPECIFIC analytical step, not vague (e.g.,
  "test across pre/post 2000 split" not "investigate further").

## Threads
Active lines of inquiry. Max 3, each with max 2 open questions.
Format: - thread name | Completeness: low/medium/high | Open: question 1; question 2

## Cross-Finding Connections
Record tested and untested connections between established findings.
Format: - [Finding A] × [Finding B] | Status: untested / tested | Result: [brief]
Connections become ESTABLISHED FINDINGS when confirmed. Max 5 tracked.

## Attention Flags
Findings where a later analysis produced a different direction, >30% magnitude change,
or changed significance. Format:
- Finding | Original: [stat] | Later: [stat] | Status: unresolved
Remove when resolved. If none, write "None".

## Biggest Gap
Single sentence: the most important thing NOT yet investigated.
The gap MUST be answerable by running code against the actual dataset — not a synthesis
question, not an operational recommendation, not a request for external data that doesn't
exist. If the current gap is a "what should we do" question, replace it with "what don't
we know yet" from the Unexplored Territory list.
If unchanged from last update: "NOTE: This gap has persisted — consider pivoting."

## Exploration Health
- Topics investigated: [count + 10 most recent. Variations on same variable = ONE topic.]
- Recent focus: [Last 5-8 analyses. Count how many of last 8 share a theme.]
- Unexplored territory: [Name specific COLUMNS or VARIABLE GROUPS from the data profile
  that have not been analysed. Do not list data that doesn't exist in the dataset.
  Cross-reference against the data profile to identify untouched columns.]
- Breadth: [LOW / MEDIUM / HIGH]
  LOW = 5+ of last 8 share a theme AND scores declining or territory untouched.
  MEDIUM = 3-4 themes in last 8, moderate unexplored territory.
  HIGH = 5+ themes, most major features examined.
- Recommendation: [What to prioritise. If LOW, name specific unexplored directions.]

END_MODEL

Rules:
- Everything between UPDATED_MODEL: and END_MODEL is the complete model.
- Be ruthlessly concise. Every word earns its place.
- Graduate aggressively: confirmed hypotheses → Established Findings.
- Exploration Health must be HONEST. If narrow, say so.
- Plain text only. No LaTeX."""


    question_selector = """Select the most promising questions for a data exploration.

{exploration_history}

**Research Model:**
{research_model}

**Current Phase: {current_phase}**

**Available questions:**
{questions}
{context_hint}

**Selection principles:**
1. If Exploration Health shows LOW breadth, prioritise new territory over refinement.
2. Prefer questions where a surprising answer would most change the research model.
3. Phase alignment: MAPPING → new dimensions; PURSUING → deepen the strongest lead.
4. If Finding Maturity shows a finding below DECOMPOSED, prefer questions that advance it.
5. During PURSUING, all selected questions must target the same finding — do not split
   parallel slots across unrelated threads.
6. Avoid questions similar to low-scoring past attempts.

Select exactly {num_to_select} questions. Respond with ONLY the question numbers,
comma-separated, best first. Nothing else."""


    # ══════════════════════════════════════════════════
    # CROSS-FINDING CONNECTION EXPLORER (NEW)
    # Called periodically by auto_explore.py when
    # established findings ≥ 4 or every N iterations.
    # ══════════════════════════════════════════════════

    connection_explorer = """You are generating connection-testing questions for a data exploration system.

The system has established several independent findings. Your job is to identify
which pairs of findings might INTERACT, COMPOUND, or EXPLAIN each other — and
generate questions that test those connections.

**Established Findings:**
{established_findings}

**Cross-Finding Connections already tested:**
{tested_connections}

**Available Data:**
{data_profile}

═══ WHAT MAKES A GOOD CONNECTION QUESTION ═══

A connection question tests whether two independently-discovered findings are related:
- COMPOUNDING: Do they amplify each other? (e.g., "Is snow loss worse on days that are
  BOTH warm AND windy, beyond what each factor predicts alone?")
- MEDIATING: Does one explain the other? (e.g., "Does the increase in warm NE days
  account for the increase in loss days?")
- CONDITIONAL: Does one modify the other? (e.g., "Is the visitor depth-response curve
  steeper in post-2000 seasons than pre-2000?")
- CONTRADICTING: Do they point in opposite directions for the same outcome?

Bad connection questions:
- Testing a connection that is obvious from the finding definitions
- Repeating an analysis that established one of the findings
- Connections already listed as tested above

═══ REQUIREMENTS ═══

1. Each question must name BOTH findings it connects
2. Each question must be answerable by executing code against the data
3. Prioritise connections where the result would change operational understanding
4. Do NOT duplicate connections already tested

Return exactly {num_questions} questions, one per numbered line. Plain text only.

1. [first connection question]
2. [second connection question]
3. [third connection question]"""


    # ══════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════

    exploration_synthesis = """You are generating a synthesis report from an autonomous data exploration.

Today's date: {0}

{1}

Task: {2}

---

Generate a synthesis of what this exploration discovered. Write for a researcher who
wants to understand WHAT was found and WHY it matters.

THE INPUT HAS FOUR SECTIONS:
- **Section A (Context):** Original question and dataset profile
- **Section B (Findings Index):** One-line summary of every analysis with scores and IDs
- **Section C (Full Evidence):** Complete numerical results for every analysis
- **Section D (Research Model):** Final understanding (may not cover earlier discoveries —
  cross-reference against the Findings Index)

YOUR APPROACH:
1. Scan the Findings Index to identify clusters of related analyses — these are themes.
2. For each theme, read Full Evidence to extract key numbers.
3. Use the Research Model for confidence levels, but don't limit to what it mentions.
4. Findings supported by multiple high-scoring analyses (7+) are more reliable.
5. Pay special attention to Cross-Finding Connections in the Research Model — these
   represent the system's most integrative discoveries.

CRITICAL — SELF-CORRECTION AWARENESS:
This exploration revised its own conclusions as new evidence arrived. Before reporting
ANY finding, apply these checks:

- **Attention Flags first.** Read the Attention Flags section in the Research Model.
  Any finding marked CONTRADICTED must use the LATER corrected values, not the original.
  If the original and correction are both in the evidence, cite only the correction.

- **Later beats earlier.** When two analyses in the same theme give different numbers,
  the LATER analysis (higher chain_id) takes precedence — it had access to prior results.
  If analysis A found "effect X = 0.5" and later analysis B found "effect X = 0.3 after
  controlling for confound Y", report 0.3 and note the confound.

- **Refutation chains.** Some analyses deliberately test whether an earlier finding
  survives a robustness check. If analysis B explicitly refutes or corrects analysis A,
  do NOT report A's original finding as established. Report the corrected version.
  Examples: timing confounds, aggregation artifacts, mediation tests, confound controls.

UNIT VERIFICATION:
Cross-check all numbers against the data profile in Section A. Snow Pack and Snow Depth
Change are in centimetres (cm). Do not write "mm" for values that come from cm-scale data.
When citing thresholds or coefficients, include the unit from the source analysis.

CITATION RULES:
- Every quantitative claim must include [[chain_id]] from the analysis's Reference field.
- RAW NUMBERS in Full Evidence are ground truth. Trust numbers over narrative.
- Every number must appear verbatim in a cited analysis. Do not round or reconstruct.
- Do not merge findings from different analyses without citing all sources.

REPORT STRUCTURE:

## Executive Summary
2-3 sentences: scope, central question, most important conclusion.

## Key Findings
Synthesise the most significant discoveries. Group related analyses into coherent
findings ordered by practical importance. For each:
- State the conclusion with key numbers and [[chain_id]] citations
- Note strength of evidence (how many analyses, score range)

## Cross-Cutting Patterns
2-3 patterns that emerged across multiple themes [[chain_id1]], [[chain_id2]].
Draw especially from confirmed Cross-Finding Connections.

## Limitations & Caveats
Methodological limitations, data gaps, small samples, unresolved confounding.

## Recommended Next Steps
2-3 specific directions. Address the Biggest Gap first if one is identified.

## Conclusion
Integrated narrative tying together key discoveries and implications.

---

IMPORTANT:
- Write about CONCLUSIONS, not individual analyses
- Do NOT build conclusions on NaN or insufficient-data results
- Include ALL major themes from the Findings Index
- Avoid references to system internals (iterations, phases, scores)

COMPLETENESS CHECK — do this AFTER drafting:
Scan the Findings Index for any score-8+ analysis not yet cited. If it represents a
distinct finding not covered in your draft, add it. If it refines a finding you already
reported, incorporate the refinement. Prioritise findings that are OPERATIONALLY
ACTIONABLE (implications for decisions) over those that are purely descriptive."""


    # ══════════════════════════════════════════════════
    # ENGINE-INTERNAL PROMPTS (code generation)
    # ══════════════════════════════════════════════════

    analyst_selector_system_auto = "Auto-explore mode active."

    code_generator_system = """You are an expert data analyst writing Python code to analyse a pandas DataFrame.

RULES:
- `df` is pre-loaded. Do NOT load, create, or redefine it.
- Use matplotlib for plots (call plt.show()). Do NOT use plotly.
- Include all imports at the top.
- Use vectorized pandas operations — not row-level loops.
- Handle nulls: check for NaN before calculations, verify column existence.
- Keep code complete and self-contained. Target 60-120 lines.

DATA QUALITY CHECKS:
- SPARSE COLUMNS: If >90% of non-null values share the same sign, the column likely
  records only events (presence-only). NaN means "not recorded", not "zero". Do NOT use
  notna().sum() as frequency. Note this in the results block.
- AGGREGATION DIRECTION: When comparing group totals on a cumulative metric, report BOTH
  per-observation rate and total. Note if they point in different directions.
- RELATIONSHIP CHANGES: When claiming a relationship changed over time, report BOTH slope
  AND correlation. If slope changed but correlation didn't, it's a scale shift not a
  structural change.

OUTPUT RULES:
- The ONLY print statements should be the results block:
  print("###RESULTS_START###")
  ... computed results ...
  print("###RESULTS_END###")
- No intermediate prints, no decorative formatting, no plot descriptions.
- Results block: ONLY computed numbers. No interpretation, no "this suggests".
- Target 8-15 lines in the results block. Print only headline results.
- INSUFFICIENT DATA threshold: if the dataset has <50 analytical units (e.g. years),
  flag only n<2. Otherwise flag n<5. Always report actual n for small groups.
- If a result is NaN or insufficient, print it explicitly — do NOT skip or interpret around it.

Return ONLY code within ```python``` blocks."""

    code_generator_user = """DataFrame info:
{schema}

Previous findings from this exploration:
{qa_pairs}

Task: {question}

Plan your approach:
1. Which columns are needed? Verify they exist in the schema.
2. Any nulls to handle? What data types?
3. Does previous context flag anomalous periods? If so, compute results with AND without.
   Report both if they diverge materially (direction, significance, or >30% magnitude).

Write complete, executable Python code. Follow the output rules from the system message.

Return code within ```python``` blocks."""

    error_corrector = """The code produced an error during execution.

Error:
{error}

DataFrame schema:
{schema}

Analyse the error:
1. What went wrong? (wrong column name, type mismatch, nulls, missing import?)
2. What fix is needed?

Return the COMPLETE corrected code within ```python``` blocks.
Follow all rules from the system message — especially the results block format."""


    # ══════════════════════════════════════════════════
    # ORIENTATION PHASE PROMPTS
    # ══════════════════════════════════════════════════

    orientation_system = """You are an expert data analyst performing an initial dataset orientation.

Goal: characterise the ANALYTICAL LANDSCAPE — not test hypotheses. Subsequent analyses
should start from knowledge, not assumption. Every observation must end with a practical
ACTION for downstream analysts (what to filter, avoid, derive, or handle specially).

RULES (same as code generation):
- `df` is pre-loaded. Do NOT redefine it. Include imports. Use vectorized ops.
- Handle nulls. Keep code concise (80-150 lines). No visualisations needed.

OUTPUT FORMAT:
Write a compact ANALYTICAL BRIEF in plain English with numbers inline.
Each section 2-4 sentences. No raw dicts, no np.float64 wrappers.

GOOD: "Chemo arm: n=341 (18%). Concentrated in Basal (67%) and Her2 (45%),
sparse in LumA (8%, n=26). → Exclude NC from subgroup analysis."

ANALYSIS DIMENSIONS (adapt to dataset — skip sections that don't apply):

1. COVERAGE MAP: Profile each column's non-null coverage (% of rows). Flag columns
   below 50% as SPARSE — these likely record only specific conditions (e.g., equipment
   active, event occurring). State what the NaN rows probably mean for each sparse column
   (e.g., "not recorded" vs "zero" vs "not applicable").
   → For each sparse column, state: what downstream analyses should assume about NaN rows.

2. OUTCOME LANDSCAPE: Likely target variable(s), distribution, where outcomes concentrate.
   → State required filters (e.g., "restrict to in-season rows for snow/visitor analysis").

3. GROUP SIZES: How groups break down. Cross-tabulate main groupings. Name thin cells.
   → Name specific infeasible comparisons and any empty cells in cross-tabulations.

4. CONFOUNDING MAP: Which variables are entangled (Cramér's V >0.3 or >20pp differences).
   CRITICAL: Before computing correlations between two columns, check their coverage overlap.
   If both columns are sparse (<50%) and their non-null rows substantially overlap (>80%
   of the sparser column's non-nulls also have non-null in the other), the correlation is
   a SELECTION ARTIFACT — it reflects shared operational context (e.g., both measured only
   when equipment is running), not a genuine relationship. Report these separately as
   "coverage-driven correlations" with the overlap percentage, NOT as confounds.
   → For genuine confounds, state the direction and what analyses they could bias.

5. VARIABLE STRUCTURE: Near-redundant variables (r>0.85 on FULLY-COVERED columns only),
   derived variables, clusters.
   → Identify DERIVABLE variables not present in the data but computable: day-over-day
   differences, ratios, rolling averages, cumulative sums. Name the source columns and
   what the derived variable would represent analytically.
   → Flag circular variables (e.g., wind direction in degrees) that need special handling:
   "bin by quadrant, do not use linear regression or Pearson correlation."

6. POWER BOUNDARIES: Based on the group sizes above, state what depth of analysis is
   feasible. Name specific feasible comparisons and specific infeasible ones.
   → If stratification creates empty cells, recommend alternative splits (e.g., "use
   pre/post breakpoint instead of decade stratification to avoid empty cells").

Use these delimiters:
print("###PROFILE_START###")
... analytical brief ...
print("###PROFILE_END###")"""

    orientation_user = """DataFrame info:
{schema}

Seed question for this exploration:
{seed_question}

Write Python code to profile this dataset's analytical landscape. Do NOT test hypotheses.
The column-level schema above already provides types, nulls, ranges, and samples.
Compute CROSS-VARIABLE relationships invisible from individual columns:

1. Coverage: profile every column's non-null percentage. Identify sparse columns (<50%).
   For sparse columns, check what their NaN rows represent.
2. Groups: which are large enough for comparison? Where are the thin/empty cells?
3. Confounds: which variables are entangled? BEFORE correlating sparse columns, check
   whether their non-null rows overlap — if so, the correlation is a coverage artifact.
4. Structure: near-redundant columns? Derivable variables (diffs, ratios, rolling)?
   Circular variables needing special handling?
5. Power: what comparisons are feasible? Where do stratifications create empty cells?

Every observation must end with a practical action: what to filter, avoid, derive, or
handle specially. Output: a compact analytical brief (30-second scan). Plain English
with numbers inline.

Return code within ```python``` blocks using ###PROFILE_START### / ###PROFILE_END### delimiters."""
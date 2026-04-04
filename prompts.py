"""
Prompt templates for delv-e.

Architecture:
  - Lean, task-specific agent prompts (no repeated preamble)
  - Finding Completeness: maturity tracking in Research Model guides PURSUING depth
  - Strategic Review: premium model runs every iteration to maintain strategic coherence,
    enforce commitments, detect missed opportunities, and surface untested connections
  - Code prompts: rules in system prompt only; user/error inherit

Templates accessed by auto_explore.py:
  - result_evaluator
  - ideas_explorer_auto
  - question_selector
  - research_model_updater
  - seed_decomposition            ← premium model: focused first question + initial trajectory
  - strategic_review              ← premium model: commitment, trajectory, missed opportunities
  - reframing_probe               ← premium model: full-results review on thread abandonment
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

{strategic_direction}

Consult the **Strategic Trajectory** section in the Research Model for the current
commitment and planned investigation sequence. All questions should align with the
CURRENT COMMITMENT stated there.

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

METHOD_USED: [One phrase describing the analytical technique — e.g. "rolling correlation
by decade", "logistic regression with interaction term", "binned comparison pre/post 2000".
This helps the strategic reviewer detect methodological monoculture.]

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

## Strategic Trajectory
<<< DO NOT MODIFY THIS SECTION — it is maintained by a separate strategic review process.
Copy it through EXACTLY as-is. If this section does not exist yet, leave this placeholder. >>>

END_MODEL

Rules:
- Everything between UPDATED_MODEL: and END_MODEL is the complete model.
- Be ruthlessly concise. Every word earns its place.
- Graduate aggressively: confirmed hypotheses → Established Findings.
- Exploration Health must be HONEST. If narrow, say so.
- The Strategic Trajectory section is READ-ONLY. Copy it verbatim. Do not edit, summarise, or remove it.
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
    # SEED DECOMPOSITION (premium model)
    # Called once before the main loop to convert a
    # broad research agenda into a focused first analysis.
    # ══════════════════════════════════════════════════

    seed_decomposition = """You are planning the first analysis for an autonomous data exploration system.
The user has provided a research agenda — potentially broad and multi-faceted. Your job
is to decompose it into a focused first analysis and a strategic plan.

**Research Agenda:**
{seed_question}

**Dataset Profile:**
{data_profile}

**Iteration Budget:** {max_iterations} iterations available.
Each iteration produces 2 parallel analyses, so plan for ~{max_iterations} analytical steps total.
Scale the plan accordingly — a 10-iteration run should focus on 2-3 core threads,
a 50+ iteration run can afford deeper investigation and more threads.

═══ TASK 1: FIRST ANALYSIS ═══

Write a single, focused analytical question that the code generator should tackle first.
This should be the FOUNDATION — the analysis whose results every subsequent investigation
will build on. Typically this means establishing baseline trends and distributions before
testing mechanisms or interactions.

Rules:
- ONE focused question, answerable in 60-120 lines of Python
- Should cover at most 2-3 closely related variables
- Must produce precise numbers (slopes, p-values, effect sizes)
- Should NOT attempt to cover the entire research agenda
- Prefer descriptive baselines over hypothesis tests for the first pass

═══ TASK 2: INITIAL STRATEGIC TRAJECTORY ═══

Decompose the full research agenda into a logical sequence of investigation arcs.
Order them by dependency — what needs to be established before what can be tested.

IMPORTANT: Plan a brief initial MAPPING phase (5-8 iterations) to establish core
baselines, then expect the strategic review to INTERLEAVE mapping and pursuing.
Each arc should take 3-5 iterations of mapping followed by 2-3 iterations of
pursuing the best discovery from that arc. Do NOT plan all mapping upfront
as one long block. The strategic review will adapt the plan based on what the
exploration actually discovers.

Structure:
- FULL AGENDA: [one-line summary of each thread in the research agenda]
- CURRENT COMMITMENT: MAPPING — [what the first 5-8 iterations should establish]
- NEXT AFTER COMMITMENT: [what to pursue once initial baselines are established]

═══ RESPONSE FORMAT (strict) ═══

FIRST_QUESTION: [the focused analytical question]
INITIAL_TRAJECTORY:
[the strategic trajectory content]
END_TRAJECTORY"""


    # ══════════════════════════════════════════════════
    # STRATEGIC REVIEW (premium model)
    # Called every iteration to maintain strategic
    # coherence, enforce commitments, and detect
    # missed opportunities including untested connections.
    # ══════════════════════════════════════════════════

    strategic_review = """You are the strategic reviewer for an autonomous data exploration system.
You are the most capable model in the system. The day-to-day exploration is run by
smaller, faster models. Your job is to maintain strategic coherence across the full arc
of the investigation — deciding WHEN to go deep, WHEN to go broad, and WHAT to explore next.

**Original question:** {seed_question}

**Iteration:** {iteration} of {max_iterations} ({remaining_iterations} remaining)

**Dataset Profile:**
{data_profile}

**Research Model:**
{research_model}

**Recent iterations (last 5):**
{recent_context}

═══ TASK 1: COMMITMENT CHECK ═══

The Strategic Trajectory (in the research model) states the current commitment.
Evaluate whether it should hold, pivot, or be abandoned.

BUDGET AWARENESS:
You have {remaining_iterations} iterations remaining. Use them. When all planned arcs
are complete but significant budget remains, do NOT declare the investigation finished.
Instead:
- Identify findings that were established at DETECTED or QUANTIFIED but never deepened
  to DECOMPOSED (robustness checks, subgroup decomposition, alternative framings)
- Decompose aggregate-level findings into finer-grained components to test whether
  the mechanism is uniform or varies across subgroups, conditions, or strata
- Test the operational implications of established findings (thresholds, breakpoints,
  actionable metrics for decision-makers)
- Run bootstrap or permutation validation on the 3-5 most important findings
- Explore columns or variable interactions flagged in the data profile but not yet tested
Do NOT spend remaining iterations on report writing, summary tables, or narrative
synthesis. These are handled by a dedicated synthesis agent after exploration ends.

HOLD when:
- The pursued finding is advancing in maturity (scores stable or improving)
- The finding has not yet reached DECOMPOSED — it needs more depth
- Recent scores are moderate (5-6) but the finding clearly has substance
  (moderate scores during deep pursuit are EXPECTED — this is NOT exhaustion)

PIVOT when:
- A clearly higher-value thread has emerged from recent results
- The current thread has stalled: 3+ iterations with no maturity advance AND
  the finding is already at DECOMPOSED or beyond
- A surprise finding (score 8+) opens a more important direction

ABANDON when:
- The pursued finding has been directly contradicted
- Results show the data cannot support further investigation of this thread
- The thread has reached COMPLETE

On PIVOT or ABANDON, you MUST provide a NEXT_DIRECTION — a specific framing for the
next thread of exploration. Name the variables, the analytical question, and WHY this
direction has high expected value. The question generator will use this as its primary
constraint.

PHASE rule:
- PURSUING = go deep on a specific finding (all questions target the same thread)
- MAPPING = go broad across unexplored territory

TRAJECTORY AWARENESS:
The initial trajectory is a PLAN, not a binding contract. When a MAPPING iteration
produces a high-value discovery (score 8+ with clear operational significance), you
should switch to PURSUING for 2-3 iterations to deepen it before returning to the
next incomplete arc. The trajectory records which arcs are complete and which remain.
After a brief PURSUING detour, return to the next incomplete arc in the trajectory.

Do NOT stay in MAPPING for 15+ consecutive iterations just because the trajectory
planned a long mapping phase. Interleave: map a topic, pursue its best finding for
2-3 iterations, return to mapping. This produces findings at DECOMPOSED maturity
rather than leaving every topic at DETECTED.

Conversely, do NOT abandon the trajectory for extended PURSUING. If a PURSUING
detour exceeds 4 iterations without the finding reaching DECOMPOSED, return to
MAPPING. The trajectory's incomplete arcs represent genuine analytical territory
that must eventually be covered.

═══ TASK 2: MISSED OPPORTUNITIES ═══

Scan the Exploration Health section and the data profile. Name any specific
unexplored angles the smaller models appear to be overlooking — columns,
variable groups, or analytical techniques not yet tried on promising threads.
If the recent iterations show methodological monoculture (same technique
repeated), name a specific alternative technique.

Also review the Cross-Finding Connections section. If 2+ established findings
have NOT been tested for interaction, note the most promising untested pair.

═══ TASK 3: TRAJECTORY UPDATE ═══

Rewrite the Strategic Trajectory section. This is the exploration's strategic
memory — it records WHY pivots happened and WHAT the current commitment is.
Structure:
- 1-2 lines per completed arc (iterations N-M: what was pursued, what was found,
  why the system moved on)
- CURRENT COMMITMENT: [phase] on [thread] because [reason]
- NEXT AFTER COMMITMENT: [direction with highest expected value]
  If untested cross-finding connections exist and would have high analytical value,
  name the most important one here. Connection testing is a valid next direction.

═══ RESPONSE FORMAT (strict) ═══

COMMITMENT: [HOLD / PIVOT / ABANDON] — [one sentence reason]
PHASE: [MAPPING / PURSUING]
NEXT_DIRECTION: [specific framing for next thread, or UNCHANGED if HOLD]
PROBE_NEEDED: [YES / NO] — YES when the raw analytical output deserves a second
  look from a different angle. This includes:
  (a) A null result that seems suspicious given the broader investigation narrative
  (b) A positive finding where you can name a SPECIFIC distributional feature,
      threshold, or decomposition that the current analysis likely missed
      (e.g., "the outcome-predictor scatter probably saturates above a threshold"
      or "the seasonal variance likely differs between eras")
  (c) A thread completing where you can identify a SPECIFIC derived metric that
      would be more operationally useful than what was tested
  Say NO for routine completions where the finding is clean and fully captured,
  and NO when you have only a vague sense that "this could be sharper" without
  a concrete alternative in mind. Most iterations should be NO.
  Expect YES roughly 5-8 times per 100 iterations.
MISSED: [specific missed opportunities or untested connections, or NONE]
UPDATED_TRAJECTORY:
[full rewrite of Strategic Trajectory section]
END_TRAJECTORY"""


    # ══════════════════════════════════════════════════
    # REFRAMING PROBE (premium model)
    # Fires when strategic review sets PROBE_NEEDED: YES.
    # Can trigger on any commitment (HOLD, PIVOT, ABANDON).
    # Receives full analytical output, not digests.
    # ══════════════════════════════════════════════════

    reframing_probe = """You are reviewing the raw analytical output from a data exploration.
The strategic review flagged this moment as one where a fresh look at the actual
numbers could reveal something the standard analysis missed.

Your job is NOT to rerun the analyses. It is to READ THE ACTUAL NUMBERS below and
notice what the headline statistics missed: distributional shifts, variance changes,
threshold effects, saturation patterns, era differences, outlier clustering, or any
pattern that suggests an alternative framing would produce a sharper or more
operationally useful finding.

**Original research agenda:** {seed_question}

**Current thread:** {thread_summary}

**Why a fresh look matters here:** {why_it_matters}

**Full analytical output from recent analyses:**

{full_results}

═══ YOUR TASK ═══

Read the numbers above carefully. Then answer three questions:

1. HIDDEN PATTERN: What pattern, threshold, regime change, or distributional feature
   in these numbers does the headline test NOT capture? Look for: variance changes
   across eras, saturation/threshold effects in scatter relationships, clustering of
   outliers in specific conditions, changes in distribution shape even if the mean
   is stable, decompositions (e.g., counting days above/below a threshold vs
   testing a continuous mean), or operational thresholds where a relationship
   changes character (e.g., an outcome variable flattening above a predictor threshold).

2. ALTERNATIVE FRAMING: What specific metric, decomposition, or analytical approach
   would produce a sharper or more operationally useful finding than what the
   standard tests captured? Define the metric precisely enough that a code generator
   can implement it (e.g., "count the number of days per season where [metric] exceeds
   [threshold] and test this count as a predictor of [outcome]").

3. NEXT QUESTION: Write one specific analytical question that the code generator
   should tackle, using your alternative framing.

If the null result is genuine and no alternative framing is warranted, say NONE for
all three fields. Do not force a finding.

═══ RESPONSE FORMAT ═══

HIDDEN_PATTERN: [what you noticed in the raw numbers, or NONE]
ALTERNATIVE_FRAMING: [the specific metric/approach, or NONE]
REFRAMING_DIRECTION: [the analytical question to pursue, or NONE]"""


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
Cross-check all numbers against the data profile in Section A. When citing thresholds
or coefficients, include the unit from the source analysis. Do not change units from
what was reported in the original analysis (e.g., do not write "mm" for cm-scale data).

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
- Write direct, linear code. Do NOT build elaborate reusable functions or
  classes. The code runs once. Inline the computation and print results
  as you go. This keeps the code short and avoids output truncation.

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
- Target 15-25 lines in the results block.
- INSUFFICIENT DATA threshold: if the dataset has <50 analytical units (e.g. years),
  flag only n<2. Otherwise flag n<5. Always report actual n for small groups.
- If a result is NaN or insufficient, print it explicitly — do NOT skip or interpret around it.

DISTRIBUTIONAL DETAIL (important for downstream analysis):
- For every group mean, also report std and range (min, max).
  GOOD: "decade_1990: mean=253, std=113, range=30-397, n=10"
  BAD:  "decade_1990: mean=253"
- When testing a linear trend, also test for VARIANCE CHANGE between the first
  and second half of the series (Levene test or F-test for equality of variances).
  Report both the trend slope/p and the variance-change p-value.
- When comparing groups, report the full distribution (mean, std, min, max, n)
  for each group, not just means or effect sizes.
- When testing a relationship (correlation, regression), also report
  top-3 and bottom-3 values with their identifiers (year, season, condition).

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
   → State required filters (e.g., "restrict to rows where [condition] for [domain] analysis").

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
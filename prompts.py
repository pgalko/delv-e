"""
Prompt templates for delv-e.

Architecture:
  - Lean, task-specific agent prompts (no repeated preamble)
  - Finding Completeness: maturity tracking in Research Model guides investigation depth
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
  - reframing_probe               ← premium model: full-results review when arc needs fresh angle
  - perspective_rotation           ← premium model: alternative lenses on completed arcs
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
    # and question generator.
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

{commitment_instruction}

Consult the **Strategic Trajectory** section in the Research Model for the current
commitment and planned investigation sequence. All questions should align with the
CURRENT COMMITMENT stated there.

If the Exploration Health section shows breadth as LOW, at least 3 of your 5 questions
MUST target unexplored territory.

**When the commitment is HOLD on a specific finding:** All 5 questions must target
THE SAME finding. Consult Finding Maturity to identify the least-mature significant
finding, then generate 5 different ways to advance it to its next stage:
  DETECTED → quantify (rate, magnitude, significance)
  QUANTIFIED → decompose (subgroups, percentiles, distribution cuts)
  DECOMPOSED → regime-test (split at temporal breakpoints, rolling windows)
  REGIME-TESTED → connect (test interaction with other established findings)
Do NOT split questions across different findings when holding.

**When pivoting to new territory:** Generate diverse questions that survey different
aspects of the new direction. Cover multiple angles and variables.

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

═══ TASK: SCORE AND SELECT ═══

Rate each solution 1-10:
- 8-10: Genuine discovery — changes understanding, opens new territory
- 5-7: Solid contribution — useful detail, moderate model update
- 3-4: Incremental — confirms what we suspected
- 1-2: Failed or no new information

Confirming what the research model already states is 3-4 at best.
Diminishing returns: if the Research Model already has high confidence on
the topic and Finding Maturity shows DECOMPOSED or beyond, score for
novelty — precision refinements on exhausted arcs are LOW value.

═══ RESPONSE FORMAT (strict) ═══

SCORES: [comma-separated scores 1-10 for each solution]
SUMMARIES: [one-to-two sentence summary per solution, separated by |]
SELECTED: [solution number with highest analytical value]
REASON: [One sentence — why the selected solution is most valuable]
FOLLOW_UP_ANGLE: [Most promising specific direction for next iteration]"""


    research_model_updater = """You are the Research Interpreter for a data exploration system.
Maintain a living research model and monitor the exploration's health.

═══ CONTEXT ═══

**Original question:** {seed_question}

**Available columns (for Exploration Health cross-reference):**
{column_list}

**Current Research Model:**
{current_model}

**Latest Result:**
Question: {question}
Quality score: {score}/10
Findings:
{result_summary}

═══ TASK 1: ASSESS IMPACT ═══

MODEL_IMPACT: [HIGH / MEDIUM / LOW]
- HIGH: Changes HOW we understand something. New mechanism, contradiction, reframing.
- MEDIUM: Changes WHAT we know in a way that affects next steps.
- LOW: Confirms or adds precision to an already-characterised finding.

If score ≤5, impact should be LOW unless it contradicts an existing finding.
Back-to-back refinements of the same claim are LOW regardless of score.

CONTRADICTION: [YES / NO]
YES if the result DIRECTLY REVERSES a claimed relationship direction, OR if the
proposed explanation for a finding invokes a mechanism that Established Findings
show to be non-significant or absent. Check before accepting any causal claim.

ARC_EXHAUSTED: [YES / NO]
YES if further investigation of the current arc would yield diminishing returns.
A finding is NOT complete until checked against alternative outcome measures
(if available) and tested for confounding by group membership.

MATURITY_ADVANCE: [finding name → new stage, or NONE]
If this result provides evidence that advances a finding's maturity, state which
finding and what stage it advances to. A single result can advance multiple stages
if the evidence covers them (e.g., a subgroup analysis that also reveals a breakpoint).

RESULT_DIGEST: [3-5 lines — the key numbers that matter for ongoing analysis]

METHOD_USED: [One phrase describing the analytical technique — e.g. "rolling correlation
by decade", "logistic regression with interaction term", "binned comparison pre/post breakpoint".
This helps the strategic reviewer detect methodological monoculture.]

═══ TASK 2: UPDATE THE RESEARCH MODEL ═══

UPDATED_MODEL:

## Active Hypotheses
Testable claims the next analysis could strengthen or refute. Max 4.
Graduate to Established Findings when confirmed with high confidence.
Format: - [H#] claim | Confidence: low/medium/high | Evidence: brief

## Established Findings
Confirmed discoveries from simulation. Max 10 bullet points, each with at least one
key number. When near the limit, consolidate related findings. Do not drop findings
that active hypotheses or maturity tracking depend on.
Do not duplicate facts already in the data profile.
The model may also contain [PUBLISHED] entries from external literature searches.
These are managed by a separate integration process and re-inserted automatically —
you MUST NOT output any [PUBLISHED] entries in Established Findings or elsewhere,
and MUST NOT output STATUS lines. Pretend they don't exist in your output.
You may reference them in Cross-Finding Connections when a simulation directly
tests a published claim (e.g., "our result confirms the published prediction"),
but do not duplicate or rewrite the published entries themselves.

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
  "test across pre/post breakpoint split" not "investigate further").

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
- Unexplored territory: [Name specific COLUMNS or VARIABLE GROUPS from the Available
  columns list above that have not been analysed. Do not list data that doesn't exist
  in the dataset. Cross-reference against the column list to identify untouched columns.]
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

{commitment_context}

**Available questions:**
{questions}
{context_hint}

**Selection principles:**
1. If Exploration Health shows LOW breadth, prioritise new territory over refinement.
2. Prefer questions where a surprising answer would most change the research model.
3. If pivoting to new territory, prefer breadth and diversity across selected questions.
4. If holding on a finding, prefer depth — all selected questions should target that finding.
5. If Finding Maturity shows a finding below DECOMPOSED, prefer questions that advance it.
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
Scale the plan accordingly — a 10-iteration run should focus on 2-3 core arcs,
a 50+ iteration run can afford deeper investigation and more arcs.

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
After 1-2 baseline iterations at the aggregate level, prioritise fine-grained
analysis (e.g., daily rows, individual records) over aggregate-level index or
category analyses — fine-grained data typically has 50-100x more statistical power
and reveals mechanisms invisible at the aggregate level.

IMPORTANT: Plan a brief initial survey (5-8 iterations) to establish core
baselines, then expect the strategic review to interleave survey and deep pursuit.
Each arc should take 3-5 iterations of survey followed by 2-3 iterations of
pursuing the best discovery from that arc. Do NOT plan all survey iterations upfront
as one long block. The strategic review will adapt the plan based on what the
exploration actually discovers.

Structure:
- FULL AGENDA: [one-line summary of each topic in the research agenda]
- CURRENT COMMITMENT: HOLD — initial survey of [what the first 5-8 iterations should establish]
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

BUDGET PLANNING: When you set ARC_COMPLETE: YES, the system automatically pursues
one alternative analytical perspective for 1-2 iterations before moving to the next
arc. Budget accordingly: each arc completion adds ~2 perspective iterations.
Plan your trajectory with {remaining_iterations} iterations left and N arcs remaining,
budget roughly N × (arc iterations + 2 perspective) to ensure coverage.

HOLD when:
- The pursued finding is advancing in maturity (scores stable or improving)
- The finding has not yet reached DECOMPOSED — it needs more depth
- Recent scores are moderate (5-6) but the finding clearly has substance
  (moderate scores during deep pursuit are EXPECTED — this is NOT exhaustion)

PIVOT when:
- A clearly higher-value arc has emerged from recent results
- The current arc has stalled: 3+ iterations with no maturity advance AND
  the finding is already at DECOMPOSED or beyond
- A surprise finding (score 8+) opens a more important direction
- An analysis required subgroups with n<5 and produced a null — one null is
  enough when the data cannot support the test. Check the data profile

ABANDON when:
- The pursued finding has been directly contradicted
- Results show the data cannot support further investigation of this arc
- The arc has reached COMPLETE

On PIVOT or ABANDON, you MUST provide a NEXT_DIRECTION — a specific framing for the
next arc of exploration. Name the variables, the analytical question, and WHY this
direction has high expected value. The question generator will use this as its primary
constraint.

TRAJECTORY AWARENESS:
The initial trajectory is a PLAN, not a binding contract. When a HOLD iteration
produces a high-value discovery (score 8+ with clear operational significance), you
should maintain HOLD for 2-3 iterations to deepen it before moving on.

When holding on a finding, do NOT hold for more than 4 iterations without the finding
reaching DECOMPOSED. The trajectory's incomplete arcs represent genuine analytical
territory that must eventually be covered.

═══ TASK 2: MISSED OPPORTUNITIES ═══

Scan the Exploration Health section and the data profile. Name any specific
unexplored angles the smaller models appear to be overlooking — columns,
variable groups, or analytical techniques not yet tried on promising arcs.
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
- CURRENT COMMITMENT: [HOLD/PIVOT/ABANDON] on [arc] because [reason]
- NEXT AFTER COMMITMENT: [direction with highest expected value]
  If untested cross-finding connections exist and would have high analytical value,
  name the most important one here. Connection testing is a valid next direction.

═══ RESPONSE FORMAT (strict) ═══

COMMITMENT: [HOLD / PIVOT / ABANDON] — [one sentence reason]
NEXT_DIRECTION: [specific framing for next arc, or UNCHANGED if HOLD]
PROBE_NEEDED: [YES / NO] — YES when the raw analytical output deserves a second
  look from a different angle. This includes:
  (a) A null result that seems suspicious given the broader investigation narrative
  (b) A positive finding where you can name a SPECIFIC distributional feature,
      threshold, or decomposition that the current analysis likely missed
      (e.g., "the outcome-predictor scatter probably saturates above a threshold"
      or "the variance likely differs between the first and second half of the series")
  (c) An arc completing where you can identify a SPECIFIC derived metric that
      would be more operationally useful than what was tested
  Say NO for routine completions where the finding is clean and fully captured,
  and NO when you have only a vague sense that "this could be sharper" without
  a concrete alternative in mind. Most iterations should be NO.
  Expect YES roughly 5-8 times per 100 iterations.
ARC_COMPLETE: [YES / NO] — Only on ABANDON. YES when this arc produced established
  findings and reached a genuine conclusion. NO when abandoning due to contradiction,
  insufficient data, or failure. When YES, the system will automatically generate and
  pursue an alternative analytical perspective for 1-2 iterations before moving to the
  next arc. Account for this in your budget planning: each arc completion adds ~2
  perspective iterations.
EARLY_STOP: [YES / NO] — YES ONLY when ALL of the following are true:
  (a) Exploration Health shows no unexplored territory
  (b) All Finding Maturity items are COMPLETE or stalled with no viable next step
  (c) The last 5+ iterations have been ABANDON with no productive new directions
  (d) The Biggest Gap requires external data not in the dataset
  This is IRREVERSIBLE — the system will skip remaining iterations and proceed
  directly to synthesis. Most runs should NEVER trigger this. Say NO unless you
  are genuinely certain that further iterations cannot produce new findings.
  Expect YES at most once per run, typically after 60-80% of budget is used.
SEARCH_NEEDED: [specific search query / NONE] — Web search for published literature.
  Only executed on PIVOT or ABANDON iterations (HOLD requests are skipped).
  Useful when:
  (a) Pivoting into a new domain where established theory would save iterations
  (b) An arc produced a surprising result worth checking against published work
  (c) Nearing synthesis and key findings should be validated
  Use a specific academic query, not a generic topic.
  Say NONE when the research model already has relevant [PUBLISHED] entries.
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
threshold effects, saturation patterns, period-to-period differences, outlier clustering, or any
pattern that suggests an alternative framing would produce a sharper or more
operationally useful finding.

**Original research agenda:** {seed_question}

**Current arc:** {arc_summary}

**Why a fresh look matters here:** {why_it_matters}

**Full analytical output from recent analyses:**

{full_results}

═══ YOUR TASK ═══

Read the numbers above carefully. Then answer three questions:

1. HIDDEN PATTERN: What pattern, threshold, regime change, or distributional feature
   in these numbers does the headline test NOT capture? Look for: variance changes
   across time periods, saturation/threshold effects in scatter relationships, clustering of
   outliers in specific conditions, changes in distribution shape even if the mean
   is stable, decompositions (e.g., counting observations above/below a threshold vs
   testing a continuous mean), or operational thresholds where a relationship
   changes character (e.g., an outcome variable flattening above a predictor threshold).

2. ALTERNATIVE FRAMING: What specific metric, decomposition, or analytical approach
   would produce a sharper or more operationally useful finding than what the
   standard tests captured? Define the metric precisely enough that a code generator
   can implement it (e.g., "count the number of observations per period where [metric] exceeds
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
    # PERSPECTIVE ROTATION (premium model)
    # Fires when an original arc completes. Generates
    # alternative analytical lenses on the same phenomenon.
    # Does NOT fire on perspective arcs (no recursion).
    # ══════════════════════════════════════════════════

    perspective_rotation = """You are reviewing a completed investigation arc from an autonomous
data exploration. The arc approached its topic through one analytical lens and reached
a conclusion. Your job is to identify fundamentally DIFFERENT analytical perspectives
on the same phenomenon that the original approach did not take.

This is NOT about going deeper with the same approach. It is about asking different
KINDS of questions about the same subject.

Example of what this means:
- Original arc investigated "what factors cause outcome Y to decline" (mechanistic perspective)
- A different perspective: "how frequently do positive vs negative outcome events occur,
  regardless of cause?" (event counting perspective)
- Another: "has the system's capacity to convert input X into outcome Y changed?"
  (efficiency perspective)
- Another: "what distinguishes cases where identical inputs produced very different
  outcomes?" (residual analysis perspective)

Each perspective leads to different metrics, different analytical questions, and
potentially different conclusions about the same underlying phenomenon.

**Original research agenda:** {seed_question}

**Completed arc:** {arc_name}
**Methods used in this arc:** {arc_methods}
**Key findings from this arc (these define the frame you must ESCAPE, not extend):** {arc_findings}

**Dataset columns available:** {available_columns}

**Previously selected perspectives (DO NOT regenerate these):** {previously_selected}

═══ YOUR TASK ═══

Propose 2-3 fundamentally different analytical perspectives on the same phenomenon
this arc investigated. For each:

1. Name the perspective in 2-4 words
2. Explain in one sentence how it differs from the original approach
3. Propose one specific analytical question a code generator could tackle

DIFFERENTIATION TEST — apply to each perspective before including it:
Describe the completed arc as "[OUTCOME] measured as a function of [PREDICTORS]
using [METHOD]." Your perspective must change the OUTCOME or the METHOD. Changing
only the predictors while keeping the same outcome and method is NOT a different
perspective. Examples of genuine changes:
- Continuous outcome → count events above/below a threshold
- Regression on predictors → ratio between opposing event types
- Individual observations → classify into types, track type frequency over time
- Complex model → simple derived metric (a count, a ratio, a rate)

Rules:
- Each perspective must use columns available in the dataset
- Prefer perspectives that produce simple, trackable metrics
  (counts, ratios, thresholds) over complex model-based approaches
- Do NOT regenerate perspectives listed in "Previously selected" above,
  or close variants with different names but the same analytical approach
- Rank from most to least promising — PERSPECTIVE_1 should be the one
  most likely to produce a finding the original arc missed
- If the arc's findings are self-contained and no meaningful alternative
  perspective exists (or all good perspectives have already been used), say NONE

═══ RESPONSE FORMAT ═══

PERSPECTIVE_1: [2-4 word name]
DIFFERS: [one sentence]
QUESTION: [specific analytical question]

PERSPECTIVE_2: [2-4 word name]
DIFFERS: [one sentence]
QUESTION: [specific analytical question]

PERSPECTIVE_3: [2-4 word name]
DIFFERS: [one sentence]
QUESTION: [specific analytical question]

Or: NONE"""


    # ══════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════

    exploration_synthesis = """You are generating a synthesis report from an autonomous data exploration.

Today's date: {today_date}
{synthesis_context}

Task: {task}

---

Generate a synthesis of what this exploration discovered. Write for an intelligent
non-specialist who is comfortable with careful reasoning but not with dense technical
jargon, specialist domain language, or statistical shorthand.
The report should still be rigorous, but the opening must be readable, relatable,
and understandable to a general audience.

THE INPUT HAS FOUR SECTIONS:
- **Section A (Context):** Original question and dataset profile
- **Section B (Findings Index):** One-line summary of every analysis with scores and IDs
- **Section C (Research Model):** Final synthesised understanding — established findings, maturity tracking, cross-finding connections, strategic trajectory. Read this FIRST to understand the investigation's narrative before examining raw evidence.
- **Section D (Evidence):** Raw numerical results, score-gated: full results for score 8+ analyses (the key findings), summaries only for score 6-7, score ≤5 omitted (see Findings Index for completeness). Every quantitative claim must trace to Section D.

YOUR APPROACH:
1. Read the Research Model (Section C) to understand the investigation's arc and conclusions.
2. Scan the Findings Index (Section B) to identify themes and any score-8+ analyses not covered by the Research Model.
3. For each theme, read Full Evidence (Section D) to extract and verify key numbers.
4. Findings supported by multiple high-scoring analyses (7+) are more reliable.
5. Pay special attention to Cross-Finding Connections in the Research Model — these represent the system's most integrative discoveries.
6. Check [PUBLISHED] entries in the Research Model. These are claims from published
   scientific literature with STATUS annotations (CONFIRMED, CONTRADICTED, or UNTESTED)
   based on this investigation's simulation results. When reporting a key finding:
   - If it CONFIRMS a published prediction, note this: "consistent with [author/theory]"
   - If it CONTRADICTS a published prediction, highlight the disagreement — this is
     potentially the most novel contribution of the investigation
   - UNTESTED published claims belong in The Open Question or Methodological Caveats
   Published literature references add credibility and context. Use them.

CRITICAL — SELF-CORRECTION AWARENESS:
This exploration revised its own conclusions as new evidence arrived.
Before reporting ANY finding, apply these checks:

- **Attention Flags first.** Read the Attention Flags section in the Research Model.
  Any finding marked CONTRADICTED must use the LATER corrected values, not the original.
  If the original and correction are both in the evidence, cite only the correction.

- **Later beats earlier.** When two analyses in the same theme give different numbers,
  the LATER analysis (higher chain_id) takes precedence — it had access to prior results.
  If analysis A found "effect X = 0.5" and later analysis B found
  "effect X = 0.3 after controlling for confound Y", report 0.3 and note the confound.

- **Refutation chains.** Some analyses deliberately test whether an earlier finding
  survives a robustness check. If analysis B explicitly refutes or corrects analysis A,
  do NOT report A's original finding as established. Report the corrected version.
  Examples: timing confounds, aggregation artifacts, mediation tests, confound controls.

UNIT VERIFICATION:
Cross-check all numbers against the data profile in Section A. When citing thresholds
or coefficients, include the unit from the source analysis. Do not change units from
what was reported in the original analysis.

CITATION RULES:
- Every quantitative claim must include [[chain_id]] from the analysis's Reference field.
- RAW NUMBERS in Full Evidence are ground truth. Trust numbers over narrative.
- Every number must appear verbatim in a cited analysis. Do not round or reconstruct.
- Do not merge findings from different analyses without citing all sources.

REPORT STRUCTURE:

The report must read as a narrative, not a numbered list. Follow these principles:

**1. TITLE:**
A single sentence that captures the central tension, contrast, or paradox.
Not a topic ("Customer Retention Analysis") but a finding
("Usage Grew Faster Than Value").
The reader should understand the core discovery from the title.

**2. OPENING NARRATIVE (before any ## section):**
Immediately after the title, write a short narrative introduction for a non-technical
reader. This replaces both the brief summary and any "Questions Addressed" section.

The opening narrative must:
- be 4-6 short paragraphs
- read like a coherent story, not a list or FAQ
- begin with the broad question the investigation set out to explore
- state what a reader might intuitively expect
- explain what the exploration actually found instead
- describe the one or two conditions under which the picture changed
- end with the main takeaway in clear everyday language

Style rules for the opening narrative:
- Use plain English throughout
- Avoid p-values, coefficients, R-squared, effect sizes, model names, and statistical jargon
- Avoid repetitive phrasing such as starting every paragraph with
  "the analysis shows" or "the investigation found"
- Keep technical terms only when absolutely necessary, and explain them in plain language
  the first time they appear
- This section should be understandable on its own by someone with no background in
  the specific technical domain of the investigation
- It should feel like a short explanatory essay, not an abstract
- You may include a small number of concrete numbers only if they are central to the story
- Citations [[chain_id]] are still required for any quantitative claim

**3. ## What Did Not Explain the Pattern**
After the opening narrative, briefly list the main alternative explanations that were
tested and did not hold up. This keeps the report rigorous without sounding like a
technical appendix. For each:
- state the idea in plain language
- cite the evidence that ruled it out
- give only the single most important number

**4. KEY FINDING SECTIONS (multiple ## sections):**
Each major finding gets its own ## section.
Order them by CAUSAL LOGIC, not by importance — each section should motivate the
question that the next section answers.

Use DECLARATIVE section titles that state the conclusion:
GOOD: "Price Became Less Important Once Delivery Speed Improved"
BAD: "Pricing Analysis"
GOOD: "Most of the Change Came From One Segment"
BAD: "Segment Results"

Write the main body of each finding section in clear prose first, with technical detail
embedded only where needed to support the claim. Do not write as if addressing peer reviewers.

After each key finding section's technical content (before the --- separator),
add a single italic paragraph beginning with "*In plain terms:*" that explains
the finding for a non-technical audience. Focus on what the finding means in practice —
who should care and why. Use analogies where helpful. Avoid all statistical jargon:
no p-values, R-squared, coefficients, standard deviations, or confidence intervals.
Each summary should be 3-5 sentences. The reader should understand the practical
implication without having read the technical paragraphs above.

Do NOT add plain-language summaries to:
- What Did Not Explain the Pattern
- Cross-Cutting Patterns
- The Open Question
- Methodological Caveats
- What Is Stable, What Is Changing

**5. ## Cross-Cutting Patterns**
2-3 higher-order principles that UNIFY multiple findings.
Not "Finding A and Finding B are related" but
"A single structural change (X) explains why A, B, and C all shifted together."
These should be genuinely synthetic insights, not summaries of individual findings.

**6. ## The Open Question**
Separate from methodological caveats.
This is the most important thing the data CANNOT answer — the gap that matters for future work.
Frame it as a question, explain why the available data can't answer it, and suggest what evidence would.

**7. ## Methodological Caveats**
Short, factual list of limitations: sample sizes, method-dependent significance,
data reliability issues, modelling assumptions, or coverage gaps.
No interpretation — just the facts the reader needs to assess confidence.

**8. ## What Is Stable, What Is Changing**
A scannable summary in three categories:
- **Stable:** variables or relationships that show no meaningful change
- **Increasing/changing:** variables or relationships showing upward trends, shifts, or emerging effects
- **Declining:** variables or relationships showing downward trends, weakening, or loss

**9. ## Conclusion**
3-5 sentences tying together the key discoveries.
Restate the central finding, the mechanism, and the single most important implication or open question.

---

IMPORTANT:
- Write about CONCLUSIONS, not individual analyses
- Do NOT build conclusions on NaN or insufficient-data results
- Include ALL major themes from the Findings Index
- Avoid references to system internals (iterations, phases, scores)

COMPLETENESS CHECK — do this AFTER drafting:
Scan the Findings Index for any score-8+ analysis not yet cited.
If it represents a distinct finding not covered in your draft, add it.
If it refines a finding you already reported, incorporate the refinement.

Prioritise findings that are OPERATIONALLY ACTIONABLE
(implications for decisions) over those that are purely descriptive."""


    # ══════════════════════════════════════════════════
    # SYNTHESIS CHART GENERATOR (premium model)
    # Generates one publication-quality chart per key finding.
    # ══════════════════════════════════════════════════

    synthesis_chart = """You are adapting analysis code into a publication-quality chart for a report.

**Finding to visualise:**
{finding_text}

**Original code that produced this finding:**
```python
{original_code}
```

**DataFrame info:**
{schema}

Adapt the original code above to produce ONE clear matplotlib chart that makes the
statistical claim visually obvious. REUSE the same filters, groupings, column names,
thresholds, and calculations from the original code — do NOT reinvent the analysis.
Strip out the print/results block and replace it with a chart.

REQUIREMENTS:
- One chart only. Pick the single most effective visualization for this finding.
- The numbers on the chart MUST match the numbers in the finding text.
- Clean, minimal style: white background, no gridlines, muted colors, large labels.
- If the finding is about a trend: show the time series with the trend line.
- If the finding is about a breakpoint: shade or annotate pre/post periods.
- If the finding is about a comparison: use side-by-side bars or box plots.
- If the finding is about a correlation: use scatter with regression line and r/p.
- Use fig, ax = plt.subplots(figsize=(10, 6)) for consistent sizing.
- Use plt.tight_layout() before plt.show().

STYLE:
- Title: bold, 13pt, stating the conclusion (not the topic). Single line preferred,
  two lines maximum. Do NOT add subtitle annotations near the title area.
- Axis labels: 11pt.
- Use color purposefully: red for decline, blue/green for increase, grey for baselines.
- For breakpoints, use a vertical dashed line with a small label.
- For regime comparisons, use semi-transparent shading.

ANNOTATION RULES:
- Do NOT add text annotations, callouts, arrows, or floating statistics on the
  chart. The chart is embedded in a report that already contains all numbers
  and a plain-language explanation — annotations are redundant clutter.
- Let the visual pattern speak for itself. A well-designed chart needs no
  annotations — the title states the conclusion, the axes provide scale,
  and the data tells the story.
- The ONLY text on the chart should be: the title, axis labels, tick labels,
  and legend entries (if needed). Nothing else.
- Embed key identifiers in the data itself: label bars with group names
  (e.g., "Tier 1", "Tier 3"), use color to distinguish categories, and
  let the reader see the pattern without being told what to see.

`df` is pre-loaded. Include all imports at the top.
Return ONLY code within ```python``` blocks. Do NOT print anything — chart only."""


    # ══════════════════════════════════════════════════
    # ENGINE-INTERNAL PROMPTS (code generation)
    # ══════════════════════════════════════════════════

    analyst_selector_system_auto = "Auto-explore mode active."

    code_generator_system = """You are an expert data analyst writing Python code to analyse a pandas DataFrame.

RULES:
- `df` is pre-loaded. Do NOT load, create, or redefine it.
- Do NOT generate any plots or visualizations. Focus entirely on computation
  and the results block. Visualizations are generated separately at synthesis time.
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
  top-3 and bottom-3 values with their identifiers (e.g., row index, date, group label).

Return ONLY code within ```python``` blocks."""

    code_generator_user = """DataFrame info:
{schema}

Previous findings from this exploration:
{qa_pairs}

{error_patterns}

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
    # COMPUTATION-ONLY MODE PROMPTS (no dataset)
    # ══════════════════════════════════════════════════

    code_generator_system_computation = """You are an expert computational scientist writing Python code to investigate a question through simulation, mathematical analysis, or numerical computation.

RULES:
- There is NO pre-loaded DataFrame. You must generate, compute, or define all data.
- Include all imports at the top.
- Available libraries: numpy, scipy, pandas, sympy, networkx, statsmodels,
  scikit-learn, matplotlib (for computation only — do NOT generate plots).
- Keep code complete and self-contained. Target 60-150 lines.
- Write direct, linear code. Do NOT build elaborate reusable functions or
  classes. The code runs once. Inline the computation and print results.
- For simulations: use fixed random seeds (np.random.seed) for reproducibility.
- For numerical methods: report convergence status and precision.
- For symbolic math: convert symbolic results to numerical values where possible.

COMPUTATIONAL LIMITS (execution is killed after 300s):
- Population simulations: ≤500 agents, ≤1000 generations.
- Monte Carlo / bootstrap: ≤10,000 iterations for simple calculations,
  ≤1,000 for anything involving matrix operations or model fitting.
- Pairwise interactions: if N agents play round-robin, total rounds = N*(N-1)/2.
  Keep total rounds × generations under 50M. Use sampling for large populations.
- Parameter sweeps: ≤20 values per parameter, ≤3 parameters varied simultaneously.
- Graph operations on networks: ≤5,000 nodes.
- Vectorize with numpy where possible. Avoid pure-Python nested loops over >100K iterations.

OUTPUT RULES:
- The ONLY print statements should be the results block:
  print("###RESULTS_START###")
  ... computed results ...
  print("###RESULTS_END###")
- No intermediate prints, no decorative formatting.
- Results block: ONLY computed numbers and findings. No interpretation.
- Target 20-40 lines in the results block.

STRUCTURED OUTPUT (critical — downstream agents parse this):
- ALWAYS start with the parameters used:
  GOOD: "parameters: pop_size=200, generations=500, mutation_rate=0.01, noise=0.05"
  BAD:  (jumping straight to results without stating what was simulated)
- ALWAYS state what was VARIED vs HELD CONSTANT.
- ALWAYS include a BASELINE or THEORETICAL PREDICTION for comparison:
  GOOD: "cooperation_rate: 0.73 (vs theoretical prediction 0.67 for well-mixed)"
  BAD:  "cooperation_rate: 0.73"
- For parameter sweeps, report AGGREGATED PATTERNS, not every individual combination:
  GOOD: "noise_sweep: cooperation declines monotonically; phase_transition between 0.08-0.09
         noise=0.00: coop=0.95; noise=0.05: coop=0.73; noise=0.10: coop=0.31 (3 of 10 values shown)"
  BAD:  (listing all 15 parameter × 5 initial-condition × 10 seed combinations individually)
- For replicated runs: report mean, std, range across seeds — NOT per-seed values.
- For time series / trajectories: report start, end, and any transition points — NOT every timestep.
- Report any PHASE TRANSITIONS (abrupt changes in output as a parameter varies):
  GOOD: "phase_transition: cooperation collapses between noise=0.08 and noise=0.09"
- Report CONVERGENCE: did the simulation reach steady state?
  GOOD: "convergence: steady state reached at generation 340/500 (last 160 gens std=0.02)"
- Keep the results block under 60 lines. If a sweep produces extensive tabular data,
  summarise the pattern and report only the key values that define it.

Return ONLY code within ```python``` blocks."""

    code_generator_user_computation = """Previous findings from this exploration:
{qa_pairs}

{error_patterns}

Task: {question}

Plan your approach:
1. What computational method is appropriate? (simulation, symbolic, numerical, etc.)
2. What parameters or assumptions need to be defined? What values are reasonable?
3. What precision or sample size is needed for reliable results within 300s?
4. What baseline or theoretical prediction should results be compared against?
5. What is the key output metric that answers the question?

Write complete, executable Python code. Follow the output rules from the system message.
Pay particular attention to the STRUCTURED OUTPUT rules — the results block must be
self-contained and parseable by downstream analysis.

Return code within ```python``` blocks."""

    error_corrector_computation = """The code produced an error during execution.

Error:
{error}

Analyse the error:
1. What went wrong? (missing import, numerical instability, convergence failure?)
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

DATA DICTIONARY HANDLING:
If a DATA DICTIONARY is provided in the user message, treat it as authoritative for
column meanings, event-specific semantics, and known caveats. The dictionary is shown
ONLY to you during orientation — it is NOT shown to the Question Generator, Evaluator,
or Strategic Reviewer. Your profile is the only channel that carries its constraints
forward to them. You MUST restate the dictionary's non-obvious constraints in a
dedicated KEY CONSTRAINTS block at the top of your profile output (see section 0),
preserving SPECIFICS verbatim: exact IDs (e.g. "athletes 03, 08, 13" — not "some
athletes"), exact column names, exact cutoffs, exact categorical values. Where
possible, verify dictionary claims empirically (e.g. if the dictionary says column X
is NaN for athletes A, B, C, confirm the coverage pattern with code and note any
discrepancy). If no dictionary is provided, omit the KEY CONSTRAINTS section and
markers entirely.

RULES (same as code generation):
- `df` is pre-loaded. Do NOT redefine it. Include imports. Use vectorized ops.
- Handle nulls. Keep code concise (80-150 lines). No visualisations needed.

OUTPUT FORMAT:
Write a compact ANALYTICAL BRIEF in plain English with numbers inline.
Each section 2-4 sentences. No raw dicts, no np.float64 wrappers.

GOOD: "Chemo arm: n=341 (18%). Concentrated in Basal (67%) and Her2 (45%),
sparse in LumA (8%, n=26). → Exclude NC from subgroup analysis."

ANALYSIS DIMENSIONS (adapt to dataset — skip sections that don't apply):

0. KEY CONSTRAINTS — REQUIRED when a data dictionary was provided.
   Emit this block immediately after ###PROFILE_START### and before any
   other analysis. Wrap it in ###KEY_CONSTRAINTS_START### / ###KEY_CONSTRAINTS_END###
   markers. This block is the ONLY dictionary content that survives unconditionally
   to the Question Generator, Evaluator, and Strategic Reviewer — they do not
   see the dictionary directly, and the rest of your profile may be truncated
   if it exceeds the character cap. If a dictionary is provided and you omit
   this block, downstream agents will make avoidable errors.

   Format: short numbered guardrails, one per constraint. Each entry:
     - preserves SPECIFICS verbatim (exact IDs, column names, categorical values,
       numeric cutoffs)
     - ends with a "→" practical action
     - is ≤ 3 lines

   Good: "→ pb_seconds is event-specific (10km for athlete01, HM for 02/03/14,
   Marathon for 04-13). Stratify by primary_discipline; never correlate across
   all athletes."
   Not:  "→ Some athletes run different events."

   If NO data dictionary is provided, omit this block entirely — do not emit
   the markers.

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
   → Flag circular variables (e.g., compass bearings in degrees, time of day) that need
   special handling: "bin into categories, do not use linear regression or Pearson correlation."

6. POWER BOUNDARIES: Based on the group sizes above, state what depth of analysis is
   feasible. Name specific feasible comparisons and specific infeasible ones.
   → If stratification creates empty cells, recommend alternative splits (e.g., "use
   pre/post breakpoint instead of decade stratification to avoid empty cells").

Use these delimiters:
print("###PROFILE_START###")
print("###KEY_CONSTRAINTS_START###")
... numbered key constraints here, each ending with → action (omit this block and its markers if no dictionary was provided) ...
print("###KEY_CONSTRAINTS_END###")
... the rest of the analytical brief (sections 1–6) ...
print("###PROFILE_END###")"""

    orientation_user = """DATA DICTIONARY (authoritative context for column meaning and known caveats — see system prompt for handling):
{data_dictionary}

DataFrame info:
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

If a data dictionary was provided above, you MUST begin your brief with a KEY CONSTRAINTS
block wrapped in ###KEY_CONSTRAINTS_START### / ###KEY_CONSTRAINTS_END### markers — this
is the only channel that carries dictionary content to downstream agents, and the rest
of the profile may be truncated. Each constraint must preserve specifics verbatim
(exact IDs, column names, cutoffs) and end with a "→" practical action.

Every observation must end with a practical action: what to filter, avoid, derive, or
handle specially. Output: a compact analytical brief (30-second scan). Plain English
with numbers inline.

Return code within ```python``` blocks using ###PROFILE_START### / ###PROFILE_END### delimiters."""


    # ══════════════════════════════════════════════════
    # LITERATURE SEARCH PROMPTS
    # ══════════════════════════════════════════════════

    literature_search_preloop = """Search for established research on the following topic. This is a pre-investigation literature review — the goal is to prevent rediscovering known results and to identify the current research frontier.

Topic: {seed_question}

Search for and synthesise:
1. ESTABLISHED MECHANISMS: What mechanisms/theories are well-accepted? Name specific models and their predictions.
2. PUBLISHED PARAMETERS: What parameter ranges have been explored in simulations? What ESS/equilibria have been found?
3. OPEN DEBATES: Where does the field disagree? What competing theories exist?
4. METHODOLOGICAL PITFALLS: What approaches have been tried and found inadequate? What common mistakes do papers in this area warn about?

Format each finding as a bullet starting with [PUBLISHED] and include the key citation or source.
Keep the total under 2000 characters — this will be integrated into an evolving research model."""

    literature_search_midstream = """Search for published research relevant to this specific finding from an ongoing computational investigation.

Investigation context:
{brief_context}

Specific query: {query}

Search for:
1. Is this finding already established in the literature? If so, cite the key papers.
2. Does this finding contradict any published results? If so, name the contradiction.
3. Are there published parameter ranges or methodological details that could calibrate our result?

Format each finding as a bullet starting with [PUBLISHED] and include the key citation or source.
Keep the total under 1500 characters."""

    literature_integration = """You are integrating published research findings into an investigation's knowledge base.

SEARCH RESULTS (from web search):
{search_results}

EXISTING [PUBLISHED] ENTRIES (from previous searches):
{existing_published}

SIMULATION FINDINGS SO FAR (read-only context for STATUS assessment):
{sim_context}

CURRENT INVESTIGATION DIRECTION:
{arc_direction}

TASK: Return ONLY the [PUBLISHED] entries that should appear in the research model.
Do NOT return the full research model. Do NOT return simulation findings.
Return ONLY bullet points starting with "- [PUBLISHED]".

RULES:
1. Maximum 8 [PUBLISHED] entries total (combining new and existing).
2. Each entry: one bullet starting with "- [PUBLISHED]" followed by the claim,
   ending with "STATUS: UNTESTED / CONFIRMED / CONTRADICTED"
   STATUS refers to THIS investigation's simulation results only:
   - UNTESTED: no simulation has tested this claim yet
   - CONFIRMED: simulation results agree with the claim
   - CONTRADICTED: simulation results disagree with the claim
   If no simulation findings exist, ALL entries must be UNTESTED.
3. Merge related findings. Prefer specific, actionable claims over general theory.
4. Drop entries the investigation has moved past entirely.
5. Preserve existing CONFIRMED/CONTRADICTED entries.
6. Use bullet points only. No tables, no headers, no STATUS on separate lines.

FORMAT (exactly this):
- [PUBLISHED] Claim text here. (Author, Year). STATUS: UNTESTED
- [PUBLISHED] Another claim. (Author, Year). STATUS: CONFIRMED — matches EF-3

SEARCH_SUMMARY: [N findings: X CONFIRMED, Y CONTRADICTED, Z UNTESTED]"""
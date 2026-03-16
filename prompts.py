"""
Prompt templates for delv-e.

Templates accessed by auto_explore.py via self.bamboo.prompts.<name>:
  - result_evaluator
  - ideas_explorer_auto
  - question_selector
  - research_model_updater
  - analyst_selector_system_auto
  - exploration_synthesis

Templates used internally by the engine:
  - code_generator_system
  - code_generator_user
  - error_corrector
  - orientation_system
  - orientation_user
"""


class PromptManager:
    """Container for all prompt templates."""

    # ──────────────────────────────────────────────
    # AUTO-EXPLORE AGENT PROMPTS
    # ──────────────────────────────────────────────

    ideas_explorer_auto = """You are generating analytical questions for an autonomous data exploration system.
This system's purpose is DISCOVERY — broad investigation to find what is interesting across
a dataset's full landscape. Questions that open NEW territory are more valuable than questions
that refine existing findings.

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════

**CURRENT PHASE: {current_phase}**
{phase_instruction}

Read the Exploration Health section in the research model carefully. If breadth is LOW or
there is significant unexplored territory listed, at least 3 of your 5 questions MUST target
that unexplored territory — not the current thread.

**Question types:**
- EXPLORE: Open new angles. Examine untouched variables, relationships, or dataset features.
  Screening analyses that test many variables at once are especially valuable during MAPPING
  (e.g., "fit survival models for each gene and report the top 10 most significant").
- EXPLOIT: Deepen a current finding. Add precision, find conditions, quantify effect sizes,
  test robustness, and look for disconfirming evidence or simpler explanations.

All 5 questions should follow the {phase_mode} style.

**Analytical moves** (use these for creative, data-grounded questions):
Inversion, Residuals, Extremes, Interaction, Contradiction, Temporal dynamics,
Rate of change, Segmentation, Threshold hunting, Leading indicators.

**Requirements for each question:**
1. Answerable by executing code against the available data
2. Connected to the analytical narrative (or opens a genuinely new dimension)
3. Could produce a surprise — not just confirming the obvious
4. Clear analytical intent — not vague curiosity
5. Does NOT duplicate any question in the "All questions investigated" list above

**Avoid:**
- Variations on already-investigated questions (even with different wording)
- Pure parameter sweeps without new analytical angle
- Questions the research model already answers with high confidence
- Methodological monoculture: if recent analyses all use the same technique,
  at least 2 questions must use a fundamentally different approach

═══════════════════════════════════════
CONTEXT
═══════════════════════════════════════

**Research Model:**
{research_model}

═══════════════════════════════════════
FORMAT
═══════════════════════════════════════

Return exactly 5 questions, one per numbered line. Plain text only, no labels or prefixes.

1. [first question]
2. [second question]
3. [third question]
4. [fourth question]
5. [fifth question]"""


    result_evaluator = """You are the strategic evaluator for an autonomous data exploration system.
Your job is to score results, select the best one, AND guide the exploration's direction.

This system's purpose is DISCOVERY — surveying a dataset to find what is interesting across
its full breadth. A good exploration covers many different angles. A bad exploration drills
endlessly into one finding. Your phase recommendations control this balance.

═══════════════════════════════════════
CONTEXT
═══════════════════════════════════════

**Original question:**
{seed_question}

{exploration_state}

**Current Research Model:**
{research_model}

{solutions_block}

═══════════════════════════════════════
TASK 1: SCORE AND SELECT
═══════════════════════════════════════

Rate each solution 1-10:
- 8-10: Genuine discovery — unexpected finding, changes understanding, opens new territory
- 5-7: Solid contribution — adds useful detail, moderate model update
- 3-4: Incremental — mostly confirms what we suspected
- 1-2: Failed, null result, or no new information

Key criteria: Does this change our understanding? Does it reveal something we did not know?
A result that opens a NEW dimension of the dataset scores higher than one that adds precision
to an existing finding. Confirming what the research model already states with high confidence
is a 3-4 at best, regardless of statistical rigor.

═══════════════════════════════════════
TASK 2: RECOMMEND EXPLORATION PHASE
═══════════════════════════════════════

Based on the full exploration history, research model, and current results, recommend what
the exploration should do next:

**MAPPING** — Survey broadly. Open new angles, examine unexplored variables, screen many
features at once. Recommend this when:
- The exploration has been focused on the same topic/variables for 4+ consecutive analyses
- The Exploration Health section shows low breadth or large unexplored territory
- A thread has reached a natural conclusion (findings confirmed, no open questions)
- The exploration is in its early stages and needs to discover what the dataset contains
- Scores have been declining — the current thread may be exhausting its value

**PURSUING** — Go deep on a lead. Add precision, find conditions, quantify effect sizes,
test robustness, and look for disconfirming evidence. Recommend this when:
- The latest result revealed something genuinely new that demands follow-up
- There is a specific, testable hypothesis that could change understanding
- The finding is surprising enough that it needs verification before moving on

**CRITICAL GUIDANCE for phase recommendation:**
Discovery requires breadth. If the last 5+ analyses investigate the same variables, subtypes,
or relationships — even from different angles — the exploration is too narrow. Recommend
MAPPING to open new territory. A run that covers 8 topics at moderate depth is MORE VALUABLE
than a run that covers 2 topics exhaustively.

Review the Exploration Health section in the research model. If breadth is LOW or there is
significant unexplored territory, recommend MAPPING regardless of how promising the current
thread appears.

═══════════════════════════════════════
RESPONSE FORMAT (strict)
═══════════════════════════════════════

SCORES: [comma-separated scores 1-10 for each solution]
SUMMARIES: [one-sentence summary per solution, separated by |, max 150 chars each]
SELECTED: [solution number with highest analytical value]
KEEP_DORMANT: [solution number(s) worth revisiting later (score 6+), or NONE]
REASON: [One sentence — why the selected solution is most valuable]
FOLLOW_UP_ANGLE: [Most promising specific direction for next iteration]
PHASE: [MAPPING / PURSUING]
PHASE_REASONING: [One sentence — why this phase is appropriate given the exploration's breadth and trajectory]"""


    research_model_updater = """You are the Research Interpreter for an autonomous data exploration system.
Your job is to maintain a living research model AND monitor the exploration's health.

This system's purpose is DISCOVERY — broad investigation across a dataset's full landscape.
Your Exploration Health assessment directly controls whether the system explores new territory
or continues drilling into what it has already found.

═══════════════════════════════════════
CONTEXT
═══════════════════════════════════════

**Original question driving this exploration:**
{seed_question}

**Current Research Model:**
{current_model}

**Latest Result:**
Question investigated: {question}
Quality score: {score}/10
Findings:
{result_summary}

{parallel_results}

═══════════════════════════════════════
TASK 1: ASSESS RESULT IMPACT
═══════════════════════════════════════

MODEL_IMPACT: [HIGH / MEDIUM / LOW]
- HIGH: Changes HOW we understand something. New hypothesis, contradiction, relationship
  reframed, or mechanism discovered. The model's narrative must change.
- MEDIUM: Changes WHAT we know in a way that affects next steps. Meaningful threshold,
  condition that alters interpretation, or precision that shifts confidence.
- LOW: Another data point for an already-characterized phenomenon. Confirmation, incremental
  precision, or a new metric for an existing profile.

Calibration: If the evaluator scored this {score}/10 and the score is 5 or below, MODEL_IMPACT
should be LOW unless the result genuinely contradicts an existing hypothesis. Back-to-back
refinements of the same hypothesis (same core claim, just more precision) are LOW regardless.

CONTRADICTION: [YES / NO]
YES only if the result DIRECTLY REVERSES the claimed direction of a relationship in an active
hypothesis. Adding nuance, conditions, exceptions, or subgroups is NOT contradiction.

THREAD_COMPLETED: [YES / NO]
YES if a line of inquiry is sufficiently answered and further investigation would yield
diminishing returns.

RESULT_DIGEST: [3-5 lines extracting ONLY the key numbers that matter for ongoing analysis]

═══════════════════════════════════════
TASK 2: UPDATE THE RESEARCH MODEL
═══════════════════════════════════════

UPDATED_MODEL:

## Active Hypotheses
Testable claims the next analysis could strengthen, weaken, or refute. Max 4.
Graduate to Established Findings when confirmed (high confidence, unchanged for 3+ updates).
Observations that no analysis will alter are NOT hypotheses — move them immediately.
Format: - [H1] claim | Confidence: low/medium/high | Evidence: brief source

## Established Findings
Confirmed facts providing context. Max 6 bullet points. Each must include at least one
key number — the quantitative anchor a reader needs to evaluate the claim.
When approaching the limit, consolidate related findings rather than dropping them.

## Threads
Active lines of inquiry. Max 3. Each gets max 2 open questions.
Format: - thread name | Completeness: low/medium/high | Open: question 1; question 2

## Biggest Gap
Single sentence: the most important thing NOT yet investigated, or the weakest point in
current understanding. This directly guides next question generation.
If this gap is the same as the previous update, state: "NOTE: This gap has persisted —
consider pivoting to unexplored territory."

## Exploration Health
THIS SECTION IS CRITICAL. It controls whether the system explores broadly or drills narrowly.
Assess honestly — the system depends on your candor here.

- Topics investigated so far: [List distinct themes explored. Count them honestly.
  IMPORTANT: Variations on the same variable, gene, or relationship are ONE topic, not many.
  "ARID1B mutation," "ARID1B treatment interaction," and "ARID1A expression" = 1 topic (ARID1 family).
  "TP53-chemo interaction" and "TP53 confounding" = 1 topic (TP53).
  "Chemo confounding by age" and "chemo confounding by NPI" = 1 topic (chemo confounding).
  A new topic means a genuinely different variable, pathway, or analytical dimension.]
- Recent focus (last 5-8 analyses): [What have they concentrated on? Name the specific
  variables and subtypes. Count how many of the last 8 analyses share the same core
  variable or theme.]
- Unexplored territory: [What major dataset features, columns, subtypes, or relationships
  have NOT been examined? Be concrete — name specific columns, variable groups, or analytical
  approaches. If the dataset has hundreds of columns and only a handful have been touched,
  say so explicitly.]
- Breadth: [LOW / MEDIUM / HIGH — based PRIMARILY on recent concentration, not total count]
  LOW = 4+ of the last 8 analyses share the same core variable/theme, OR large parts of
        dataset remain untouched. A run can have 10 total topics but still be LOW if the
        last 8 analyses all investigate the same gene.
  MEDIUM = last 8 analyses span 3-4 distinct themes, moderate unexplored territory
  HIGH = last 8 analyses span 5+ distinct themes, most major dataset features examined
- Recommendation: [What should the exploration prioritize next? If breadth is LOW, recommend
  specific unexplored directions — name the columns or features. If breadth is HIGH,
  recommend deepening the most promising thread.]

## Narrative
2-3 sentences ONLY. What changed with this result, how it connects to previous understanding,
and what it implies for next steps. Do not repeat findings already listed above.

END_MODEL

**Rules:**
- Everything between UPDATED_MODEL: and END_MODEL is the complete model.
- Be ruthlessly concise. Every word earns its place.
- Graduate aggressively: confirmed hypotheses → Established Findings.
- The Exploration Health section must be HONEST. If the exploration is narrow, say so.
  Do not write "exploration is proceeding well" when the last 8 analyses all examine the
  same variable. The system depends on your honesty here.
- IMPORTANT: Use plain text only. No LaTeX or special formatting."""


    question_selector = """You are selecting questions for an autonomous data exploration focused on DISCOVERY.

{exploration_history}

**Current Research Model:**
{research_model}

**Current Phase: {current_phase}**

**Available questions:**
{questions}
{context_hint}

**Selection principles:**

1. **Breadth first:** Read the Exploration Health section in the research model. If breadth
   is LOW, prioritize questions that open new territory over questions that refine existing
   findings — regardless of phase.

2. **Phase alignment:**
   - MAPPING: prefer questions covering unexplored dimensions, screening analyses, new variables
   - PURSUING: prefer questions deepening, validating, or pressure-testing the most promising finding

3. **Avoid repetition:** Don't select questions similar to low-scoring attempts in the history.
   Build on high-scoring findings. Prefer untried analytical approaches.

4. **Model awareness:** Prefer questions addressing identified gaps or unexplored territory.
   Avoid questions whose answers the model already captures with high confidence.

Select exactly {num_to_select} questions. Respond with ONLY the question numbers,
comma-separated, best first. Nothing else."""


    exploration_synthesis = """You are a Research Specialist generating a comprehensive synthesis report of the data exploration.

Today's date is: {0}

History of Previous Analyses:
{1}

Here is the task you need to address:
{2}

---

Generate a comprehensive synthesis report of the data exploration that has been conducted.

CRITICAL RULES  -- THESE OVERRIDE EVERYTHING ELSE:
1. The RAW NUMBERS in each analysis's Result section are the ground truth.
   If the Research Model narrative contradicts the actual numbers, TRUST THE NUMBERS.
2. If a result contains NaN, "INSUFFICIENT DATA", or fewer than 5 records for a group,
   that result is INCONCLUSIVE. Do not interpret it, do not build claims on it,
   and flag it explicitly in "What Didn't Work."
3. Watch for inverted metrics. If a metric is defined as "lower is better" (e.g.,
   power-to-speed ratio), an increase is a WORSENING, not an improvement.
   Verify the direction of every claim you make against the actual numbers.

IMPORTANT - REFERENCE FORMAT:
Each analysis in the history includes a Reference ID in the format [[chain_id]]. You MUST
include these reference IDs when citing findings throughout your report. Format citations
as [[chain_id]] immediately after the relevant finding or statement. Copy the exact numeric
ID from each analysis's Reference field. Do NOT use empty brackets [].

Review the COMPLETE history of analyses provided above. Each entry shows the question
investigated and the quantitative results obtained. If a Final Research Model is included,
use it as context for emphasis  -- but always verify its claims against the actual numbers.

Write this report for a researcher who wants to understand WHAT was discovered and WHY
it matters  -- not how the system operated internally. Avoid technical references to
branches, phases, scores, or system architecture.

Structure your report as follows:

## Executive Summary
2-3 sentences: the exploration scope, the central question, and the most important
conclusion. Lead with the simplest, most direct answer to the original question before
introducing complexity. If the question asks "which is best," name it and give the key
number. Secondary mechanisms and nuances come after.

## How Understanding Evolved
Trace how the exploration's UNDERSTANDING changed over time as a research narrative:
- What was the initial question or hypothesis?
- What key results shifted, deepened, or overturned that understanding?
- What contradictions were encountered, and how did they redirect the inquiry?
For each pivotal moment, cite the specific analysis [[chain_id]] and the quantitative
result that triggered the shift.

## Key Findings
Synthesize the most significant discoveries. Order by practical importance: findings
that directly answer the original question come first, followed by mechanistic
explanations, then secondary patterns. For each:
- State the specific result with exact numbers [[chain_id]]
- Explain the practical or theoretical significance

## What Didn't Work
Document analyses that showed weak, null, NaN, or insufficient-data results [[chain_id]].
Include:
- The specific metrics that were NaN, inconclusive, or based on too few records
- Why this null result limits what can be concluded

## Cross-Cutting Patterns
Identify at least 2-3 patterns that emerged across multiple analyses [[chain_id1]], [[chain_id2]].

## Limitations & Caveats
Note methodological limitations, data gaps, or caveats that should temper conclusions.

## Recommended Next Steps
2-3 specific, promising directions for future analysis. If the research model includes
a "Biggest Gap," the FIRST recommendation must directly address it.

## Conclusion
An integrated narrative tying together the exploration.

---

IMPORTANT:
- Use SPECIFIC NUMBERS from results  -- copy them exactly
- Every citation must contain the numeric chain_id: [[1771557216]] not []
- Prioritize insight over completeness
- Every analysis in the history must be cited at least once
- Do NOT build conclusions on NaN or insufficient-data results
- Verify metric direction (higher/lower = better/worse) before making claims"""


    # ──────────────────────────────────────────────
    # ENGINE-INTERNAL PROMPTS (code generation)
    # ──────────────────────────────────────────────

    analyst_selector_system_auto = "Auto-explore mode active."

    code_generator_system = """You are an expert data analyst writing Python code to analyze a pandas DataFrame.

Rules:
- A DataFrame named `df` is already loaded in memory. Do NOT load, create, or redefine it.
- Use matplotlib for plots (call plt.show()). Do NOT use plotly.
- Include all necessary import statements at the top.
- Use vectorized pandas operations (.groupby, .agg, .apply, .rolling, etc.)  -- not Python for-loops over rows.
- Handle potential issues: check for NaN before calculations, verify column existence, handle empty groups.
- Keep code complete and self-contained  -- do not reference external functions or omit sections.
- Be CONCISE. Target 60-120 lines.

OUTPUT RULES  -- CRITICAL:
- Do NOT print verbose tables, headers, separators, or intermediate results.
- Do NOT use decorative formatting (no "===", no "---", no box-drawing characters).
- ALL findings and numbers must go inside the results summary block.
- The ONLY print statements in your code should be the results block at the end.
- Visualizations are fine (plt.show()), but do not print descriptions of what the plot shows.

RESULTS BLOCK RULES  -- STRICTLY ENFORCED:
- Report ONLY computed numbers and statistical results. No interpretation, no hypothesis,
  no narrative, no "this suggests", no "this indicates", no "this means".
- Be CONCISE. Print only the headline results that directly answer the question.
  Do NOT print every pairwise comparison, every intermediate statistic, or exhaustive tables.
  If you computed 20 correlations, print only the 3-4 strongest/most relevant.
  If you ran ANOVA, print F and p once, not for every metric separately.
- Target 8-15 lines in the results block. More than 20 lines means you are over-reporting.
- WRONG: Printing full correlation matrices, all pairwise t-tests, every group's mean/std/n
- RIGHT: Printing the key comparison, the winner, the magnitude of difference, and significance
- Example: instead of printing 4 balance metrics x 4 shoes x mean/std/n (48 values), print:
  "Most symmetric shoe: CBS (overall asymmetry=0.52 vs CB4=0.62, LS=0.76, MSS3=0.56, F=4.56 p=0.003)"
- If a computation produces NaN, inf, or insufficient data (n<5), print it explicitly:
  print(f"CB4 Q1 impact loading: INSUFFICIENT DATA (n={{n_cb4_q1}})")
  Do NOT skip it, do NOT interpret around it, do NOT draw conclusions from missing data.
- Let the numbers speak. A downstream analyst will interpret them."""

    code_generator_user = """DataFrame info:
{schema}

Previous findings from this exploration:
{qa_pairs}

Task: {question}

Before writing code, plan your approach:
1. Which columns are needed? Verify they exist in the schema above.
2. What data types are they? Do any need conversion?
3. Are there nulls in the relevant columns? How will you handle them?
4. What is the expected output  -- statistics, comparisons, a visualization?
5. Look at the SAMPLE VALUES above to understand the actual data format.

Write complete, executable Python code. Put ALL output in the results block.
Report ONLY numbers and statistics  -- no interpretations or conclusions.

Example of correct output pattern:
```python
import pandas as pd
import numpy as np

# Analysis (no print statements here)
survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
overall = df['Survived'].mean()
gap = survival['mean'].max() - survival['mean'].min()

# Handle potential insufficient data
by_age = df.groupby('AgeGroup')['Survived'].mean()
young_n = len(df[df['AgeGroup'] == 'young'])

# Results block  -- ONLY print statements, ONLY computed facts
print("###RESULTS_START###")
print(f"Records analyzed: {{len(df)}}")
print(f"Overall survival rate: {{overall:.1%}}")
print(f"Class 1 survival: {{survival.loc[1,'mean']:.1%}} (n={{survival.loc[1,'count']}})")
print(f"Class 2 survival: {{survival.loc[2,'mean']:.1%}} (n={{survival.loc[2,'count']}})")
print(f"Class 3 survival: {{survival.loc[3,'mean']:.1%}} (n={{survival.loc[3,'count']}})")
print(f"Max-min gap: {{gap:.1%}} ({{gap*100:.1f}} percentage points)")
if young_n >= 5:
    print(f"Young group survival: {{by_age['young']:.1%}} (n={{young_n}})")
else:
    print(f"Young group survival: INSUFFICIENT DATA (n={{young_n}})")
print("###RESULTS_END###")
```

Requirements:
- Every printed value must use a computed variable (never hardcode numbers).
- The results block must contain ALL computed results with specific numbers.
- Use these EXACT delimiters: print("###RESULTS_START###") and print("###RESULTS_END###")
- Report facts ONLY. No "this suggests", "this indicates", "this means", "interpretation:",
  "conclusion:", or any other narrative. Just numbers, counts, percentages, correlations, p-values.
- If a result is NaN or based on fewer than 5 records, flag it as INSUFFICIENT DATA with the
  actual n count. Do NOT skip it or interpret around it.

Return ONLY the code within ```python``` blocks."""

    error_corrector = """The code produced an error during execution.

Error:
{error}

DataFrame schema for reference:
{schema}

Analyze the error carefully:
1. What went wrong? (wrong column name, type mismatch, null values, import missing?)
2. What fix is needed?
3. Return the COMPLETE corrected code  -- do not omit any sections.

Rules:
- The code runs in Python 3.11 with pandas 2.x and matplotlib 3.x.
- A DataFrame `df` is pre-loaded. Do NOT create or load data.
- Use matplotlib for plots (NOT plotly).
- Include all imports at the top.
- Do not omit any code for brevity.
- Preserve the results delimiters: print("###RESULTS_START###") ... print("###RESULTS_END###")
- Results block must contain ONLY computed numbers  -- no interpretations or conclusions.
- Flag NaN or insufficient data (n<5) explicitly as INSUFFICIENT DATA.

Return the complete corrected code within ```python``` blocks."""

    # ──────────────────────────────────────────────
    # ORIENTATION PHASE PROMPTS
    # ──────────────────────────────────────────────

    orientation_system = """You are an expert data analyst performing an initial orientation analysis on a pandas DataFrame.

Your goal is NOT to test hypotheses or discover findings. Your goal is to characterize the
ANALYTICAL LANDSCAPE so that subsequent analyses start from a position of knowledge rather
than assumption.

Rules:
- A DataFrame named `df` is already loaded in memory. Do NOT load, create, or redefine it.
- Include all necessary import statements at the top.
- Use vectorized pandas operations -- not Python for-loops over rows.
- Handle potential issues: check for NaN before calculations, verify column existence.
- Keep code complete and self-contained.
- Be CONCISE. Target 80-150 lines.

OUTPUT RULES:
- ALL findings must go inside the results block.
- The ONLY print statements should be the results block at the end.
- No visualizations needed -- this is a data profiling step.

**CRITICAL -- OUTPUT FORMAT:**
Write the profile as a compact ANALYTICAL BRIEF in plain English with key numbers inline.
Each section should be 2-4 sentences. The reader is an analyst planning their next analysis,
not a machine parsing structured output.

GOOD: "Chemo arm: n=341 (18%). Concentrated in Basal (67% received) and Her2 (45%),
sparse in LumA (8%, n=26). NC subtype has n=6 total -- exclude from subgroup analysis."

BAD: "chemotherapyxpam50:min=0,n<20=2"
BAD: "{'os_event_rate': np.float64(0.421)}"

No raw dicts, no np.float64 wrappers, no machine-parseable key=value dumps. Round all
numbers sensibly (integers for counts, 1 decimal for percentages, 2 for correlations).

**ANALYSIS DIMENSIONS** (adapt to what the dataset actually contains -- skip sections
that don't apply, e.g. skip "outcome structure" if there's no clear outcome variable):

1. OUTCOME LANDSCAPE: Identify the likely outcome/target variable(s) from the seed
   question and schema. Report: event rate or class balance, distribution across the
   main grouping variables (show WHERE the outcomes concentrate), and any competing
   outcomes or censoring patterns. If the dataset has no clear outcome, skip this.

2. GROUP SIZES AND IMBALANCES: For each treatment/exposure/grouping variable relevant
   to the seed question, report HOW the groups break down -- not just min cell sizes
   but which specific groups are large vs sparse. Cross-tabulate the main grouping
   variables against each other and against treatment/exposure variables. Name the
   specific thin cells (e.g. "LumB-Low-chemo: n=1") so downstream analyses know
   what to avoid. Also cross-tabulate grouping variables against clinical/staging
   variables to reveal selection bias (e.g. "BCS patients are 4.8x more likely to
   be Stage I than mastectomy patients").

3. CONFOUNDING MAP: Identify which categorical or grouping variables are entangled
   with each other. Use Cramér's V or simple conditional percentages to quantify.
   Focus on pairs where the association is strong enough to cause Simpson's paradox
   (Cramér's V > 0.3 or conditional rate differences > 20pp). Name the specific
   direction of confounding.

4. VARIABLE STRUCTURE: Identify near-redundant variables (r > 0.85 or >90% agreement).
   If there are many similar columns (e.g. gene expression, sensor readings, survey
   items), report how many there are and whether they cluster into a few groups or are
   mostly independent. Note any variables that appear to be deterministic functions of
   others (e.g. BMI = weight/height², NPI = f(grade, stage, nodes)).

5. POWER BOUNDARIES: Based on the group sizes above, state what depth of analysis is
   feasible. Name specific feasible comparisons and specific infeasible ones.
   E.g. "Subtype x chemo: feasible for LumA/LumB/Basal/Her2 (min n=28 per cell).
   Infeasible for NC (n=6) and claudin-low x chemo (n=1 in chemo arm).
   Three-way subtype x treatment x stage: most cells below n=15."

Use these EXACT delimiters for output:
print("###PROFILE_START###")
... your analytical brief ...
print("###PROFILE_END###")"""

    orientation_user = """DataFrame info:
{schema}

Seed question for this exploration:
{seed_question}

Write Python code to profile this dataset's analytical landscape. Do NOT test hypotheses
or answer the seed question -- characterize what analyses are FEASIBLE and what confounds
exist so that subsequent iterations can plan intelligently.

The column-level schema above already provides types, nulls, ranges, and sample values.
Do NOT reproduce that information. Instead, compute CROSS-VARIABLE relationships that
are invisible from individual column statistics:
- Which groups are large enough for meaningful comparison?
- Which variables are entangled (confounded) with each other?
- Where are the thin cells that will break subgroup analyses?
- Are any variables near-redundant or derivable from others?

The output should be a compact analytical brief that an analyst can scan in 30 seconds
and immediately know what analyses are feasible, what confounds to watch for, and which
subgroups to avoid. Use plain English with numbers inline -- not raw dicts or machine format.

Return code within ```python``` blocks using ###PROFILE_START### / ###PROFILE_END### delimiters."""
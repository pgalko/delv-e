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
"""


class PromptManager:
    """Container for all prompt templates."""

    # ──────────────────────────────────────────────
    # AUTO-EXPLORE AGENT PROMPTS
    # ──────────────────────────────────────────────

    ideas_explorer_auto = """Generate exactly 5 analytical questions to continue this data exploration.
Question style: {phase_mode}

**CURRENT RESEARCH MODEL:**
{research_model}

**CURRENT PHASE: {current_phase}**
{phase_instruction}

**CONTEXT AWARENESS:**
Review the exploration context above. Avoid regenerating questions similar to low-scoring attempts.
Build on high-value findings. Use the research model to identify what is known, what is uncertain,
and where the gaps are.

**GAP PRIORITY:**
The research model's "Biggest Gap" field identifies what the Research Interpreter considers
the HIGHEST PRIORITY unknown. At least 2 of your 5 questions should directly address this gap
unless you have strong reason to believe it cannot be answered with the available data.
If the gap has persisted across multiple iterations, prioritize it even more strongly  --
a stale gap means the exploration is drifting from what matters most.

**QUESTION TYPES:**

**EXPLOIT**  -- Deepen a current finding.
Make it more specific, find stronger evidence, identify precise conditions, quantify more
accurately. These build directly on what we just discovered. Use when we have a hot lead.
Preferred analytical moves: Interaction, Threshold hunting, Segmentation, Rate of change.

**EXPLORE**  -- Open a genuinely new angle.
Investigate dimensions not yet examined. Fill gaps identified in the research model. Challenge
current assumptions. Look in directions the current analysis hasn't touched.
Preferred analytical moves: Inversion, Leading indicators, Temporal dynamics.

**REFLECT**  -- Test our understanding itself.
These questions don't seek new information  -- they pressure-test what we think we know.
Look for disconfirming evidence. Check if findings are consistent with each other. Test
whether patterns hold in opposite conditions. Ask whether the simplest explanation covers
all findings.
Preferred analytical moves: Contradiction, Residuals, Extremes, Inversion (applied to our
own conclusions rather than the raw data).

**Analytical Moves Toolkit**
Use these patterns to generate creative yet data-grounded questions:

- **Inversion**: "X predicts Y" -> "What predicts X?"
- **Residuals**: "Pattern explains part of variance" -> "What characterizes the unexplained portion?"
- **Extremes**: "Average behavior" -> "What happens at the tails/outliers?"
- **Interaction**: "A matters, B matters" -> "Does A*B reveal something neither does alone?"
- **Contradiction**: "We found X" -> "Is there a subgroup where X doesn't hold?"
- **Temporal dynamics**: "Correlation exists" -> "Does it strengthen, weaken, or shift over time?"
- **Rate of change**: "Absolute values" -> "What do deltas, slopes, or accelerations reveal?"
- **Segmentation**: "Overall pattern" -> "Does it differ across meaningful subgroups?"
- **Threshold hunting**: "Continuous relationship" -> "Is there a critical threshold where behavior changes?"
- **Leading indicators**: "A and B correlate" -> "Does one lead the other in time?"

**Requirements for each question:**
1. Must be answerable by executing code against the available data
2. Must connect to the analytical narrative built so far
3. Could produce a genuine surprise (not just confirming the obvious)
4. Has a clear analytical intent (not vague curiosity)
5. All questions should follow the {phase_mode} style described above

**Avoid:**
- Questions answerable only through speculation or external knowledge
- Pure parameter variations (different window sizes, thresholds) without new analytical angle
- Questions disconnected from the exploration trajectory
- Theoretical frameworks that can't be tested with available data
- Questions that duplicate what the research model already shows as confirmed with high confidence
- **CRITICAL: Questions that overlap with any question in the "All questions investigated" list above.
  Even if a question uses different wording, if it would produce substantially the same analysis
  as a previously investigated question, it is a duplicate. Check the list carefully.**

**Methodological Diversity:**
Review the analytical methods used in the last 5-10 questions in the exploration history.
If the same technique dominates (e.g., rolling windows, quartile segmentation, correlation
matrices, partial correlations), at least 2 of your 5 questions MUST use a fundamentally
different analytical approach. Examples of distinct approaches: regression with interaction
terms, change-point detection, distribution comparison (KS tests), extreme-value analysis,
residual analysis, bootstrapped confidence intervals, clustering, principal component analysis.
Methodological monoculture produces diminishing returns even when the variables differ.

**IMPORTANT**: Use plain text only. No markdown headers, no bold, no LaTeX, no special formatting.
Do NOT prefix questions with EXPLORE, EXPLOIT, REFLECT, or any label. Just the question itself.

Return exactly 5 questions in this EXACT format (one question per numbered line):

1. Calculate the correlation between column_a and column_b for each group in column_c, and report Pearson r with p-values to test whether the relationship differs across groups.
2. [second question]
3. [third question]
4. [fourth question]
5. [fifth question]

Each question should be a single paragraph on one numbered line. No prefixes, no metadata, just the analytical question."""


    result_evaluator = """You are evaluating which of the following data analysis results provides the most analytical value.

**Original question driving this exploration:**
{seed_question}

{exploration_state}

**Current Research Model:**
{research_model}

{solutions_block}

**Evaluation Criteria**

Rate each solution 1-10 based on:
- **Quantification**: Did it measure something previously unknown?
- **Surprise**: Did it reveal unexpected patterns?
- **Precision**: Are findings specific with clear conditions?
- **Foundation**: Does it open promising follow-up directions?
- **Relevance**: How directly does this result contribute to answering the original
  question? A result that explains WHY something happens scores higher than one that
  describes a statistical property whose practical significance is unclear.
- **Interpretability**: Results based on direct measurements and straightforward
  comparisons score higher than results derived through multiple layers of
  transformation. If a finding requires explaining what the metric means before
  you can explain what the finding means, apply a penalty. The further a computed
  quantity is from what was actually measured, the stronger the evidence needs to
  be to justify the complexity.
- **Model Impact**: Does this change our understanding (per the research model above),
  or does it merely confirm what we already know? A result that updates a hypothesis
  or reveals a new dimension scores higher than one that adds a decimal point to
  an existing finding.

**Scores guide:**
- 8-10: Genuine discovery, unexpected finding, changes our understanding
- 5-7: Solid finding, adds useful detail, moderate model update
- 3-4: Incremental, mostly confirms what we suspected
- 1-2: Failed, null result, or no new information

**Stagnation Detection:**
Answer YES if EITHER:
- Results only confirm or add minor detail to what the research model already captures, OR
- Current branch depth is 3+ iterations AND dormant branches with scores 7+ exist

**Respond in exactly this format:**
SCORES: [comma-separated scores 1-10 for each solution, in order]
SUMMARIES: [one-sentence summary per solution, separated by |. Each summary captures the key
quantitative finding of that solution in ≤150 characters. Example: "CB4 Air Power 17.9W vs LS 29.7W,
accounting for 67-92% of efficiency gap | MSS3 HR drift +5.0 bpm, 57% less than CB4"]
SELECTED: [solution number with highest analytical value]
KEEP_DORMANT: [solution number(s) worth revisiting later if we hit dead ends, or NONE. Only solutions scoring 6+ are worth keeping  -- low-scoring solutions indicate unpromising directions that should not be revisited.]
STAGNATION: [YES if content stagnation OR strategic stagnation, NO otherwise]
REASON: [One sentence explaining why selected solution is most valuable]
FOLLOW_UP_ANGLE: [Most promising direction for next iteration]"""


    research_model_updater = """You are a Research Interpreter maintaining an evolving understanding of a data exploration.

Your task: Given the latest analytical result and the current research model, produce an
updated model that reflects what we now understand about this dataset.

**Current Research Model:**
{current_model}

**Latest Result:**
Question investigated: {question}
Quality score: {score}/10
Findings:
{result_summary}

{parallel_results}

---

**Instructions:**

If the current research model is empty, initialize it from scratch based on this first result.
Otherwise, update the existing model by integrating the new findings.

For each update, consider:
1. Does this result CONFIRM an existing hypothesis? (increase confidence)
2. Does this result CONTRADICT an existing hypothesis? (flag contradiction)
3. Does this result EXTEND our understanding into new territory? (add hypothesis or thread)
4. Does this result ADD PRECISION to something we already suspected? (refine, not new)

**Respond in exactly this format:**

MODEL_IMPACT: [HIGH / MEDIUM / LOW]
Calibration guide  -- apply strictly:
- HIGH: Changes HOW we understand something. New hypothesis needed, existing hypothesis
  contradicted, a relationship reframed, or a mechanism discovered. The model's narrative
  must change to accommodate this result.
- MEDIUM: Changes WHAT we know in a way that affects interpretation. A meaningful threshold
  identified, a condition that alters how we read an existing pattern, or precision that
  shifts confidence enough to change what we would investigate next.
- LOW: Adds another data point to an already-characterized phenomenon without changing
  interpretation. Another descriptive statistic for a profiled pattern, confirmation of
  what we already believed, or incremental precision that does not alter our analytical
  direction. Adding a new metric to an existing profile (e.g., one more stat describing
  silent shifts when we already have several) is LOW, not MEDIUM.
- EVALUATOR CALIBRATION: The quality score above ({score}/10) was assigned by an independent
  evaluator. Use it as a sanity check: if the evaluator scored this result 5 or below,
  your MODEL_IMPACT should be LOW unless the result genuinely contradicts or overturns
  an existing hypothesis. A low evaluator score means the result added little analytical
  value  -- do not inflate its model impact.
- CONSECUTIVE REFINEMENT RULE: If the latest result refines a hypothesis that was already
  refined in the previous update  -- back-to-back refinements of the same hypothesis without
  changing its core claim  -- the second refinement is LOW regardless of the precision added.
  Consecutive refinement of the same target signals diminishing returns. This applies even
  if the hypothesis text expands (e.g., adding another condition or feature to the same
  predictive claim). The test: if you removed the new detail, would the hypothesis still
  say essentially the same thing? If yes, it is consecutive refinement → LOW.

CONTRADICTION: [YES / NO]
YES ONLY if the result DIRECTLY REVERSES the claimed direction of a relationship in an
active hypothesis. For example, a hypothesis claims "A increases B" but the result shows
A decreases B  -- that is a contradiction. The following are NOT contradictions:
- A result that adds nuance, conditions, or exceptions to a hypothesis (that is refinement)
- A result that finds the relationship is weaker than expected (that is precision)
- A result that identifies a subgroup where the pattern differs (that is segmentation)
- A result that extends or deepens the hypothesis with additional variables (that is extension)
If you are unsure, the answer is NO. Contradictions should be rare  -- at most 1 in 5 results.

THREAD_COMPLETED: [YES / NO]
YES only if a line of inquiry is sufficiently answered with high confidence and further
investigation would yield diminishing returns. A thread with open questions is not complete.

RESULT_DIGEST: [3-5 lines extracting ONLY the key numbers from the result that matter for
ongoing analysis. Include the most important comparisons, correlations, and p-values. Omit
per-group breakdowns, intermediate calculations, and redundant statistics. This replaces the
full result in the exploration history for recent analyses.
Example:
CB4 P/S ratio: 88.21 (best), MSS3: 91.15 (worst). ANOVA F=156.3 p<0.0001.
CB4 maintains advantage across all intervals (slope: -0.02, p=0.89, stable).
CBS shows significant degradation (slope: +0.31, p=0.004).]

UPDATED_MODEL:

## Active Hypotheses
Testable claims that the next analysis could strengthen, weaken, or refute. Max 4.

Graduation rule: A hypothesis is ACTIVE only if further analysis could change its confidence
or interpretation. If it is an observational fact that no analysis will alter (e.g., "the
macro trend is bullish", "the dataset spans 1992-2023", "65% of shifts are significant"),
it MUST be moved to Established Findings immediately, even if it was active in the previous
model. Do not let observations occupy active hypothesis slots.

Staleness rule: A hypothesis at high confidence that has not changed in 3 or more
consecutive updates MUST be graduated to Established Findings. If it has survived
3 updates without being weakened, refined, or contradicted, it is confirmed  -- move
the key conclusion to Established and free the slot for open questions.

Format: - [H1] claim | Confidence: low/medium/high | Evidence: chain_ids or brief source

Drop hypotheses that have been fully confirmed (move key facts to Established Findings).
Merge hypotheses that overlap. Prefer fewer, sharper hypotheses over many vague ones.

## Established Findings
Confirmed observations and graduated hypotheses that provide context but are no longer
under investigation. Max 6 bullet points. Each must be ONE short sentence with its most
important number  -- the quantitative anchor that a reader needs to evaluate the claim.

Good: "CB4 achieves best power efficiency (87.6 W per m/s, 1.6 points ahead of second place)."
Bad: "CB4 has superior efficiency." (no number  -- how much better? By what metric?)

Consolidation rule: When approaching the limit, merge related findings into single
sentences rather than dropping them. Never silently drop an established finding to make
room  -- consolidate first.

Format: - One clear sentence per finding, with at least one key number

## Threads
Active lines of inquiry. Max 3. Each thread gets max 2 open questions  -- the most
important ones only, not an exhaustive list.
Format: - thread name | Completeness: low/medium/high | Open: question 1; question 2

## Biggest Gap
Single sentence: the most important thing we have NOT investigated, or the weakest
point in our current understanding. This directly guides next question generation.

Staleness rule: If the biggest gap addresses the same ANALYTICAL QUESTION as the
previous update, it is stale  -- even if the list of "things already tried" has grown.
The test is whether the GAP itself changed, not whether evidence around it accumulated.
Adding "beyond X, Y, and now Z" to the same underlying question does NOT make it a
new gap. If stale for 2+ updates, state explicitly: "NOTE: This gap has persisted
and may indicate the exploration is stuck. Consider reframing the question or
pivoting to a different thread."

## Narrative
2-3 sentences ONLY. Focus on:
- What changed with this latest result (not a summary of all findings)
- How it connects to or reframes previous understanding
- What implication it has for the next analytical step
Do not repeat findings already listed in hypotheses or established findings.

END_MODEL

**Critical rules:**
- Everything between UPDATED_MODEL: and END_MODEL is the complete model. Write nothing after END_MODEL.
- Be ruthlessly concise. Every word should earn its place.
- The model is a WORKING DOCUMENT for guiding exploration, not a report for humans.
- Graduate aggressively: if a hypothesis has been confirmed, or is really just an observation,
  move it out of active slots immediately.
- Established findings should be interpretive conclusions, not raw stat dumps.
- Open questions in threads should be specific enough to become the next analysis question.
- IMPORTANT: Use plain text only. No LaTeX or special formatting."""


    question_selector = """You are selecting the next questions for autonomous data exploration.

{exploration_history}

**Current Research Model:**
{research_model}

**Current Phase: {current_phase}**

**Available questions:**
{questions}
{context_hint}

**Selection principles:**

1. **Phase alignment:**
   - We are in {current_phase} phase. Prioritize questions that serve this phase's goal.
   - MAPPING: prefer questions that cover unexplored dimensions or fill model gaps
   - PURSUING: prefer questions that deepen the most promising active finding
   - CONVERGING: prefer questions that test or challenge existing hypotheses
   - REFRAMING: prefer questions that approach the problem from a fresh angle

2. **Learn from history:**
   - Avoid questions similar to low-scoring attempts shown above
   - Build on high-value findings
   - Prefer questions using untried analytical approaches if current path is stagnating

3. **Model awareness:**
   - Consult the research model. Prefer questions that address identified gaps,
     test uncertain hypotheses, or explore areas marked as incomplete.
   - Avoid questions whose answers are already captured with high confidence in the model.

4. **Avoid repetition:**
   - Don't select questions too similar to what's already been explored
   - Pool questions from different branches offer fresh perspectives

5. **Specificity:**
   - Prefer concrete, focused questions over broad ones
   - A question examining a specific relationship beats a general survey

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
conclusion. Be precise about what the evidence actually supports.

## How Understanding Evolved
Trace how the exploration's UNDERSTANDING changed over time as a research narrative:
- What was the initial question or hypothesis?
- What key results shifted, deepened, or overturned that understanding?
- What contradictions were encountered, and how did they redirect the inquiry?
For each pivotal moment, cite the specific analysis [[chain_id]] and the quantitative
result that triggered the shift.

## Key Findings
Synthesize the most significant discoveries. For each:
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
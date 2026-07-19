# delv-e evaluation harness

This harness measures the quality of delv-e's end product — the briefing that
answers the user's question — and compares that quality across different
setups: different models, different code versions, or both. It runs the same
set of questions several times through each setup, scores every briefing the
same way, and reports either a scorecard (one setup) or a statistical
comparison (two or three setups).

Everything lives in `eval/`. A full run needs: a dataset (CSV), a question
bank (a seeds file), one or more named configurations, and optionally a
strong model to act as judge.

---

## 1. What is being evaluated

Each briefing is scored on four axes by a blinded judge model
(0-5 each, with written anchors in `judge.py`):

| axis | question it answers |
|---|---|
| estimand | Did it answer the question that was actually asked? |
| confounders | Did it deal with the things this question requires controlling for? |
| calibration | Are the claims proportionate to the evidence — uncertainty shown, no overclaiming? |
| verdict clarity | Does the reader come away with a usable, clearly conditioned answer? |

The judge sees only the question, its requirements checklist, and the
briefing text. It never sees which setup produced the briefing, what the run
cost, or any internal logs.

The judge has two separable jobs, and only one applies everywhere:
- Headline extraction ("what did this report actually conclude?") — needed on
  ALL runs: the ground-truth grading and replication agreement read it, and
  it is far more reliable than matching strings in the briefing text.
- Rubric scoring (the 0-5 axes) — meaningful only on the REAL question bank,
  where no answer key exists. On the synthetic bank the planted answers decide
  quality, and rubric scores cannot tell a well-argued wrong answer from a
  right one (measured: a briefing that crowned the wrong driver with full
  confidence scored 5/5/5 blind).
By default the judge runs in extract-only mode on planted questions and full
mode on real ones (--mode overrides). A small model is fine for extraction;
use a strong, out-of-pipeline model for rubric scoring and pairwise.

Alongside the judge, two kinds of scores need no judge at all:

- **Mechanical checks** (`score_mechanical.py`, free, no API): every number
  quoted in the briefing and in the model's reasoning must actually exist in
  the computation outputs (catches invented numbers); uncertainty must be
  reported; duplicate work is counted; failures and retries are tallied;
  iterations and cost are recorded.
- **Replication agreement**: each question runs several times per setup. The
  headline answers are extracted and compared — a trustworthy pipeline gives
  the same answer to the same question from the same data.

## 2. The two question banks

**`seeds_f1.json` — real questions, no answer key.** Eight open analytical
questions on a real Formula 1 dataset (any race-results CSV with drivers,
constructors, seasons, grid and finishing positions works). Open questions
have no "correct answer", so each one carries a checklist of what a competent
analysis must address (e.g. "car quality controlled via teammate comparison,
not raw totals"). The judge grades against that checklist. This measures
whether the analysis was done well — it cannot detect a well-presented wrong
number.

**`seeds_planted.json` — synthetic questions with known answers.** For these,
`planted_dgp.py` generates an artificial racing dataset from a process we
control, so the truth is known exactly:

- a true skill ranking with a planted hard case: the genuinely best
  driver has a short career in mid-grid cars, while a nearly-as-good
  rival has a long career in the best cars — volume- and results-based
  analyses name the wrong driver, within-team designs find the right one,
- a home advantage of known size, twice as strong in the early era,
- a nationality effect that is exactly zero (any "nation X produces better
  drivers" conclusion is a false discovery — this tests whether the pipeline
  says "nothing there" when nothing is there),
- a known reliability trend (failure rates fall across seasons; worse cars
  fail more),
- and three data traps: a decoy `fan_rating` column that actually tracks car
  quality (a seductive shortcut to "skill" that is wrong at the top), a leaky
  `career_podiums_total` column (future information and a pure volume
  metric), and one fully duplicated season (an ingestion fault a data-profiling
  pass should catch).

`score_recovery.py` grades briefings directly against these truths. Generate
the data with:

    python3 eval/planted_dgp.py --out eval/planted

This writes `synth_f1.csv` (give this to the runs) and `truth.json`
(**never** place it in the dataset directory or mention it in a question).

Both banks are frozen once scoring starts: editing a question or a checklist
after the first scored run invalidates every comparison that uses it.

## 3. How a test run works

A **configuration** is a name plus a repo checkout plus a model setup:

    {"name": "grok45", "repo": ".",
     "investigator_model": "openrouter:x-ai/grok-4.5",
     "executor_model": "ollama:glm-5.2:cloud",
     "synth_model": "ollama:glm-5.2:cloud"}

`run_matrix.py` runs every question in the bank, several times
(`--reps`, default 3), through each configuration. Results land in:

    <out>/<configuration name>/<question id>/rep<k>/
        briefing.md               <- the product being graded
        logs/<ts>/run_log.json    <- full record, used by the mechanical scorer
        mech_scores.json          <- written by score_mechanical.py
        judge_scores.json         <- written by judge.py
        recovery.json             <- written by score_recovery.py (planted only)
        condition.json            <- (one per configuration) the recorded setup

Runs are resumable: re-invoking skips any cell that already has a briefing,
so an interrupted batch continues where it stopped, and new configurations
can be added to the same results directory later. A configuration name is a
contract — reusing a name with a different model setup is refused, because
mixed results would be uninterpretable.

Repeats matter: single runs of the same question legitimately vary, so never
compare setups on one run each.

## 4. Reading the report

`report.py` prints, for each configuration, a **scorecard**: judged scores
per question, replication agreement, and the mechanical summary (grounding,
redundancy, iterations, cost per run).

With two or more configurations it adds a **comparison** against a baseline
(`--baseline <name>`, default: alphabetically first). For each question it
averages the repeats, takes the difference (candidate minus baseline), and
summarizes across questions with a confidence interval. Two conclusions are
reported in plain terms:

- **"No worse"**: the candidate's overall score is not more than half a point
  below the baseline, with the interval clearly excluding a bigger drop.
  This is the primary question when the candidate is a cheaper or faster setup.
- **"Better on rigor"**: the candidate beats the baseline on confounders or
  calibration with an interval clearly above zero.

The report also shows head-to-head results: the judge is given both
briefings for the same question (order randomized, labels hidden) and picks
the better one per axis. At small sample sizes this head-to-head view is
often the most sensitive signal.

The mechanical table is the cross-check on the judge: if the judge prefers a
setup whose grounding rate dropped, trust the grounding rate and read the
briefings yourself.

## 5. Recipes

Search is kept off in all recipes (`--no-search` is the default): live web
results make runs incomparable.

**Score one setup** (capture a baseline scorecard):

    python3 eval/run_matrix.py --repo . --name grok45 \
        --investigator-model openrouter:x-ai/grok-4.5 \
        --executor-model ollama:glm-5.2:cloud \
        --dataset data/f1.csv --seeds eval/seeds_f1.json \
        --out eval/results --judge-model <strong-model-not-in-the-pipeline>

**Compare model setups** — either add a second configuration later into the
same `--out` (the report switches to comparison automatically):

    python3 eval/run_matrix.py --repo . --name opus \
        --investigator-model anthropic:claude-opus-4-8 \
        --executor-model ollama:glm-5.2:cloud \
        --dataset data/f1.csv --seeds eval/seeds_f1.json \
        --out eval/results --judge-model <judge> --baseline grok45

or declare 2-3 configurations up front in a JSON file and run once:

    python3 eval/run_matrix.py --conditions conditions.json \
        --dataset data/f1.csv --seeds eval/seeds_f1.json \
        --out eval/results --judge-model <judge>

**Compare code versions**: same as above, but the configurations point to
different repo checkouts and share identical models.

**Ground-truth check** (can the pipeline recover known answers?):

    python3 eval/planted_dgp.py --out eval/planted
    python3 eval/run_matrix.py --repo . --name grok45 \
        --investigator-model ... --executor-model ... \
        --dataset eval/planted/synth_f1.csv --seeds eval/seeds_planted.json \
        --truth eval/planted/truth.json \
        --out eval/results_planted --judge-model <judge>

   With --truth given, recovery is scored automatically at the end and saved
   as recovery_report.txt next to report.txt in the results directory. (To
   re-score later without re-running: eval/score_recovery.py <results>
   --truth <truth.json>.)

**Re-score or re-report at any time** (no new runs):

    python3 eval/score_mechanical.py eval/results
    python3 eval/judge.py eval/results --judge-model <judge>
    python3 eval/report.py eval/results --baseline grok45

## 6. Rules that keep the results valid

1. Freeze the question banks before the first scored run; never edit after.
2. Same `--iterations`, same dataset, search off, for every configuration in
   a comparison.
3. The judge model must not be a model used inside any configuration being
   compared (a model grading its own work is biased toward itself).
4. `truth.json` never sits in or near the dataset given to runs.
5. Never reuse a configuration name for a different setup (enforced).
6. Recovery checks on planted questions are string-based; before quoting a
   surprising recovery number, open the flagged briefings and confirm the
   grading by eye.
7. One dataset family means conclusions hold for that task family; to
   generalize, add a second dataset with its own question bank.

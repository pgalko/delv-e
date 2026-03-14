"""
Auto-explore module for autonomous recursive exploration.
Handles the exploration loop, result evaluation, and adaptive branching.

Modifications from original BambooAI version:
  - Removed: Supabase auth, orchestrator heartbeat, billing, webui queue pushes
  - Changed: _get_dataset_schema uses pandas directly (no bambooai.utils)
  - Changed: format_trajectory_for_synthesis uses node data (no interaction_store)
  - Changed: _generate_branching_questions sets chain_id unconditionally
  - Added: iteration context setting, file writing hooks, synthesis step
  - Added: finding_summary compaction, tiered history, score-weighted synthesis,
           phase oscillation cooldown, evaluator-score-driven CONVERGING
"""

import json
import os
import re
import time
import traceback

import style

from logger_config import get_logger
logger = get_logger(__name__)


class AutoExplorer:
    """
    Autonomous exploration engine that recursively explores analytical space.
    Operates on an ExplorationEngine instance (or any object providing the
    self.bamboo.* interface).
    """

    PHASE_INSTRUCTIONS = {
        "MAPPING": (
            "EXPLORE",
            "We are in MAPPING phase — surveying the landscape of the data. "
            "We need to discover what dimensions, patterns, and structures exist before "
            "committing to any direction. Generate questions that open new angles, examine "
            "unexplored variables, and build a broad understanding of what this dataset contains. "
            "Prioritize coverage over depth. The research model's 'Biggest Gap' field is especially "
            "relevant — try to address gaps in our current understanding."
        ),
        "PURSUING": (
            "EXPLOIT",
            "We are in PURSUING phase — we have found something interesting and need to go deep. "
            "Generate questions that add precision to our most promising finding: find exact thresholds, "
            "identify the specific conditions under which the pattern holds, quantify effect sizes, "
            "and test whether the relationship is robust across subgroups. The goal is to turn a "
            "suggestive finding into a well-evidenced conclusion."
        ),
        "CONVERGING": (
            "REFLECT",
            "We are in CONVERGING phase — our findings have been consistent but are no longer "
            "updating our understanding. Before moving on, we need to pressure-test what we think "
            "we know. Generate questions that look for disconfirming evidence, check consistency "
            "between findings, test whether patterns hold in opposite conditions, and ask whether "
            "simpler explanations might account for what we have observed. If our understanding "
            "survives these tests, we can consider this thread complete. If not, we may need to reframe."
        ),
        "REFRAMING": (
            "EXPLORE",
            "We are in REFRAMING phase — our current line of inquiry has hit a contradiction or "
            "dead end. We need to step back and look at the data from a fundamentally different angle. "
            "Generate questions that challenge the assumptions underlying our previous approach. "
            "Consider: are we looking at the right variables? The right level of aggregation? "
            "The right time window? The research model's contradictions and weak points should "
            "guide what new directions to try."
        ),
    }

    def __init__(self, bamboo_instance):
        self.bamboo = bamboo_instance

        # Tree-based exploration state
        self.insight_tree = {}
        self.active_branch_id = None
        self.dormant_branches = []
        self.question_pool = []
        self.root_node_id = None

        # Research model state
        self.research_model = ""
        self.seed_question = ""  # original exploration question — used as relevance anchor
        self.current_phase = "MAPPING"
        self.phase_history = []
        self.model_impact_history = []
        self.evaluator_score_history = []   # FIX 4: track evaluator scores for phase decisions
        self.biggest_gap_history = []        # last N "Biggest Gap" texts from research model
        self.stagnation_count = 0            # consecutive iterations evaluator flagged stagnation

        # Reset branch endpoints
        self.bamboo.branch_endpoints = []

    @property
    def kill_signal(self):
        return self.bamboo.kill_signal

    @property
    def chain_id(self):
        return self.bamboo.chain_id

    @chain_id.setter
    def chain_id(self, value):
        self.bamboo.chain_id = value

    def _llm_stream_silent(self, messages, agent):
        """Run llm_stream in silent mode and return captured output."""
        output_manager = self.bamboo.output_manager
        output_manager.set_silent(True)

        try:
            response = self.bamboo.llm_stream(
                self.bamboo.prompts,
                self.bamboo.log_and_call_manager,
                output_manager,
                messages,
                agent=agent,
                chain_id=self.chain_id,
                reasoning_models=self.bamboo.reasoning_models,
                reasoning_effort="low",
                stop_event=self.bamboo._stop_event,
            )
            if not response or (isinstance(response, str) and response.strip() == ''):
                response = output_manager.get_captured_output()
            return response
        except Exception as e:
            logger.warning(f"LLM call failed for {agent}: {e}")
            return ""
        finally:
            output_manager.set_silent(False)

    def _evaluate_results_comparative(self, solutions_data, remaining_iterations=0, max_iterations=10):
        """
        Compare multiple solution results and select the most analytically valuable one.
        """
        if self.kill_signal:
            return 0, [5] * len(solutions_data), [], False, "Interrupted", None, [''] * len(solutions_data)

        # Build exploration state context
        if self.dormant_branches:
            dormant_scores = [
                self.insight_tree[nid]['quality_score']
                for nid in self.dormant_branches
                if nid in self.insight_tree
            ]
            best_dormant_score = max(dormant_scores) if dormant_scores else 0
            dormant_info = f"{len(self.dormant_branches)} branch(es) available, best score: {best_dormant_score}"
        else:
            best_dormant_score = 0
            dormant_info = "None"

        exploration_state = f"""**Current Exploration State:**
- Dormant branches: {dormant_info}
- Remaining iterations: {remaining_iterations} of {max_iterations}

{self._get_exploration_history()}"""

        research_model_context = self.research_model if self.research_model else "(No model yet)"

        # Build solutions block
        solutions_parts = []
        for i, sol in enumerate(solutions_data, 1):
            has_code = bool(sol['code'] and sol['code'].strip())
            has_results = bool(sol['results'] and str(sol['results']).strip())

            if has_code and has_results:
                result_text = str(sol['results'])
            elif has_code:
                result_text = "Code executed but no results produced"
            else:
                result_text = str(sol.get('text_answer', 'No output'))

            error_note = " (Error occurred during execution)" if sol['error_occurred'] else ""
            solutions_parts.append(f"""**Solution {i}:**
Question: {sol['question']}
Result:{error_note}
{result_text}
""")

        solutions_block = "\n---\n".join(solutions_parts)

        eval_prompt = self.bamboo.prompts.result_evaluator.format(
            seed_question=self.seed_question,
            exploration_state=exploration_state,
            solutions_block=solutions_block,
            research_model=research_model_context,
        )

        eval_messages = [{"role": "user", "content": eval_prompt}]
        response = self._llm_stream_silent(eval_messages, 'Result Evaluator')

        # Parse response
        scores = [5] * len(solutions_data)
        if 'SCORES:' in response.upper():
            scores_line = response.upper().split('SCORES:')[1].split('\n')[0]
            score_matches = re.findall(r'\d+', scores_line)
            for i, match in enumerate(score_matches[:len(solutions_data)]):
                scores[i] = max(1, min(10, int(match)))

        selected_index = 0
        if 'SELECTED:' in response.upper():
            selected_line = response.upper().split('SELECTED:')[1].split('\n')[0]
            match = re.search(r'\d+', selected_line)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(solutions_data):
                    selected_index = idx

        keep_dormant_indices = []
        if 'KEEP_DORMANT:' in response.upper():
            dormant_line = response.upper().split('KEEP_DORMANT:')[1].split('\n')[0]
            if 'NONE' not in dormant_line.upper():
                for match in re.findall(r'\d+', dormant_line):
                    idx = int(match) - 1
                    if 0 <= idx < len(solutions_data) and idx != selected_index:
                        keep_dormant_indices.append(idx)

        is_stagnating = False
        if 'STAGNATION:' in response.upper():
            stag_line = response.upper().split('STAGNATION:')[1].split('\n')[0]
            is_stagnating = 'YES' in stag_line.upper()

        # Parse SUMMARIES — one-sentence summary per solution, pipe-separated
        summaries = [''] * len(solutions_data)
        if 'SUMMARIES:' in response:
            summary_text = response.split('SUMMARIES:')[1]
            # Take until next field
            for end in ['SELECTED:', 'KEEP_DORMANT:', 'STAGNATION:', 'REASON:', '\n\n']:
                idx = summary_text.find(end)
                if idx > 0:
                    summary_text = summary_text[:idx]
                    break
            parts = [s.strip() for s in summary_text.split('|')]
            for i, part in enumerate(parts[:len(solutions_data)]):
                if part and len(part) > 5:
                    summaries[i] = part[:200]

        reason = "No reason provided"
        if 'REASON:' in response:
            reason = response.split('REASON:')[1].split('\n')[0].strip()

        follow_up_angle = None
        if 'FOLLOW_UP_ANGLE:' in response:
            angle_text = response.split('FOLLOW_UP_ANGLE:')[1].split('\n')[0].strip()
            if angle_text and angle_text.lower() not in ['', 'none', 'n/a']:
                follow_up_angle = angle_text

        return selected_index, scores, keep_dormant_indices, is_stagnating, reason, follow_up_angle, summaries

    def _generate_branching_questions(self, use_chain_id=None, follow_up_hint=None):
        """
        Generate 5 branching questions. Phase-driven: all 5 are the same type.
        """
        if self.kill_signal:
            return [], {}, ""

        # --- CHANGED: set chain_id unconditionally (was inside webui block) ---
        if use_chain_id is not None:
            self.chain_id = use_chain_id
        else:
            self.chain_id = int(time.time())

        # Build context
        exploration_context = self._get_exploration_history()
        # FIX 5: use slim schema for question generation (non-code agent)
        dataset_schema = self._get_dataset_schema_slim()

        hint_section = ""
        if follow_up_hint:
            hint_section = f"\n**Promising direction:** {follow_up_hint}\n"

        # Periodic grounding check — every 8 iterations, nudge the QG to step back
        # and consider whether basic decompositions or direct comparisons were missed.
        current_iter = getattr(self, '_current_iteration', 0)
        if current_iter > 0 and current_iter % 8 == 0:
            hint_section += (
                "\n**Grounding check:** Step back from the current thread and consider "
                "the original question. Are there basic decompositions, direct comparisons, "
                "or simple summary statistics that have not been computed yet? At least one "
                "of your 5 questions should address a fundamental aspect of the original "
                "question using a straightforward analytical approach.\n"
            )

        phase_mode, phase_instruction = self.PHASE_INSTRUCTIONS.get(
            self.current_phase,
            ("EXPLORE", "Generate exploratory questions."),
        )

        model_context = self.research_model if self.research_model else "(No model yet — first iteration)"

        phase_prompt = self.bamboo.prompts.ideas_explorer_auto.format(
            phase_mode=phase_mode,
            current_phase=self.current_phase,
            phase_instruction=phase_instruction,
            research_model=model_context,
        )

        # Complete question log (prevents circular exploration)
        all_questions_log = self.bamboo.message_manager.format_all_questions()

        gen_prompt = f"""Based on our exploration so far:

{dataset_schema}

{exploration_context}

{all_questions_log}
{hint_section}

{phase_prompt}"""

        gen_messages = [{"role": "user", "content": gen_prompt}]

        agent = 'Question Generator'
        questions_response = self._llm_stream_silent(gen_messages, agent)

        questions, categories = self._parse_questions_with_categories(questions_response)

        return questions, categories, questions_response

    def _select_best_questions(self, questions, categories, context_hint=None, num_to_select=1):
        """Use LLM to select the most promising questions from a pool."""
        if not questions:
            return [], []
        if self.kill_signal:
            return [], []

        num_to_select = min(num_to_select, len(questions))

        formatted_questions = []
        for i, q in enumerate(questions):
            cat = categories.get(i, 'exploit')
            formatted_questions.append(f"{i+1}. [{cat.upper()}] {q}")

        pool_section = ""
        if self.question_pool:
            pool_lines = []
            for j, pq in enumerate(self.question_pool[-5:]):
                pool_idx = len(questions) + j + 1
                pool_lines.append(f"{pool_idx}. [POOL-{pq['category'].upper()}] {pq['question']}")
            pool_section = f"""

**Question Pool (from previous iterations):**
{chr(10).join(pool_lines)}

You may select from EITHER the new questions (1-{len(questions)}) OR the pool ({len(questions)+1}-{len(questions)+len(pool_lines)}).
Pool questions are valuable when current direction feels exhausted."""

        exploration_context = self._get_exploration_history()
        # Append complete question log so selector can avoid duplicates
        all_questions_log = self.bamboo.message_manager.format_all_questions()
        exploration_history = exploration_context + "\n\n" + all_questions_log if all_questions_log else exploration_context

        context_section = ""
        if context_hint:
            context_section = f"\n**Promising direction from recent result:** {context_hint}\n"

        research_model_context = self.research_model if self.research_model else "(No model yet)"

        selection_prompt = self.bamboo.prompts.question_selector.format(
            exploration_history=exploration_history,
            questions=chr(10).join(formatted_questions) + pool_section,
            context_hint=context_section,
            num_to_select=num_to_select,
            research_model=research_model_context,
            current_phase=self.current_phase,
        )

        select_messages = [{"role": "user", "content": selection_prompt}]
        selection = self._llm_stream_silent(select_messages, 'Question Selector')

        # Parse selection
        selected_questions = []
        selected_categories = []
        selected_indices = []

        try:
            matches = re.findall(r'\d+', selection.strip())
            for match in matches:
                idx = int(match) - 1
                if idx < len(questions):
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        selected_questions.append(questions[idx])
                        selected_categories.append(categories.get(idx, 'exploit'))
                else:
                    pool_idx = idx - len(questions)
                    pool_subset = self.question_pool[-5:]
                    if 0 <= pool_idx < len(pool_subset):
                        pq = pool_subset[pool_idx]
                        selected_questions.append(pq['question'])
                        selected_categories.append(pq['category'])
                        self.question_pool.remove(pq)
                if len(selected_questions) >= num_to_select:
                    break
        except (AttributeError, ValueError):
            pass

        # Fallback: if nothing parsed at all, take first N
        if not selected_questions:
            selected_indices = list(range(min(num_to_select, len(questions))))
            selected_questions = [questions[i] for i in selected_indices]
            selected_categories = [categories.get(i, 'exploit') for i in selected_indices]

        # Pad: if parsed fewer than needed, fill from remaining questions
        if len(selected_questions) < num_to_select:
            for i in range(len(questions)):
                if i not in selected_indices and len(selected_questions) < num_to_select:
                    selected_indices.append(i)
                    selected_questions.append(questions[i])
                    selected_categories.append(categories.get(i, 'exploit'))

        # Store unselected in pool
        for i, q in enumerate(questions):
            if i not in selected_indices:
                self.question_pool.append({
                    'question': q,
                    'source_branch_id': self.active_branch_id,
                    'category': categories.get(i, 'exploit'),
                    'iteration_added': getattr(self, '_current_iteration', 0),
                })
        self._trim_question_pool()

        return selected_questions, selected_categories

    def _parse_questions_with_categories(self, questions_response):
        """Parse numbered questions from LLM response."""
        questions = []
        phase_mode = self.PHASE_INSTRUCTIONS.get(
            self.current_phase, ("EXPLORE",)
        )[0].lower()

        if not questions_response or not questions_response.strip():
            return [], {}

        lines = questions_response.split('\n')
        current_q = []

        for line in lines:
            stripped = line.strip()
            # Match numbering: "1.", "1)", "1:", "## 1.", "- 1.", etc.
            is_numbered = re.match(r'^(?:[-•*#]*\s*)?([1-9])[.):]', stripped)

            if is_numbered:
                if current_q:
                    question_text = ' '.join(current_q).strip()
                    if len(question_text) > 10:
                        questions.append(question_text)
                # Strip the number prefix
                cleaned = re.sub(r'^(?:[-•*#]*\s*)?[1-9][.):][\s]*', '', stripped)
                current_q = [cleaned] if cleaned else []
            elif current_q and stripped and stripped != '---':
                current_q.append(stripped)

        if current_q:
            question_text = ' '.join(current_q).strip()
            if len(question_text) > 10:
                questions.append(question_text)

        categories = {i: phase_mode for i in range(len(questions))}
        return questions, categories

    def _interpret_and_update_model(self, winning_solution, quality_score,
                                     solutions_data=None, selected_index=0, scores=None):
        """Interpret the latest result and update the research model.

        Returns:
            (updated_model, model_impact, contradiction, thread_completed, result_digest)
        """
        if self.kill_signal:
            return self.research_model, "LOW", False, False, ""

        result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"
        current_model = self.research_model if self.research_model else "(No model yet — this is the first result. Initialize the model from scratch.)"

        parallel_context = ""
        if solutions_data and len(solutions_data) > 1 and scores:
            parts = ["**Other parallel results this iteration (not selected):**"]
            for i, sol in enumerate(solutions_data):
                if i == selected_index:
                    continue
                result_text = str(sol['results']) if sol['results'] else "No results"
                parts.append(
                    f"- [{scores[i]}/10] {sol['question']}\n"
                    f"  Finding: {result_text}"
                )
            parallel_context = '\n'.join(parts)

        interpret_prompt = self.bamboo.prompts.research_model_updater.format(
            current_model=current_model,
            question=winning_solution['question'],
            score=quality_score,
            result_summary=result_summary,
            parallel_results=parallel_context,
        )

        interpret_messages = [{"role": "user", "content": interpret_prompt}]
        response = self._llm_stream_silent(interpret_messages, 'Research Interpreter')

        # Parse MODEL_IMPACT
        model_impact = "MEDIUM"
        if 'MODEL_IMPACT:' in response.upper():
            impact_line = response.upper().split('MODEL_IMPACT:')[1].split('\n')[0]
            if 'HIGH' in impact_line:
                model_impact = "HIGH"
            elif 'LOW' in impact_line:
                model_impact = "LOW"

        # Parse CONTRADICTION
        contradiction = False
        if 'CONTRADICTION:' in response.upper():
            contra_line = response.upper().split('CONTRADICTION:')[1].split('\n')[0]
            contradiction = 'YES' in contra_line

        # Parse THREAD_COMPLETED
        thread_completed = False
        if 'THREAD_COMPLETED:' in response.upper():
            thread_line = response.upper().split('THREAD_COMPLETED:')[1].split('\n')[0]
            thread_completed = 'YES' in thread_line

        # Parse RESULT_DIGEST — RI-curated extract of key numbers (3-5 lines)
        result_digest = ""
        if 'RESULT_DIGEST:' in response:
            digest_text = response.split('RESULT_DIGEST:')[1]
            for end_marker in ['UPDATED_MODEL:', '\n\n\n']:
                if end_marker in digest_text:
                    digest_text = digest_text.split(end_marker)[0]
                    break
            result_digest = digest_text.strip()
            # Safety cap — this should be 3-5 lines, ~500-800 chars
            if len(result_digest) > 1000:
                result_digest = result_digest[:997] + "..."

        # Extract updated model
        updated_model = self.research_model
        if 'UPDATED_MODEL:' in response:
            model_text = response.split('UPDATED_MODEL:')[1]
            if 'END_MODEL' in model_text:
                model_text = model_text.split('END_MODEL')[0]
            model_text = model_text.strip()
            if model_text:
                updated_model = model_text

        return updated_model, model_impact, contradiction, thread_completed, result_digest

    def _determine_phase(self, model_impact, contradiction, thread_completed,
                         evaluator_stagnating=0):
        """Rule-based phase transition with stagnation detection.

        Stagnation signals:
        - evaluator_stagnating: consecutive iterations the evaluator flagged STAGNATION: YES.
          2+ consecutive triggers CONVERGING — this is the primary structural signal for
          diminishing returns, since the evaluator independently assesses each result.
        """

        # Oscillation cooldown — checked FIRST so it can override everything.
        current_iter = getattr(self, '_current_iteration', 0)
        if current_iter >= 10:
            recent_changes = sum(
                1 for i, _o, _n in self.phase_history
                if i >= current_iter - 4
            )
            if recent_changes >= 3:
                logger.info("Phase → PURSUING (oscillation cooldown: %d transitions in 5 iterations)", recent_changes)
                return "PURSUING"

        # Stagnation detection — evaluator has flagged consecutive stagnation.
        # Only activate after iteration 5 to let early exploration breathe.
        if current_iter >= 5 and self.current_phase != "CONVERGING":
            if evaluator_stagnating >= 2:
                logger.info("Phase → CONVERGING (evaluator flagged stagnation %d consecutive times)", evaluator_stagnating)
                return "CONVERGING"

        if contradiction:
            latest_score = self.evaluator_score_history[-1] if self.evaluator_score_history else 5
            if latest_score <= 5:
                logger.info("Phase → REFRAMING (contradiction confirmed by low evaluator score %d)", latest_score)
                return "REFRAMING"
            else:
                logger.info("Contradiction overridden — evaluator score %d suggests refinement, not reversal", latest_score)

        if thread_completed:
            logger.info("Phase → MAPPING (thread completed)")
            return "MAPPING"

        if self.current_phase == "CONVERGING":
            recent_lows = sum(1 for m in self.model_impact_history[-3:] if m == "LOW")
            if recent_lows >= 3:
                logger.info("Phase → MAPPING (CONVERGING exhausted)")
                return "MAPPING"

        if model_impact == "HIGH":
            logger.info("Phase → PURSUING (high model impact)")
            return "PURSUING"

        if model_impact == "LOW":
            recent = self.model_impact_history[-3:] if len(self.model_impact_history) >= 3 else self.model_impact_history
            low_count = sum(1 for m in recent if m == "LOW")
            if low_count >= 2:
                logger.info("Phase → CONVERGING (sustained low model impact)")
                return "CONVERGING"

        # Evaluator-score-driven CONVERGING trigger
        if self.current_phase in ("PURSUING", "REFRAMING") and len(self.evaluator_score_history) >= 3:
            recent_scores = self.evaluator_score_history[-3:]
            if all(s <= 6 for s in recent_scores):
                logger.info("Phase → CONVERGING (3 consecutive winning scores ≤ 6: %s)", recent_scores)
                return "CONVERGING"

        if self.current_phase == "REFRAMING" and model_impact == "MEDIUM":
            logger.info("Phase → MAPPING (reframing produced new angle)")
            return "MAPPING"

        logger.info(f"Phase → {self.current_phase} (maintained)")
        return self.current_phase

    # ══════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════

    def run(self, seed_question, initial_image=None, max_iterations=5, num_parallel_solutions=2,
            interactive=False, resumed_state=None):
        """Run the autonomous exploration loop.

        Args:
            seed_question: Starting question (or new direction when resuming).
            max_iterations: Number of iterations to run. When resuming, this is
                *additional* iterations on top of what was already completed.
            resumed_state: If provided, a dict loaded from state.json to restore
                the exploration from a previous run.
        """

        num_parallel_solutions = max(2, min(5, num_parallel_solutions))

        # Store original settings
        original_user_feedback = self.bamboo.user_feedback
        original_analyst_system_content = self.bamboo.message_manager.select_analyst_messages[0]["content"]
        original_max_errors = self.bamboo.MAX_ERROR_CORRECTIONS

        # Switch to auto-explore mode
        self.bamboo.user_feedback = False
        self.bamboo.message_manager.select_analyst_messages[0]["content"] = self.bamboo.prompts.analyst_selector_system_auto
        self.bamboo.MAX_ERROR_CORRECTIONS = 3

        try:
            current_image = initial_image

            if resumed_state:
                # ── RESTORE from checkpoint ──
                self._restore_checkpoint(resumed_state)
                start_iteration = resumed_state['iterations_completed']
                max_iterations = start_iteration + max_iterations  # additive
                last_solution_chain = resumed_state.get('last_solution_chain')
                last_follow_up_angle = resumed_state.get('last_follow_up_angle')

                # The new seed_question becomes the first analysis in the resumed run
                current_questions = [seed_question]
                current_categories = ['exploit']
                # Keep original seed_question as the relevance anchor (restored from checkpoint)

                logger.info(f"Resuming from iteration {start_iteration}, "
                            f"running {max_iterations - start_iteration} more")
            else:
                # ── FRESH start ──
                current_questions = [seed_question]
                current_categories = ['exploit']
                last_solution_chain = None
                last_follow_up_angle = None

                self.insight_tree = {}
                self.active_branch_id = None
                self.dormant_branches = []
                self.question_pool = []
                self.root_node_id = None
                self.research_model = ""
                self.seed_question = seed_question  # store as relevance anchor
                self.current_phase = "MAPPING"
                self.phase_history = []
                self.model_impact_history = []
                self.evaluator_score_history = []
                self.biggest_gap_history = []
                self.stagnation_count = 0

                start_iteration = 0

            iteration = start_iteration
            self._current_iteration = start_iteration

            # --- Header ---
            df_shape = f"{self.bamboo.df.shape[0]:,} rows × {self.bamboo.df.shape[1]} cols"
            agent_model = self.bamboo.models.agent_model
            code_model = self.bamboo.models.code_model
            if interactive and not resumed_state:
                print(style.config_lines(df_shape, max_iterations, num_parallel_solutions,
                                         self.bamboo.output_dir, agent_model, code_model))
            else:
                extra = ""
                if resumed_state:
                    extra = f" (resuming from iteration {start_iteration})"
                print(style.splash_header(df_shape, max_iterations, num_parallel_solutions,
                                          self.bamboo.output_dir, agent_model, code_model))
                if extra:
                    print(f"    {style.DIM}{extra}{style.RESET}")
            print()

            logger.info(f"Starting auto-explore: {seed_question[:50]}... max_iterations={max_iterations}")

            while iteration < max_iterations:
                self._current_iteration = iteration

                if self.kill_signal:
                    break

                # --- ADDED: set iteration context on engine ---
                self.bamboo._iteration = iteration + 1
                self.bamboo._max_iterations = max_iterations
                self.bamboo._phase = self.current_phase

                # Determine questions to process
                if iteration == 0:
                    questions_to_process = current_questions[:1]
                    categories_to_process = current_categories[:1]
                elif iteration == 1:
                    questions_to_process = current_questions[:num_parallel_solutions + 1]
                    categories_to_process = current_categories[:num_parallel_solutions + 1]
                else:
                    questions_to_process = current_questions[:num_parallel_solutions]
                    categories_to_process = current_categories[:num_parallel_solutions]

                # --- Print iteration header ---
                print(style.iteration_bar(iteration + 1, max_iterations, self.current_phase))

                # Process each question
                solutions_data = []

                for q_idx, question in enumerate(questions_to_process):
                    if self.kill_signal:
                        break

                    # Restore to common starting point for parallel branches
                    if len(questions_to_process) > 1 and last_solution_chain is not None:
                        self.bamboo.message_manager.restore_interaction(
                            self.bamboo.thread_id,
                            last_solution_chain,
                        )

                    self.chain_id = int(time.time()) + q_idx
                    solution_chain_id = self.chain_id

                    cat = categories_to_process[q_idx] if q_idx < len(categories_to_process) else ""
                    print()
                    print(style.question_display(q_idx + 1, len(questions_to_process), cat, question))

                    # Process question
                    error_occurred = False
                    try:
                        self.bamboo._process_question(question, current_image if q_idx == 0 else None, None, None)
                    except Exception as e:
                        error_occurred = True
                        logger.warning(f"Execution error: {e}")
                        self.bamboo.output_manager.display_error(f"Execution error: {e}", chain_id=self.chain_id)

                    solutions_data.append({
                        'question': question,
                        'results': self.bamboo.message_manager.code_exec_results,
                        'code': self.bamboo.message_manager.last_code,
                        'text_answer': self.bamboo.message_manager.last_plan,
                        'chain_id': solution_chain_id,
                        'error_occurred': error_occurred,
                        'category': categories_to_process[q_idx] if q_idx < len(categories_to_process) else 'exploit',
                    })

                    self.bamboo.message_manager.reset_non_cumul_messages()
                    time.sleep(1)

                # Supporting chain
                self.chain_id = int(time.time())
                supporting_chain_id = self.chain_id

                # === EVALUATE ===
                if len(solutions_data) == 1:
                    selected_index = 0
                    scores = [7]
                    keep_dormant_indices = []
                    is_stagnating = False
                    reason = "Seed question — establishing baseline"
                    follow_up_angle = None
                    summaries = ['']  # no evaluator for seed; summary populated below
                    _is_seed = True
                else:
                    with style.spinner("Evaluating results"):
                        selected_index, scores, keep_dormant_indices, is_stagnating, reason, follow_up_angle, summaries = \
                            self._evaluate_results_comparative(
                                solutions_data,
                                remaining_iterations=max_iterations - iteration,
                                max_iterations=max_iterations,
                            )
                    _is_seed = False

                # FIX 4: track evaluator scores for phase decisions
                self.evaluator_score_history.append(scores[selected_index])
                if len(self.evaluator_score_history) > 15:
                    self.evaluator_score_history = self.evaluator_score_history[-15:]

                # === UPDATE TREE ===
                winning_solution = solutions_data[selected_index]
                result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"

                # Save the common parent BEFORE updating active_branch_id
                common_parent_id = self.active_branch_id

                new_node_id = self._add_node_to_tree(
                    question=winning_solution['question'],
                    result_summary=result_summary,
                    quality_score=scores[selected_index],
                    chain_id=winning_solution['chain_id'],
                    parent_id=common_parent_id,
                )
                # Store evaluator-generated summary on winning node
                if selected_index < len(summaries) and summaries[selected_index]:
                    self.insight_tree[new_node_id]['finding_summary'] = summaries[selected_index]

                self.active_branch_id = new_node_id
                last_solution_chain = winning_solution['chain_id']
                last_follow_up_angle = follow_up_angle

                # === INTERPRET & UPDATE MODEL ===
                with style.spinner("Updating research model"):
                    updated_model, model_impact, contradiction, thread_completed, result_digest = \
                        self._interpret_and_update_model(
                            winning_solution,
                            scores[selected_index],
                            solutions_data=solutions_data,
                            selected_index=selected_index,
                            scores=scores,
                        )

                # Store RI-generated result_digest on winning node
                if new_node_id in self.insight_tree:
                    if result_digest:
                        self.insight_tree[new_node_id]['result_digest'] = result_digest

                self.research_model = updated_model
                # Size guard: if model gets very large, nudge interpreter to consolidate
                if len(self.research_model) > 6000:
                    self.research_model += (
                        "\n\n**NOTE: This model is getting long. On next update, "
                        "consider consolidating Established Findings that overlap "
                        "and keeping the Narrative focused on the latest shift.**"
                    )
                self.model_impact_history.append(model_impact)
                if len(self.model_impact_history) > 10:
                    self.model_impact_history = self.model_impact_history[-10:]

                # === STAGNATION TRACKING ===
                # Track Biggest Gap evolution (stored for debugging)
                self._update_gap_stability(updated_model)

                # Evaluator signal: did the evaluator flag stagnation?
                if is_stagnating:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0

                # === DETERMINE PHASE ===
                old_phase = self.current_phase
                new_phase = self._determine_phase(
                    model_impact, contradiction, thread_completed,
                    evaluator_stagnating=self.stagnation_count,
                )
                if iteration == 0:
                    new_phase = "MAPPING"

                if new_phase != self.current_phase:
                    self.phase_history.append((iteration, self.current_phase, new_phase))

                self.current_phase = new_phase

                # === ADD ALL NON-WINNING SOLUTIONS TO TREE ===
                for idx in range(len(solutions_data)):
                    if idx == selected_index:
                        continue  # winner already added above
                    sol = solutions_data[idx]
                    node_id = self._add_node_to_tree(
                        question=sol['question'],
                        result_summary=str(sol['results']) if sol['results'] else "No results",
                        quality_score=scores[idx] if idx < len(scores) else 0,
                        chain_id=sol['chain_id'],
                        parent_id=common_parent_id,
                    )
                    # Store evaluator-generated summary
                    if idx < len(summaries) and summaries[idx]:
                        self.insight_tree[node_id]['finding_summary'] = summaries[idx]

                    if idx in keep_dormant_indices:
                        # Compose hypothesis label — describes what this branch was pursuing
                        summary = self.insight_tree[node_id].get('finding_summary', '')
                        label = sol['question'][:120]
                        if summary:
                            label += f" — Found: {summary[:150]}"
                        self.insight_tree[node_id]['hypothesis_label'] = label
                        self._add_to_dormant(node_id)
                    else:
                        self.insight_tree[node_id]['status'] = 'runner_up'

                # --- ADDED: write iteration summary ---
                self.bamboo.write_iteration_summary(
                    iteration=iteration + 1,
                    phase=old_phase,
                    solutions_data=solutions_data,
                    scores=scores,
                    selected_index=selected_index,
                    model_impact=model_impact,
                    contradiction=contradiction,
                    thread_completed=thread_completed,
                    new_phase=new_phase,
                    old_phase=old_phase,
                )

                # Increment
                iteration += 1

                # Save checkpoint after each completed iteration
                self._save_checkpoint(iteration, last_solution_chain, last_follow_up_angle)

                if iteration >= max_iterations:
                    # Show final evaluation summary even though we won't generate more questions
                    print(style.pipeline_summary(
                        selected_q=selected_index + 1,
                        selected_score=scores[selected_index],
                        reason=reason,
                        model_impact=model_impact,
                        old_phase=old_phase,
                        new_phase=new_phase,
                        n_questions=0,
                        phase_mode="—",
                        n_selected=0,
                        is_seed=_is_seed,
                    ))
                    break
                if self.kill_signal:
                    break

                # === PHASE-DRIVEN BRANCH DECISION ===
                forced_mapping = (
                    new_phase == "MAPPING"
                    and old_phase == "CONVERGING"
                    and not thread_completed
                    and not contradiction
                )

                if new_phase == "REFRAMING" or (new_phase == "MAPPING" and (thread_completed or forced_mapping)):
                    self.bamboo.branch_endpoints.append(last_solution_chain)

                    if new_phase == "REFRAMING":
                        reason_msg = "Contradiction detected — reframing"
                    elif forced_mapping:
                        reason_msg = "CONVERGING exhausted — pivoting"
                    else:
                        reason_msg = "Thread complete — new territory"
                    print(style.branch_event(reason_msg))

                    if self.dormant_branches:
                        new_branch_id = self._pop_best_dormant_branch()
                        new_branch = self.insight_tree[new_branch_id]
                        self.bamboo.message_manager.restore_interaction(
                            self.bamboo.thread_id,
                            new_branch['chain_id'],
                        )
                        self.active_branch_id = new_branch_id
                        last_solution_chain = new_branch['chain_id']
                        # Connect the dormant branch's hypothesis to the QG hint
                        last_follow_up_angle = new_branch.get('hypothesis_label') or new_branch['question'][:150]
                        print(style.branch_event(f"Switched to dormant branch (score: {style.score(new_branch['quality_score'])})"))

                # === GENERATE NEW QUESTIONS ===
                with style.spinner("Generating questions"):
                    new_questions, new_categories, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                        follow_up_hint=last_follow_up_angle,
                    )

                # Retry once if parsing failed
                if not new_questions:
                    logger.warning(f"Question generation returned empty. Raw response: {raw_response[:200] if raw_response else '(empty)'}")
                    print(style.error_msg("Question generation failed, retrying..."))
                    new_questions, new_categories, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                    )

                if not new_questions:
                    print(style.error_msg("Could not generate new questions after retry. Exploration complete."))
                    break

                # === SELECT QUESTIONS ===
                num_to_select = num_parallel_solutions + 1 if iteration == 1 else num_parallel_solutions
                with style.spinner("Selecting questions"):
                    selected_questions, selected_categories = self._select_best_questions(
                        new_questions,
                        new_categories,
                        context_hint=last_follow_up_angle,
                        num_to_select=num_to_select,
                    )
                if not selected_questions:
                    break

                # --- Print pipeline summary (evaluate → interpret → plan) ---
                phase_mode = self.PHASE_INSTRUCTIONS.get(self.current_phase, ("EXPLORE",))[0]
                print(style.pipeline_summary(
                    selected_q=selected_index + 1,
                    selected_score=scores[selected_index],
                    reason=reason,
                    model_impact=model_impact,
                    old_phase=old_phase,
                    new_phase=new_phase,
                    n_questions=len(new_questions),
                    phase_mode=phase_mode,
                    n_selected=len(selected_questions),
                    is_seed=_is_seed,
                ))

                current_questions = selected_questions
                current_categories = selected_categories
                current_image = None

            # === CAPTURE FINAL BRANCH ENDPOINT ===
            if last_solution_chain:
                self.bamboo.branch_endpoints.append(last_solution_chain)

            # === BUILD TRAJECTORY ===
            self.bamboo.exploration_trajectory = self._build_exploration_trajectory()

            # === SYNTHESIS ===
            with style.spinner("Generating synthesis report"):
                synthesis_text = self._generate_synthesis(seed_question)

            if not synthesis_text:
                print(f"  {style.YELLOW}✗ Synthesis failed (individual analyses saved in output/exploration/){style.RESET}")
                synthesis_text = ""

            # === EXPLORATION TREE ===
            print(style.exploration_tree(self.insight_tree, self.root_node_id))

            # === FINAL SUMMARY ===
            dormant_count = len(self.dormant_branches)
            avg_score = sum(n['quality_score'] for n in self.insight_tree.values()) / len(self.insight_tree) if self.insight_tree else 0

            print(style.final_box(
                iterations=iteration,
                analyses=len(self.insight_tree),
                avg=avg_score,
                dormant=dormant_count,
                phase_hist=self.phase_history,
                cost_str=self.bamboo.cost_tracker.report(),
                output_dir=self.bamboo.output_dir,
            ))

            # Write final files
            self.bamboo.write_final_outputs(
                research_model=self.research_model,
                phase_history=self.phase_history,
                synthesis_text=synthesis_text,
            )
            print()  # trailing newline

        finally:
            # Restore original settings
            self.bamboo.user_feedback = original_user_feedback
            self.bamboo.message_manager.select_analyst_messages[0]["content"] = original_analyst_system_content
            self.bamboo.MAX_ERROR_CORRECTIONS = original_max_errors

    def _generate_synthesis(self, seed_question, max_retries=3):
        """Generate final synthesis report using exploration trajectory."""
        trajectory = self.bamboo.exploration_trajectory
        if not trajectory:
            return None

        # Use full untruncated results for synthesis (not the 500-char tree summaries)
        full_store = self.bamboo.message_manager.full_results_store
        synthesis_context = format_trajectory_for_synthesis(
            trajectory, full_results_store=full_store
        )

        import datetime
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = self.bamboo.prompts.exploration_synthesis.format(
            today,
            synthesis_context,
            f"Synthesize all findings from the exploration seeded by: {seed_question}",
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, max_retries + 1):
            response = self._llm_stream_silent(messages, 'Synthesis Generator')
            if response and response.strip():
                return response
            if attempt < max_retries:
                wait = attempt * 5
                logger.warning(f"Synthesis attempt {attempt}/{max_retries} failed, retrying in {wait}s...")
                time.sleep(wait)

        logger.error("Synthesis generation failed after all retries")
        return None

    ##################
    # Helper Methods #
    ##################

    _node_counter = 0

    def _generate_node_id(self):
        AutoExplorer._node_counter += 1
        return f"node_{int(time.time() * 1000)}_{AutoExplorer._node_counter}"

    def _extract_biggest_gap(self, model_text):
        """Extract the Biggest Gap section from the research model text."""
        if not model_text or '## Biggest Gap' not in model_text:
            return ""
        gap_text = model_text.split('## Biggest Gap')[1]
        # Take until next section header or end
        for marker in ['## Narrative', '## Phase History', '## ']:
            idx = gap_text.find(marker)
            if idx > 0:
                gap_text = gap_text[:idx]
                break
        return gap_text.strip()

    def _update_gap_stability(self, updated_model):
        """Track whether the Biggest Gap has changed. Returns True if gap is stale."""
        current_gap = self._extract_biggest_gap(updated_model)
        if not current_gap:
            return False

        self.biggest_gap_history.append(current_gap)
        # Keep last 5
        if len(self.biggest_gap_history) > 5:
            self.biggest_gap_history = self.biggest_gap_history[-5:]

        if len(self.biggest_gap_history) < 2:
            return False

        # Compare current gap to previous — use simple word overlap ratio
        # rather than exact string match, since phrasing may change slightly
        prev_gap = self.biggest_gap_history[-2]
        prev_words = set(prev_gap.lower().split())
        curr_words = set(current_gap.lower().split())

        if not prev_words or not curr_words:
            return False

        overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
        # > 70% word overlap means the gap hasn't substantively changed
        return overlap > 0.70

    def _add_node_to_tree(self, question, result_summary, quality_score, chain_id, parent_id=None):
        node_id = self._generate_node_id()
        # Extract the results block if present.
        if result_summary:
            start_marker = "###RESULTS_START###"
            end_marker = "###RESULTS_END###"
            s_idx = result_summary.find(start_marker)
            e_idx = result_summary.find(end_marker)
            if s_idx >= 0 and e_idx > s_idx:
                result_summary = result_summary[s_idx + len(start_marker):e_idx].strip()
            elif len(result_summary) > 800:
                result_summary = result_summary[:800] + "\n[...truncated]"
        self.insight_tree[node_id] = {
            'question': question,
            'result_summary': result_summary,
            'quality_score': quality_score,
            'parent_id': parent_id,
            'children_ids': [],
            'status': 'active',
            'depth': 0 if parent_id is None else self.insight_tree[parent_id]['depth'] + 1,
            'chain_id': chain_id,
            'finding_summary': '',  # populated by evaluator SUMMARIES
            'result_digest': '',   # populated by RI — key numbers for full-detail history tier
            'hypothesis_label': '',  # populated when marked dormant — describes the analytical direction
        }
        if parent_id and parent_id in self.insight_tree:
            self.insight_tree[parent_id]['children_ids'].append(node_id)
        if self.root_node_id is None:
            self.root_node_id = node_id
        return node_id

    def _get_exploration_history(self, max_entries=40, full_detail_count=15):
        """Build exploration history with tiered compaction.

        Three tiers of detail:
        - Compact tier (older entries): question + finding_summary (one sentence)
        - Full tier (recent entries): question + result_digest (RI-curated key numbers)
        - Fallback: raw result_summary when RI-generated fields are unavailable
        """
        if not self.insight_tree:
            return ""
        all_nodes = sorted(self.insight_tree.values(), key=lambda n: n['chain_id'])
        recent = all_nodes[-max_entries:]

        # Split into compact (older) and full-detail (recent)
        if len(recent) > full_detail_count:
            compact_nodes = recent[:-full_detail_count]
            full_nodes = recent[-full_detail_count:]
        else:
            compact_nodes = []
            full_nodes = recent

        history_parts = ["**Exploration History (all branches):**"]

        if compact_nodes:
            history_parts.append(f"\n*Earlier analyses (compacted, {len(compact_nodes)} entries):*")
            for node in compact_nodes:
                score = node['quality_score']
                question = node['question']
                status_marker = " [DORMANT]" if node['status'] == 'dormant' else ""

                # Use finding_summary if available, else first line of result
                summary = node.get('finding_summary', '')
                if not summary:
                    finding = node['result_summary'] if node['result_summary'] else "No results"
                    summary = finding.split('\n')[0][:200]

                history_parts.append(
                    f"- [{score}/10]{status_marker} Q: {question}\n"
                    f"  → {summary}"
                )

        if full_nodes:
            if compact_nodes:
                history_parts.append(f"\n*Recent analyses ({len(full_nodes)} entries):*")
            for node in full_nodes:
                score = node['quality_score']
                question = node['question']
                status_marker = " [DORMANT]" if node['status'] == 'dormant' else ""

                # Only winning (active) nodes with a digest get full-detail display.
                # Non-winning nodes (runner_up, dormant, abandoned) use compact format
                # even in the recent tier — their raw result_summaries are too verbose
                # and they're preserved in full_results_store for synthesis anyway.
                digest = node.get('result_digest', '')
                if digest and node['status'] == 'active':
                    history_parts.append(
                        f"- [{score}/10]{status_marker} Q: {question}\n"
                        f"  Finding: {digest}"
                    )
                else:
                    summary = node.get('finding_summary', '')
                    if not summary:
                        finding = node['result_summary'] if node['result_summary'] else "No results"
                        summary = finding.split('\n')[0][:200]
                    history_parts.append(
                        f"- [{score}/10]{status_marker} Q: {question}\n"
                        f"  → {summary}"
                    )

        if self.phase_history:
            transitions = [f"{old}→{new} (iter {i})" for i, old, new in self.phase_history]
            history_parts.append(f"\n**Phase transitions:** {' | '.join(transitions)}")
        return '\n'.join(history_parts)

    def _get_dataset_schema(self):
        """Get full dataset schema for Code Generator context."""
        try:
            return f"**Available Data:**\n{self.bamboo._get_df_schema()}"
        except Exception as e:
            logger.warning(f"Could not get dataset schema: {e}")
            return ""

    def _get_dataset_schema_slim(self):
        """FIX 5: Lightweight schema for non-code agents (QG, Selector, Evaluator)."""
        try:
            return f"**Available Data:**\n{self.bamboo._get_df_schema_slim()}"
        except Exception as e:
            logger.warning(f"Could not get slim dataset schema: {e}")
            # Fall back to full schema if slim not available
            return self._get_dataset_schema()

    def _add_to_dormant(self, node_id):
        if node_id not in self.insight_tree:
            return False
        node = self.insight_tree[node_id]
        # Minimum depth: must have some development
        if node['depth'] < 2:
            return False
        # Minimum score: only save genuinely promising branches
        if node['quality_score'] < 6:
            node['status'] = 'abandoned'
            return False
        if len(self.dormant_branches) >= 2:
            dormant_scores = [(nid, self.insight_tree[nid]['quality_score'])
                              for nid in self.dormant_branches]
            min_dormant = min(dormant_scores, key=lambda x: x[1])
            if node['quality_score'] > min_dormant[1]:
                old_id = min_dormant[0]
                self.dormant_branches.remove(old_id)
                self.insight_tree[old_id]['status'] = 'abandoned'
            else:
                node['status'] = 'abandoned'
                return False
        self.dormant_branches.append(node_id)
        node['status'] = 'dormant'
        return True

    def _pop_best_dormant_branch(self):
        if not self.dormant_branches:
            return None
        best_id = max(self.dormant_branches,
                      key=lambda nid: self.insight_tree[nid]['quality_score'])
        self.dormant_branches.remove(best_id)
        self.insight_tree[best_id]['status'] = 'active'
        return best_id

    def _trim_question_pool(self):
        if len(self.question_pool) > 10:
            self.question_pool = self.question_pool[-10:]

    # ══════════════════════════════════════════════
    # CHECKPOINT: SAVE & RESTORE
    # ══════════════════════════════════════════════

    def _save_checkpoint(self, iteration, last_solution_chain, last_follow_up_angle):
        """Save complete exploration state to disk after each iteration.

        Writes atomically (tmp + rename) so a crash mid-write won't corrupt
        the checkpoint.  The companion _restore_checkpoint() rebuilds all
        in-memory structures from this file.
        """
        state = {
            "version": 1,
            "iterations_completed": iteration,
            "explorer": {
                "insight_tree": self.insight_tree,
                "active_branch_id": self.active_branch_id,
                "dormant_branches": self.dormant_branches,
                "question_pool": self.question_pool,
                "root_node_id": self.root_node_id,
                "research_model": self.research_model,
                "seed_question": self.seed_question,
                "current_phase": self.current_phase,
                "phase_history": self.phase_history,
                "model_impact_history": self.model_impact_history,
                "evaluator_score_history": self.evaluator_score_history,
                "biggest_gap_history": self.biggest_gap_history,
                "stagnation_count": self.stagnation_count,
                "node_counter": AutoExplorer._node_counter,
            },
            "message_manager": {
                "qa_pairs": self.bamboo.message_manager.qa_pairs,
                "all_questions": self.bamboo.message_manager.all_questions,
                "full_results_store": self.bamboo.message_manager.full_results_store,
            },
            "branch_endpoints": list(self.bamboo.branch_endpoints),
            "last_solution_chain": last_solution_chain,
            "last_follow_up_angle": last_follow_up_angle,
        }

        path = os.path.join(self.bamboo.output_dir, "state.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, path)

    def _restore_checkpoint(self, state):
        """Restore exploration state from a loaded checkpoint dict."""
        ex = state['explorer']
        self.insight_tree = ex['insight_tree']
        self.active_branch_id = ex['active_branch_id']
        self.dormant_branches = ex['dormant_branches']
        self.question_pool = ex['question_pool']
        self.root_node_id = ex['root_node_id']
        self.research_model = ex['research_model']
        self.seed_question = ex.get('seed_question', '')
        self.current_phase = ex['current_phase']
        # phase_history: JSON stores tuples as lists — convert back
        self.phase_history = [tuple(t) for t in ex['phase_history']]
        self.model_impact_history = ex['model_impact_history']
        self.evaluator_score_history = ex.get('evaluator_score_history', [])
        self.biggest_gap_history = ex.get('biggest_gap_history', [])
        self.stagnation_count = ex.get('stagnation_count', 0)
        AutoExplorer._node_counter = ex.get('node_counter', 0)

        mm = state['message_manager']
        self.bamboo.message_manager.qa_pairs = mm['qa_pairs']
        self.bamboo.message_manager.all_questions = mm['all_questions']
        self.bamboo.message_manager.full_results_store = mm['full_results_store']

        self.bamboo.branch_endpoints = state.get('branch_endpoints', [])

    def _build_exploration_trajectory(self):
        """Build trajectory for synthesis. Includes result_summary for each node."""
        if not self.insight_tree:
            return None

        explored_nodes = sorted(
            [
                {
                    'node_id': nid,
                    'question': node['question'],
                    'chain_id': node['chain_id'],
                    'quality_score': node['quality_score'],
                    'parent_id': node['parent_id'],
                    'depth': node['depth'],
                    'status': node['status'],
                    'result_summary': node.get('result_summary', 'Results not available'),
                }
                for nid, node in self.insight_tree.items()
                if node['status'] != 'abandoned'
            ],
            key=lambda n: n['chain_id'],
        )

        branches = []
        for endpoint_chain in self.bamboo.branch_endpoints:
            endpoint_node = None
            for nid, node in self.insight_tree.items():
                if node['chain_id'] == endpoint_chain:
                    endpoint_node = nid
                    break
            if endpoint_node is None:
                continue
            path = []
            current = endpoint_node
            while current is not None:
                if current in self.insight_tree:
                    path.append(current)
                    current = self.insight_tree[current]['parent_id']
                else:
                    break
            path.reverse()
            branches.append({
                'endpoint_chain_id': endpoint_chain,
                'node_ids': path,
            })

        return {
            'nodes': explored_nodes,
            'branches': branches,
            'branch_endpoints': list(self.bamboo.branch_endpoints),
            'research_model': self.research_model,
            'phase_history': self.phase_history,
        }


# ══════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════

def format_trajectory_for_synthesis(trajectory, full_results_store=None,
                                    selected_chain_id=None, max_nodes=40):
    """
    Format exploration trajectory for the synthesis prompt.

    FIX 3: Uses **score-weighted** node selection instead of recency-only.
    Ensures high-scoring early analyses survive into synthesis even at 100+
    iterations, while still including recent context.

    Prefers full_results_store for complete stdout. Falls back to tree node
    result_summary if store is unavailable.
    """
    if not trajectory or not trajectory['nodes']:
        return "No exploration data available."

    nodes = trajectory['nodes']
    if selected_chain_id:
        nodes = [n for n in nodes if n['chain_id'] <= int(selected_chain_id)]

    if not nodes:
        return "No exploration data available for this point."

    # ── FIX 3: Score-weighted node selection ──
    # Instead of nodes[-max_nodes:] (pure recency), blend top-scoring with recent.
    if len(nodes) > max_nodes:
        # Always include the last `recent_count` nodes for continuity
        recent_count = min(15, max_nodes // 3)
        recent_nodes = nodes[-recent_count:]
        recent_chain_ids = {n['chain_id'] for n in recent_nodes}

        # From remaining pool, take highest-scoring nodes
        remaining = [n for n in nodes if n['chain_id'] not in recent_chain_ids]
        remaining.sort(key=lambda n: n['quality_score'], reverse=True)
        top_scoring = remaining[:max_nodes - recent_count]

        # Combine and restore chronological order
        nodes = sorted(recent_nodes + top_scoring, key=lambda n: n['chain_id'])
    # ── End FIX 3 ──

    full_results_store = full_results_store or {}
    parts = []

    filtered_node_ids = {n['node_id'] for n in nodes}
    relevant_branches = [
        b for b in trajectory['branches']
        if any(nid in filtered_node_ids for nid in b['node_ids'])
    ]
    parts.append(f"**Exploration Overview:** {len(nodes)} analyses across {len(relevant_branches)} branch(es)\n")

    if trajectory['phase_history']:
        transitions = [f"{old}→{new} (iter {i})" for i, old, new in trajectory['phase_history']]
        parts.append(f"**Phase transitions:** {' | '.join(transitions)}\n")

    # Map nodes to branches
    node_branches = {}
    for b_idx, branch in enumerate(relevant_branches, 1):
        for nid in branch['node_ids']:
            if nid not in node_branches:
                node_branches[nid] = []
            node_branches[nid].append(b_idx)
    node_to_branch = {}
    for nid, branches in node_branches.items():
        node_to_branch[nid] = "Shared" if len(branches) > 1 else f"Branch {branches[0]}"

    parts.append("**Complete Analysis History:**\n")

    for i, node in enumerate(nodes, 1):
        branch_label = node_to_branch.get(node['node_id'], "Shared")
        status_label = f" [{node['status'].upper()}]" if node['status'] in ('dormant', 'runner_up') else ""

        # Prefer full results from store; fall back to tree node summary
        chain_key = str(node['chain_id'])
        result_text = full_results_store.get(chain_key, node.get('result_summary', 'Results not available'))

        # Safety cap — generous for synthesis (runs once, can afford tokens)
        if len(result_text) > 3000:
            result_text = result_text[:1500] + "\n[...]\n" + result_text[-1500:]

        parts.append(
            f"{i}. [{branch_label}]{status_label} Task: {node['question']}\n"
            f"   Reference: [[{node['chain_id']}]]\n"
            f"   Score: {node['quality_score']}/10\n"
            f"   Result:\n{result_text}\n"
            f"{'─' * 5}"
        )

    if trajectory['research_model']:
        parts.append(f"\n**Final Research Model:**\n{trajectory['research_model']}")

    return '\n'.join(parts)
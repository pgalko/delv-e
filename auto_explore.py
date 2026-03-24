"""
Auto-explore module for autonomous recursive exploration.
Handles the exploration loop, result evaluation, and adaptive branching.

Key mechanisms:
  - Finding Maturity: tracks significant findings through an analytical arc
    (DETECTED → QUANTIFIED → DECOMPOSED → REGIME-TESTED → COMPLETE)
  - Cross-Finding Connections: periodically tests interactions between
    established findings to discover compound effects
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
    self.engine.* interface).
    """

    PHASE_INSTRUCTIONS = {
        "MAPPING": (
            "EXPLORE",
            "We are in MAPPING phase — survey broadly. Prioritise coverage: "
            "open new angles, examine unexplored variables, screen many features at once."
        ),
        "PURSUING": (
            "EXPLOIT",
            "We are in PURSUING phase — all 5 questions must advance THE SAME finding. "
            "Consult Finding Maturity: pick the least-mature finding and generate 5 "
            "different ways to advance it to its next stage. Do not split across topics."
        ),
    }

    # ── NEW: Connection trigger configuration ──
    CONNECTION_INTERVAL = 8          # check every N iterations
    CONNECTION_MIN_FINDINGS = 4      # need this many established findings
    CONNECTION_QUESTIONS = 3         # how many connection questions to generate

    def __init__(self, engine_instance):
        self.engine = engine_instance

        # Tree-based exploration state
        self.insight_tree = {}
        self.active_branch_id = None
        self.dormant_branches = []
        self.question_pool = []
        self.root_node_id = None

        # Research model state
        self.research_model = ""
        self.seed_question = ""
        self.current_phase = "MAPPING"
        self.phase_history = []
        self.model_impact_history = []
        self.evaluator_score_history = []
        self.biggest_gap_history = []
        self.stagnation_count = 0

        # Orientation data profile
        self.data_profile = ""

        # ── NEW: track when connections were last run ──
        self.last_connection_iteration = 0

        # Model override for orientation + synthesis (set from --premium-model)
        self.premium_model = None

        self.engine.branch_endpoints = []

    @property
    def kill_signal(self):
        return self.engine.kill_signal

    @property
    def chain_id(self):
        return self.engine.chain_id

    @chain_id.setter
    def chain_id(self, value):
        self.engine.chain_id = value

    def _llm_stream_silent(self, messages, agent, model_override=None):
        """Run llm_stream in silent mode and return captured output."""
        output_manager = self.engine.output_manager
        output_manager.set_silent(True)

        try:
            response = self.engine.llm_stream(
                self.engine.prompts,
                self.engine.log_and_call_manager,
                output_manager,
                messages,
                agent=agent,
                chain_id=self.chain_id,
                reasoning_models=self.engine.reasoning_models,
                reasoning_effort="low",
                stop_event=self.engine._stop_event,
                model_override=model_override,
            )
            if not response or (isinstance(response, str) and response.strip() == ''):
                response = output_manager.get_captured_output()
            return response
        except Exception as e:
            logger.warning(f"LLM call failed for {agent}: {e}")
            return ""
        finally:
            output_manager.set_silent(False)

    def _call_agent_with_retry(self, messages, agent, model_override=None):
        """Shared agent call: try → retry → fallback to alternate model."""
        # Attempt 1
        response = self._llm_stream_silent(messages, agent, model_override=model_override)
        if response and response.strip():
            return response

        # Attempt 2: retry same model
        response = self._llm_stream_silent(messages, agent, model_override=model_override)
        if response and response.strip():
            return response

        # Attempt 3: fallback
        fallback_model = self._get_fallback_model(agent)
        if fallback_model:
            logger.info(f"{agent} empty after retry, falling back to {fallback_model}")
            response = self._llm_stream_silent(messages, agent, model_override=fallback_model)

        return response or ""

    # ══════════════════════════════════════════════
    # EVALUATOR
    # ══════════════════════════════════════════════

    def _evaluate_results_comparative(self, solutions_data, remaining_iterations=0, max_iterations=10):
        """Compare multiple solution results and select the most analytically valuable one."""
        if self.kill_signal:
            return 0, [5] * len(solutions_data), [], False, "Interrupted", None, [''] * len(solutions_data), None

        # Build context
        dormant_info = "None"
        if self.dormant_branches:
            dormant_scores = [
                self.insight_tree[nid]['quality_score']
                for nid in self.dormant_branches
                if nid in self.insight_tree
            ]
            if dormant_scores:
                dormant_info = f"{len(self.dormant_branches)} branch(es), best score: {max(dormant_scores)}"

        exploration_state = (
            f"**Current Exploration State:**\n"
            f"- Dormant branches: {dormant_info}\n"
            f"- Remaining iterations: {remaining_iterations} of {max_iterations}\n\n"
            f"{self._get_exploration_history()}"
        )

        if self.data_profile:
            exploration_state = f"**Analytical Profile:**\n{self.data_profile}\n\n{exploration_state}"

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
            solutions_parts.append(
                f"**Solution {i}:**\n"
                f"Question: {sol['question']}\n"
                f"Result:{error_note}\n{result_text}\n"
            )

        solutions_block = "\n---\n".join(solutions_parts)

        eval_prompt = self.engine.prompts.result_evaluator.format(
            seed_question=self.seed_question,
            exploration_state=exploration_state,
            solutions_block=solutions_block,
            research_model=research_model_context,
        )

        eval_messages = [{"role": "user", "content": eval_prompt}]
        response = self._call_agent_with_retry(eval_messages, 'Result Evaluator')

        # Parse response
        scores = [5] * len(solutions_data)
        if 'SCORES:' in response.upper():
            scores_line = response.upper().split('SCORES:')[1].split('\n')[0]
            for i, match in enumerate(re.findall(r'\d+', scores_line)[:len(solutions_data)]):
                scores[i] = max(1, min(10, int(match)))

        selected_index = 0
        if 'SELECTED:' in response.upper():
            match = re.search(r'\d+', response.upper().split('SELECTED:')[1].split('\n')[0])
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

        # Parse PHASE recommendation
        phase_recommendation = None
        if 'PHASE:' in response.upper():
            phase_line = response.upper().split('PHASE:')[1].split('\n')[0].strip()
            for candidate in ['MAPPING', 'PURSUING']:
                if candidate in phase_line:
                    phase_recommendation = candidate
                    break

        # Parse SUMMARIES
        summaries = [''] * len(solutions_data)
        if 'SUMMARIES:' in response:
            summary_text = response.split('SUMMARIES:')[1]
            for end in ['SELECTED:', 'KEEP_DORMANT:', 'REASON:', 'PHASE:', '\n\n']:
                idx = summary_text.find(end)
                if idx > 0:
                    summary_text = summary_text[:idx]
                    break
            for i, part in enumerate([s.strip() for s in summary_text.split('|')][:len(solutions_data)]):
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

        return (selected_index, scores, keep_dormant_indices, False,
                reason, follow_up_angle, summaries, phase_recommendation)

    # ══════════════════════════════════════════════
    # QUESTION GENERATION & SELECTION
    # ══════════════════════════════════════════════

    def _generate_branching_questions(self, use_chain_id=None, follow_up_hint=None, model_override=None):
        """Generate 5 branching questions. Phase-driven."""
        if self.kill_signal:
            return [], {}, ""

        if use_chain_id is not None:
            self.chain_id = use_chain_id
        else:
            self.chain_id = int(time.time())

        exploration_context = self._get_exploration_history()
        dataset_schema = self._get_dataset_schema_slim()

        hint_section = ""
        if follow_up_hint:
            hint_section = f"\n**Promising direction:** {follow_up_hint}\n"

        phase_mode, phase_instruction = self.PHASE_INSTRUCTIONS.get(
            self.current_phase,
            ("EXPLORE", "Generate exploratory questions."),
        )

        model_context = self.research_model if self.research_model else "(No model yet — first iteration)"

        phase_prompt = self.engine.prompts.ideas_explorer_auto.format(
            current_phase=self.current_phase,
            phase_instruction=phase_instruction,
            research_model=model_context,
        )

        all_questions_log = self.engine.message_manager.format_all_questions()

        profile_section = ""
        if self.data_profile:
            profile_section = f"\n**Analytical Profile:**\n{self.data_profile}\n"

        gen_prompt = (
            f"Based on our exploration so far:\n\n"
            f"{dataset_schema}\n{profile_section}\n{exploration_context}\n\n"
            f"{all_questions_log}\n{hint_section}\n\n{phase_prompt}"
        )

        gen_messages = [{"role": "user", "content": gen_prompt}]
        questions_response = self._call_agent_with_retry(gen_messages, 'Question Generator', model_override=model_override)
        questions, categories = self._parse_questions_with_categories(questions_response)

        return questions, categories, questions_response

    def _select_best_questions(self, questions, categories, context_hint=None, num_to_select=1):
        """Use LLM to select the most promising questions from a pool."""
        if not questions or self.kill_signal:
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
            pool_section = (
                f"\n\n**Question Pool (from previous iterations):**\n"
                f"{chr(10).join(pool_lines)}\n\n"
                f"You may select from new questions (1-{len(questions)}) "
                f"or pool ({len(questions)+1}-{len(questions)+len(pool_lines)})."
            )

        exploration_context = self._get_exploration_history()
        all_questions_log = self.engine.message_manager.format_all_questions()
        exploration_history = exploration_context + "\n\n" + all_questions_log if all_questions_log else exploration_context

        if self.data_profile:
            exploration_history = f"**Analytical Profile:**\n{self.data_profile}\n\n{exploration_history}"

        context_section = ""
        if context_hint:
            context_section = f"\n**Promising direction from recent result:** {context_hint}\n"

        research_model_context = self.research_model if self.research_model else "(No model yet)"

        selection_prompt = self.engine.prompts.question_selector.format(
            exploration_history=exploration_history,
            questions=chr(10).join(formatted_questions) + pool_section,
            context_hint=context_section,
            num_to_select=num_to_select,
            research_model=research_model_context,
            current_phase=self.current_phase,
        )

        select_messages = [{"role": "user", "content": selection_prompt}]
        selection = self._call_agent_with_retry(select_messages, 'Question Selector')

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

        # Fallback
        if not selected_questions:
            selected_indices = list(range(min(num_to_select, len(questions))))
            selected_questions = [questions[i] for i in selected_indices]
            selected_categories = [categories.get(i, 'exploit') for i in selected_indices]

        # Pad if needed
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
            is_numbered = re.match(r'^(?:[-•*#]*\s*)?([1-9])[.):]', stripped)

            if is_numbered:
                if current_q:
                    question_text = ' '.join(current_q).strip()
                    if len(question_text) > 10:
                        questions.append(question_text)
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

    # ══════════════════════════════════════════════
    # RESEARCH MODEL INTERPRETER
    # ══════════════════════════════════════════════

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

        if self.data_profile and not self.research_model:
            current_model += f"\n\n**Analytical Profile (reference only):**\n{self.data_profile}"

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

        interpret_prompt = self.engine.prompts.research_model_updater.format(
            seed_question=self.seed_question,
            current_model=current_model,
            question=winning_solution['question'],
            score=quality_score,
            result_summary=result_summary,
            parallel_results=parallel_context,
        )

        interpret_messages = [{"role": "user", "content": interpret_prompt}]
        response = self._call_agent_with_retry(interpret_messages, 'Research Interpreter')

        # Parse structured fields
        model_impact = self._parse_field(response, 'MODEL_IMPACT', default='MEDIUM',
                                          valid={'HIGH', 'MEDIUM', 'LOW'})
        contradiction = 'YES' in self._parse_field(response, 'CONTRADICTION', default='NO')
        thread_completed = 'YES' in self._parse_field(response, 'THREAD_COMPLETED', default='NO')

        # ── NEW: Parse MATURITY_ADVANCE (logged for observability) ──
        maturity_advance = self._parse_field(response, 'MATURITY_ADVANCE', default='NONE')
        if maturity_advance and maturity_advance != 'NONE':
            logger.info(f"Finding maturity advanced: {maturity_advance}")

        # Parse RESULT_DIGEST
        result_digest = ""
        if 'RESULT_DIGEST:' in response:
            digest_text = response.split('RESULT_DIGEST:')[1]
            for end_marker in ['UPDATED_MODEL:', '\n\n\n']:
                if end_marker in digest_text:
                    digest_text = digest_text.split(end_marker)[0]
                    break
            result_digest = digest_text.strip()
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

    @staticmethod
    def _parse_field(response, field_name, default='', valid=None):
        """Parse a single FIELD_NAME: value line from an LLM response."""
        key = f'{field_name}:'
        if key not in response.upper():
            return default
        line = response.upper().split(key)[1].split('\n')[0].strip()
        if valid:
            for v in valid:
                if v in line:
                    return v
            return default
        # Return original-case version
        orig_key = f'{field_name}:'
        if orig_key in response:
            return response.split(orig_key)[1].split('\n')[0].strip()
        return line

    # ══════════════════════════════════════════════
    # CROSS-FINDING CONNECTION EXPLORER (NEW)
    # ══════════════════════════════════════════════

    def _should_run_connections(self, iteration):
        """Determine whether to run cross-finding connection testing this iteration.

        Triggers when:
        1. Enough iterations since last run (CONNECTION_INTERVAL)
        2. Enough established findings (CONNECTION_MIN_FINDINGS)
        3. Not in early exploration (iteration > 10)
        """
        if iteration <= 10:
            return False
        if iteration - self.last_connection_iteration < self.CONNECTION_INTERVAL:
            return False
        # Count established findings in the research model
        n_findings = self._count_established_findings()
        if n_findings < self.CONNECTION_MIN_FINDINGS:
            return False
        return True

    def _count_established_findings(self):
        """Count bullet points in ## Established Findings section."""
        findings_text = self._extract_model_section('Established Findings')
        if not findings_text:
            return 0
        return len([line for line in findings_text.split('\n') if line.strip().startswith('-')])

    def _extract_model_section(self, section_name):
        """Extract content of a named ## section from the research model."""
        if not self.research_model or f'## {section_name}' not in self.research_model:
            return ""
        text = self.research_model.split(f'## {section_name}')[1]
        # Take until next ## header or end
        next_section = text.find('\n## ')
        if next_section > 0:
            text = text[:next_section]
        return text.strip()

    def _generate_connection_questions(self):
        """Generate questions testing interactions between established findings.

        Returns list of connection questions to inject into the question pool.
        """
        if self.kill_signal:
            return []

        established = self._extract_model_section('Established Findings')
        if not established:
            return []

        tested = self._extract_model_section('Cross-Finding Connections')
        if not tested:
            tested = "(None tested yet)"

        profile = self.data_profile if self.data_profile else "(No profile)"

        prompt = self.engine.prompts.connection_explorer.format(
            established_findings=established,
            tested_connections=tested,
            data_profile=profile,
            num_questions=self.CONNECTION_QUESTIONS,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_agent_with_retry(
            messages, 'Connection Explorer', model_override=self.premium_model)

        # Parse numbered questions (reuse existing parser)
        questions, _ = self._parse_questions_with_categories(response)

        logger.info(f"Connection explorer generated {len(questions)} questions")
        return questions

    def _inject_connection_questions(self, iteration):
        """Run connection explorer and inject questions into pool."""
        logger.info(f"Running cross-finding connection test (iteration {iteration})")
        self.last_connection_iteration = iteration

        connection_questions = self._generate_connection_questions()
        for q in connection_questions:
            self.question_pool.append({
                'question': q,
                'source_branch_id': self.active_branch_id,
                'category': 'connection',
                'iteration_added': iteration,
            })
        if connection_questions:
            print(style.branch_event(f"Connection explorer: {len(connection_questions)} interaction questions added to pool"))

    # ══════════════════════════════════════════════
    # PHASE DETERMINATION
    # ══════════════════════════════════════════════

    def _determine_phase(self, model_impact, contradiction, thread_completed,
                         phase_recommendation=None):
        """Model-driven phase transition.

        The evaluator recommends a phase based on the full context.
        Thread completion is the only structural override.
        """
        if thread_completed:
            logger.info("Phase → MAPPING (thread completed)")
            return "MAPPING"

        if phase_recommendation in ("MAPPING", "PURSUING"):
            if phase_recommendation != self.current_phase:
                logger.info("Phase → %s (evaluator recommendation)", phase_recommendation)
            return phase_recommendation

        logger.info(f"Phase → {self.current_phase} (maintained)")
        return self.current_phase

    # ══════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════

    def run(self, seed_question, initial_image=None, max_iterations=5, num_parallel_solutions=2,
            interactive=False, resumed_state=None, orientation=True):
        """Run the autonomous exploration loop."""

        num_parallel_solutions = max(2, min(5, num_parallel_solutions))

        # Store original settings
        original_user_feedback = self.engine.user_feedback
        original_analyst_system_content = self.engine.message_manager.select_analyst_messages[0]["content"]
        original_max_errors = self.engine.MAX_ERROR_CORRECTIONS

        # Switch to auto-explore mode
        self.engine.user_feedback = False
        self.engine.message_manager.select_analyst_messages[0]["content"] = self.engine.prompts.analyst_selector_system_auto
        self.engine.MAX_ERROR_CORRECTIONS = 3

        try:
            current_image = initial_image

            if resumed_state:
                self._restore_checkpoint(resumed_state)
                start_iteration = resumed_state['iterations_completed']
                max_iterations = start_iteration + max_iterations
                last_solution_chain = resumed_state.get('last_solution_chain')
                last_follow_up_angle = resumed_state.get('last_follow_up_angle')

                current_questions = [seed_question]
                current_categories = ['exploit']

                logger.info(f"Resuming from iteration {start_iteration}, "
                            f"running {max_iterations - start_iteration} more")
            else:
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
                self.seed_question = seed_question
                self.current_phase = "MAPPING"
                self.phase_history = []
                self.model_impact_history = []
                self.evaluator_score_history = []
                self.biggest_gap_history = []
                self.stagnation_count = 0
                self.data_profile = ""
                self.last_connection_iteration = 0

                start_iteration = 0

            iteration = start_iteration
            self._current_iteration = start_iteration

            # --- Header ---
            df_shape = f"{self.engine.df.shape[0]:,} rows × {self.engine.df.shape[1]} cols"
            agent_model = self.engine.models.agent_model
            code_model = self.engine.models.code_model
            if interactive and not resumed_state:
                print(style.config_lines(df_shape, max_iterations, num_parallel_solutions,
                                         self.engine.output_dir, agent_model, code_model,
                                         premium_model=self.premium_model))
            else:
                extra = ""
                if resumed_state:
                    extra = f" (resuming from iteration {start_iteration})"
                print(style.splash_header(df_shape, max_iterations, num_parallel_solutions,
                                          self.engine.output_dir, agent_model, code_model,
                                          premium_model=self.premium_model))
                if extra:
                    print(f"    {style.DIM}{extra}{style.RESET}")
            print()

            logger.info(f"Starting auto-explore: {seed_question[:50]}... max_iterations={max_iterations}")

            # ── ORIENTATION (fresh runs only) ──
            if orientation and not resumed_state and not self.kill_signal:
                logger.info("Running orientation phase")
                try:
                    self.data_profile = self.engine.run_orientation(
                        seed_question, model_override=self.premium_model)
                except Exception as e:
                    logger.warning(f"Orientation failed: {e}")
                    self.data_profile = ""
                if self.data_profile:
                    self.engine.data_profile = self.data_profile
                    logger.info(f"Orientation complete: {len(self.data_profile)} chars")

            # ══════════════════════════════════════════════
            # MAIN ITERATION LOOP
            # ══════════════════════════════════════════════

            while iteration < max_iterations:
                self._current_iteration = iteration

                if self.kill_signal:
                    break

                self.engine._iteration = iteration + 1
                self.engine._max_iterations = max_iterations
                self.engine._phase = self.current_phase

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

                print(style.iteration_bar(iteration + 1, max_iterations, self.current_phase))

                # ── PROCESS QUESTIONS ──
                solutions_data = []

                for q_idx, question in enumerate(questions_to_process):
                    if self.kill_signal:
                        break

                    if len(questions_to_process) > 1 and last_solution_chain is not None:
                        self.engine.message_manager.restore_interaction(
                            self.engine.thread_id,
                            last_solution_chain,
                        )

                    self.chain_id = int(time.time()) + q_idx
                    solution_chain_id = self.chain_id

                    cat = categories_to_process[q_idx] if q_idx < len(categories_to_process) else ""
                    print()
                    print(style.question_display(q_idx + 1, len(questions_to_process), cat, question))

                    error_occurred = False
                    try:
                        self.engine._process_question(question, current_image if q_idx == 0 else None, None, None)
                    except Exception as e:
                        error_occurred = True
                        logger.warning(f"Execution error: {e}")
                        self.engine.output_manager.display_error(f"Execution error: {e}", chain_id=self.chain_id)

                    solutions_data.append({
                        'question': question,
                        'results': self.engine.message_manager.code_exec_results,
                        'code': self.engine.message_manager.last_code,
                        'text_answer': self.engine.message_manager.last_plan,
                        'chain_id': solution_chain_id,
                        'error_occurred': error_occurred,
                        'category': categories_to_process[q_idx] if q_idx < len(categories_to_process) else 'exploit',
                    })

                    self.engine.message_manager.reset_non_cumul_messages()
                    time.sleep(1)

                self.chain_id = int(time.time())
                supporting_chain_id = self.chain_id

                # ── EVALUATE ──
                if len(solutions_data) == 1:
                    selected_index = 0
                    scores = [7]
                    keep_dormant_indices = []
                    reason = "Seed question — establishing baseline"
                    follow_up_angle = None
                    summaries = ['']
                    phase_recommendation = None
                    _is_seed = True
                else:
                    with style.spinner("Evaluating results"):
                        (selected_index, scores, keep_dormant_indices, _,
                         reason, follow_up_angle, summaries, phase_recommendation) = \
                            self._evaluate_results_comparative(
                                solutions_data,
                                remaining_iterations=max_iterations - iteration,
                                max_iterations=max_iterations,
                            )
                    _is_seed = False

                self.evaluator_score_history.append(scores[selected_index])
                if len(self.evaluator_score_history) > 15:
                    self.evaluator_score_history = self.evaluator_score_history[-15:]

                # ── UPDATE TREE ──
                winning_solution = solutions_data[selected_index]
                result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"
                common_parent_id = self.active_branch_id

                new_node_id = self._add_node_to_tree(
                    question=winning_solution['question'],
                    result_summary=result_summary,
                    quality_score=scores[selected_index],
                    chain_id=winning_solution['chain_id'],
                    parent_id=common_parent_id,
                )
                if selected_index < len(summaries) and summaries[selected_index]:
                    self.insight_tree[new_node_id]['finding_summary'] = summaries[selected_index]

                self.active_branch_id = new_node_id
                last_solution_chain = winning_solution['chain_id']
                last_follow_up_angle = follow_up_angle

                # ── INTERPRET & UPDATE MODEL ──
                with style.spinner("Updating research model"):
                    updated_model, model_impact, contradiction, thread_completed, result_digest = \
                        self._interpret_and_update_model(
                            winning_solution,
                            scores[selected_index],
                            solutions_data=solutions_data,
                            selected_index=selected_index,
                            scores=scores,
                        )

                if new_node_id in self.insight_tree and result_digest:
                    self.insight_tree[new_node_id]['result_digest'] = result_digest

                self.research_model = updated_model
                if len(self.research_model) > 6000:
                    self.research_model += (
                        "\n\n**NOTE: Model getting long — consolidate on next update.**"
                    )
                self.model_impact_history.append(model_impact)
                if len(self.model_impact_history) > 10:
                    self.model_impact_history = self.model_impact_history[-10:]

                self._update_gap_stability(updated_model)

                # ── DETERMINE PHASE ──
                old_phase = self.current_phase
                new_phase = self._determine_phase(
                    model_impact, contradiction, thread_completed,
                    phase_recommendation=phase_recommendation,
                )
                if iteration == 0:
                    new_phase = "MAPPING"

                if new_phase != self.current_phase:
                    self.phase_history.append((iteration, self.current_phase, new_phase))
                self.current_phase = new_phase

                # ── ADD NON-WINNING SOLUTIONS TO TREE ──
                for idx in range(len(solutions_data)):
                    if idx == selected_index:
                        continue
                    sol = solutions_data[idx]
                    node_id = self._add_node_to_tree(
                        question=sol['question'],
                        result_summary=str(sol['results']) if sol['results'] else "No results",
                        quality_score=scores[idx] if idx < len(scores) else 0,
                        chain_id=sol['chain_id'],
                        parent_id=common_parent_id,
                    )
                    if idx < len(summaries) and summaries[idx]:
                        self.insight_tree[node_id]['finding_summary'] = summaries[idx]

                    if idx in keep_dormant_indices:
                        summary = self.insight_tree[node_id].get('finding_summary', '')
                        label = sol['question'][:120]
                        if summary:
                            label += f" — Found: {summary[:150]}"
                        self.insight_tree[node_id]['hypothesis_label'] = label
                        self._add_to_dormant(node_id)
                    else:
                        self.insight_tree[node_id]['status'] = 'runner_up'

                # ── WRITE ITERATION SUMMARY ──
                self.engine.write_iteration_summary(
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

                iteration += 1
                self._save_checkpoint(iteration, last_solution_chain, last_follow_up_angle)

                if iteration >= max_iterations:
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

                # ── BRANCH DECISION ──
                if new_phase == "MAPPING" and thread_completed:
                    self.engine.branch_endpoints.append(last_solution_chain)
                    print(style.branch_event("Thread complete — new territory"))

                    if self.dormant_branches:
                        new_branch_id = self._pop_best_dormant_branch()
                        new_branch = self.insight_tree[new_branch_id]
                        self.engine.message_manager.restore_interaction(
                            self.engine.thread_id,
                            new_branch['chain_id'],
                        )
                        self.active_branch_id = new_branch_id
                        last_solution_chain = new_branch['chain_id']
                        last_follow_up_angle = new_branch.get('hypothesis_label') or new_branch['question'][:150]
                        print(style.branch_event(f"Switched to dormant branch (score: {style.score(new_branch['quality_score'])})"))

                # ── NEW: CROSS-FINDING CONNECTION CHECK ──
                if self._should_run_connections(iteration):
                    with style.spinner("Testing cross-finding connections"):
                        self._inject_connection_questions(iteration)

                # ── GENERATE NEW QUESTIONS ──
                with style.spinner("Generating questions"):
                    new_questions, new_categories, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                        follow_up_hint=last_follow_up_angle,
                    )

                # Retry once if parsing failed
                if not new_questions:
                    logger.info(f"Question generation empty. Raw: {raw_response[:200] if raw_response else '(empty)'}")
                    print(style.error_msg("Question generation failed, retrying..."))
                    new_questions, new_categories, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                        follow_up_hint=last_follow_up_angle,
                    )

                if not new_questions:
                    print(style.error_msg("Could not generate questions. Exploration complete."))
                    break

                # ── SELECT QUESTIONS ──
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

            # ── POST-LOOP ──
            if last_solution_chain:
                self.engine.branch_endpoints.append(last_solution_chain)

            self.engine.exploration_trajectory = self._build_exploration_trajectory()

            with style.spinner("Generating synthesis report"):
                synthesis_text = self._generate_synthesis(seed_question)

            if not synthesis_text:
                print(f"  {style.YELLOW}✗ Synthesis failed{style.RESET}")
                synthesis_text = ""

            print(style.exploration_tree(self.insight_tree, self.root_node_id,
                                         phase_history=self.phase_history,
                                         total_iterations=iteration))

            dormant_count = len(self.dormant_branches)
            avg_score = sum(n['quality_score'] for n in self.insight_tree.values()) / len(self.insight_tree) if self.insight_tree else 0

            print(style.final_box(
                iterations=iteration,
                analyses=len(self.insight_tree),
                avg=avg_score,
                dormant=dormant_count,
                phase_hist=self.phase_history,
                cost_str=self.engine.cost_tracker.report(),
                output_dir=self.engine.output_dir,
            ))

            self.engine.write_final_outputs(
                research_model=self.research_model,
                phase_history=self.phase_history,
                synthesis_text=synthesis_text,
            )
            print()

        finally:
            self.engine.user_feedback = original_user_feedback
            self.engine.message_manager.select_analyst_messages[0]["content"] = original_analyst_system_content
            self.engine.MAX_ERROR_CORRECTIONS = original_max_errors

    def _generate_synthesis(self, seed_question):
        """Generate final synthesis report."""
        if not self.insight_tree:
            return None

        full_store = self.engine.message_manager.full_results_store

        synthesis_context = format_synthesis_input(
            insight_tree=self.insight_tree,
            full_results_store=full_store,
            research_model=self.research_model,
            seed_question=seed_question,
            data_profile=self.data_profile or '',
        )

        import datetime
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = self.engine.prompts.exploration_synthesis.format(
            today,
            synthesis_context,
            f"Synthesize all findings from the exploration seeded by: {seed_question}",
        )

        messages = [{"role": "user", "content": prompt}]

        # Synthesis is the payoff of the entire run — retry indefinitely
        # with exponential backoff (10→20→40→80s cap) until a response is received.
        attempt = 0
        backoff = 10
        while True:
            attempt += 1
            response = self._call_agent_with_retry(
                messages, 'Synthesis Generator', model_override=self.premium_model)
            if response and response.strip():
                if attempt > 1:
                    logger.info(f"Synthesis succeeded on attempt {attempt}")
                return response

            logger.warning(f"Synthesis attempt {attempt} failed, retrying in {backoff}s...")
            print(f"  {style.YELLOW}⟳ Synthesis attempt {attempt} failed — retrying in {backoff}s...{style.RESET}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 80)

    # ══════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════

    _node_counter = 0

    def _generate_node_id(self):
        AutoExplorer._node_counter += 1
        return f"node_{int(time.time() * 1000)}_{AutoExplorer._node_counter}"

    def _extract_biggest_gap(self, model_text):
        """Extract the Biggest Gap section from the research model."""
        return self._extract_model_section('Biggest Gap') if model_text else ""

    def _update_gap_stability(self, updated_model):
        """Track whether the Biggest Gap has changed."""
        current_gap = self._extract_biggest_gap(updated_model)
        if not current_gap:
            return False

        self.biggest_gap_history.append(current_gap)
        if len(self.biggest_gap_history) > 5:
            self.biggest_gap_history = self.biggest_gap_history[-5:]

        if len(self.biggest_gap_history) < 2:
            return False

        prev_words = set(self.biggest_gap_history[-2].lower().split())
        curr_words = set(current_gap.lower().split())
        if not prev_words or not curr_words:
            return False

        overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
        return overlap > 0.70

    def _add_node_to_tree(self, question, result_summary, quality_score, chain_id, parent_id=None):
        node_id = self._generate_node_id()
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
            'finding_summary': '',
            'result_digest': '',
            'hypothesis_label': '',
        }
        if parent_id and parent_id in self.insight_tree:
            self.insight_tree[parent_id]['children_ids'].append(node_id)
        if self.root_node_id is None:
            self.root_node_id = node_id
        return node_id

    def _get_exploration_history(self, max_entries=40, full_detail_count=15):
        """Build exploration history with tiered compaction."""
        if not self.insight_tree:
            return ""
        all_nodes = sorted(self.insight_tree.values(), key=lambda n: n['chain_id'])
        recent = all_nodes[-max_entries:]

        if len(recent) > full_detail_count:
            compact_nodes = recent[:-full_detail_count]
            full_nodes = recent[-full_detail_count:]
        else:
            compact_nodes = []
            full_nodes = recent

        history_parts = ["**Exploration History (all branches):**"]

        if compact_nodes:
            history_parts.append(f"\n*Earlier analyses ({len(compact_nodes)} entries):*")
            for node in compact_nodes:
                score = node['quality_score']
                question = node['question']
                status_marker = " [DORMANT]" if node['status'] == 'dormant' else ""
                summary = node.get('finding_summary', '')
                if not summary:
                    finding = node['result_summary'] if node['result_summary'] else "No results"
                    summary = finding.split('\n')[0][:200]
                history_parts.append(
                    f"- [{score}/10]{status_marker} Q: {question}\n  → {summary}"
                )

        if full_nodes:
            if compact_nodes:
                history_parts.append(f"\n*Recent analyses ({len(full_nodes)} entries):*")
            for node in full_nodes:
                score = node['quality_score']
                question = node['question']
                status_marker = " [DORMANT]" if node['status'] == 'dormant' else ""

                digest = node.get('result_digest', '')
                if digest and node['status'] == 'active':
                    history_parts.append(
                        f"- [{score}/10]{status_marker} Q: {question}\n  Finding: {digest}"
                    )
                else:
                    summary = node.get('finding_summary', '')
                    if not summary:
                        finding = node['result_summary'] if node['result_summary'] else "No results"
                        summary = finding.split('\n')[0][:200]
                    history_parts.append(
                        f"- [{score}/10]{status_marker} Q: {question}\n  → {summary}"
                    )

        if self.phase_history:
            transitions = [f"{old}→{new} (iter {i})" for i, old, new in self.phase_history]
            history_parts.append(f"\n**Phase transitions:** {' | '.join(transitions)}")
        return '\n'.join(history_parts)

    def _get_dataset_schema(self):
        try:
            return f"**Available Data:**\n{self.engine._get_df_schema()}"
        except Exception as e:
            logger.warning(f"Could not get dataset schema: {e}")
            return ""

    def _get_dataset_schema_slim(self):
        try:
            return f"**Available Data:**\n{self.engine._get_df_schema_slim()}"
        except Exception as e:
            logger.warning(f"Could not get slim schema: {e}")
            return self._get_dataset_schema()

    def _add_to_dormant(self, node_id):
        if node_id not in self.insight_tree:
            return False
        node = self.insight_tree[node_id]
        if node['depth'] < 2:
            return False
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

    def _get_fallback_model(self, agent):
        return self.engine._get_fallback_model(agent)

    def _trim_question_pool(self):
        if len(self.question_pool) > 10:
            self.question_pool = self.question_pool[-10:]

    # ══════════════════════════════════════════════
    # CHECKPOINT: SAVE & RESTORE
    # ══════════════════════════════════════════════

    def _save_checkpoint(self, iteration, last_solution_chain, last_follow_up_angle):
        """Save complete exploration state to disk."""
        state = {
            "version": 2,  # bumped for new fields
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
                "data_profile": self.data_profile,
                "last_connection_iteration": self.last_connection_iteration,  # NEW
            },
            "message_manager": {
                "qa_pairs": self.engine.message_manager.qa_pairs,
                "all_questions": self.engine.message_manager.all_questions,
                "full_results_store": self.engine.message_manager.full_results_store,
            },
            "branch_endpoints": list(self.engine.branch_endpoints),
            "last_solution_chain": last_solution_chain,
            "last_follow_up_angle": last_follow_up_angle,
        }

        path = os.path.join(self.engine.output_dir, "state.json")
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
        self.phase_history = [tuple(t) for t in ex['phase_history']]
        self.model_impact_history = ex['model_impact_history']
        self.evaluator_score_history = ex.get('evaluator_score_history', [])
        self.biggest_gap_history = ex.get('biggest_gap_history', [])
        self.stagnation_count = ex.get('stagnation_count', 0)
        AutoExplorer._node_counter = ex.get('node_counter', 0)
        self.data_profile = ex.get('data_profile', '')
        self.last_connection_iteration = ex.get('last_connection_iteration', 0)  # NEW
        self.engine.data_profile = self.data_profile

        mm = state['message_manager']
        self.engine.message_manager.qa_pairs = mm['qa_pairs']
        self.engine.message_manager.all_questions = mm['all_questions']
        self.engine.message_manager.full_results_store = mm['full_results_store']

        self.engine.branch_endpoints = state.get('branch_endpoints', [])

    def _build_exploration_trajectory(self):
        """Build trajectory for synthesis."""
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
        for endpoint_chain in self.engine.branch_endpoints:
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
            'branch_endpoints': list(self.engine.branch_endpoints),
            'research_model': self.research_model,
            'phase_history': self.phase_history,
        }


# ══════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════

def format_trajectory_for_synthesis(trajectory, full_results_store=None,
                                    selected_chain_id=None, max_nodes=40):
    """Format exploration trajectory for synthesis. Score-weighted node selection."""
    if not trajectory or not trajectory['nodes']:
        return "No exploration data available."

    nodes = trajectory['nodes']
    if selected_chain_id:
        nodes = [n for n in nodes if n['chain_id'] <= int(selected_chain_id)]

    if not nodes:
        return "No exploration data available for this point."

    # Score-weighted selection: blend top-scoring with recent
    if len(nodes) > max_nodes:
        recent_count = min(15, max_nodes // 3)
        recent_nodes = nodes[-recent_count:]
        recent_chain_ids = {n['chain_id'] for n in recent_nodes}

        remaining = [n for n in nodes if n['chain_id'] not in recent_chain_ids]
        remaining.sort(key=lambda n: n['quality_score'], reverse=True)
        top_scoring = remaining[:max_nodes - recent_count]

        nodes = sorted(recent_nodes + top_scoring, key=lambda n: n['chain_id'])

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

        chain_key = str(node['chain_id'])
        result_text = full_results_store.get(chain_key, node.get('result_summary', 'Results not available'))

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


def format_synthesis_input(insight_tree, full_results_store, research_model,
                           seed_question, data_profile=''):
    """Build structured synthesis input from exploration state."""
    full_results_store = full_results_store or {}
    active = sorted(
        [n for n in insight_tree.values() if n.get('status') == 'active'],
        key=lambda n: n['chain_id'],
    )

    if not active:
        return "No exploration data available."

    parts = []

    # Section A: Framing
    parts.append("═══════════════════════════════════════")
    parts.append("SECTION A: EXPLORATION CONTEXT")
    parts.append("═══════════════════════════════════════\n")
    parts.append(f"**Original question:** {seed_question}\n")
    if data_profile:
        parts.append(f"**Dataset profile:**\n{data_profile}\n")
    parts.append(f"**Exploration scope:** {len(active)} analyses completed\n")

    # Section B: Findings Index
    parts.append("═══════════════════════════════════════")
    parts.append("SECTION B: COMPLETE FINDINGS INDEX")
    parts.append("═══════════════════════════════════════\n")

    for n in active:
        fs = n.get('finding_summary', '') or '(no summary)'
        parts.append(f"[{n['quality_score']}] [[{n['chain_id']}]] {fs}")

    # Section C: Full Evidence
    parts.append("\n═══════════════════════════════════════")
    parts.append("SECTION C: FULL EVIDENCE")
    parts.append("═══════════════════════════════════════\n")

    for n in active:
        chain_key = str(n['chain_id'])
        result_text = full_results_store.get(
            chain_key, n.get('result_summary', 'Results not available')
        )
        if len(result_text) > 3000:
            result_text = result_text[:1500] + "\n[...truncated...]\n" + result_text[-1500:]

        parts.append(
            f"[[{n['chain_id']}]] Score: {n['quality_score']}/10\n"
            f"Question: {n['question']}\n"
            f"Results:\n{result_text}\n"
            f"{'─' * 5}"
        )

    # Section D: Research Model
    parts.append("\n═══════════════════════════════════════")
    parts.append("SECTION D: FINAL RESEARCH MODEL")
    parts.append("═══════════════════════════════════════\n")
    parts.append(research_model or "(No research model available)")

    return '\n'.join(parts)
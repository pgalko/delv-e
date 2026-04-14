"""
Auto-explore module for autonomous recursive exploration.
Handles the exploration loop, result evaluation, and adaptive branching.

Key mechanisms:
  - Finding Maturity: tracks significant findings through an analytical arc
    (DETECTED → QUANTIFIED → DECOMPOSED → REGIME-TESTED → COMPLETE)
  - Strategic Review: premium model runs every iteration to enforce commitment,
    detect missed opportunities, surface untested connections, and maintain
    a Strategic Trajectory narrative in the research model
  - Reframing Probe: when strategic review requests it (on any commitment type),
    premium model reads full analytical output (not digests) and looks for
    patterns the headline tests missed: distributional shifts on null results,
    threshold effects on positive findings, or derived metrics that would
    sharpen the operational value of existing discoveries
  - Perspective Rotation: when an original arc completes, premium model generates
    2-3 ranked alternative analytical lenses on the same phenomenon. The top-ranked
    perspective is automatically pursued for 1-2 iterations before the next planned
    arc. Perspective arcs do not trigger further rotations.
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

    def __init__(self, engine_instance):
        self.engine = engine_instance

        # Exploration state
        self.insight_tree = {}
        self.question_pool = []

        # Research model state
        self.research_model = ""
        self.seed_question = ""
        self.commitment_history = []  # [(iteration, action)] for dashboard
        self.model_impact_history = []
        self.evaluator_score_history = []
        self.biggest_gap_history = []
        self.stagnation_count = 0

        # Orientation data profile
        self.data_profile = ""

        # ── Strategic review state ──
        self.last_review_iteration = 0
        self.strategic_next_direction = ""   # set by premium model on PIVOT/ABANDON
        self.current_arc_direction = ""      # the arc currently being pursued
        self._initial_trajectory = ""        # set by seed decomposition, consumed on iteration 0
        self.last_probe_iteration = 0        # tracks when last probe ran (observability)
        self.probe_history = []              # [(iteration, brief_result)] for dashboard
        self.completed_original_arcs = set() # arc directions that have been rotated (no recursion)
        self.rotation_history = []           # [(iteration, parent_arc, [{name, question}])] for dashboard
        self.arc_history = []                # [(start_iter, label)] for dashboard heatmap

        # Model override for orientation, strategic review, and synthesis
        self.premium_model = None

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
            return 0, [5] * len(solutions_data), [], False, "Interrupted", None, [''] * len(solutions_data)

        # Build context
        exploration_state = (
            f"**Current Exploration State:**\n"
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

        # Parse SUMMARIES
        summaries = [''] * len(solutions_data)
        if 'SUMMARIES:' in response:
            summary_text = response.split('SUMMARIES:')[1]
            for end in ['SELECTED:', 'REASON:', '\n\n']:
                idx = summary_text.find(end)
                if idx > 0:
                    summary_text = summary_text[:idx]
                    break
            for i, part in enumerate([s.strip() for s in summary_text.split('|')][:len(solutions_data)]):
                if part and len(part) > 5:
                    summaries[i] = part

        reason = "No reason provided"
        if 'REASON:' in response:
            reason = response.split('REASON:')[1].split('\n')[0].strip()

        follow_up_angle = None
        if 'FOLLOW_UP_ANGLE:' in response:
            angle_text = response.split('FOLLOW_UP_ANGLE:')[1].split('\n')[0].strip()
            if angle_text and angle_text.lower() not in ['', 'none', 'n/a']:
                follow_up_angle = angle_text

        return (selected_index, scores, [], False,
                reason, follow_up_angle, summaries)

    # ══════════════════════════════════════════════
    # QUESTION GENERATION & SELECTION
    # ══════════════════════════════════════════════

    def _generate_branching_questions(self, use_chain_id=None, follow_up_hint=None, model_override=None):
        """Generate 5 branching questions. Commitment-driven."""
        if self.kill_signal:
            return [], ""

        if use_chain_id is not None:
            self.chain_id = use_chain_id
        else:
            self.chain_id = int(time.time())

        exploration_context = self._get_exploration_history()
        dataset_schema = self._get_dataset_schema_slim()

        hint_section = ""
        if follow_up_hint:
            hint_section = f"\n**Promising direction:** {follow_up_hint}\n"

        model_context = self.research_model if self.research_model else "(No model yet — first iteration)"

        # Build commitment instruction from strategic state
        if self.strategic_next_direction:
            commitment_instruction = (
                f"**STRATEGIC DIRECTION (from strategic review — this is a binding constraint):**\n"
                f"{self.strategic_next_direction}\n"
                f"All 5 questions must align with this direction. Do not generate questions "
                f"on the previous arc. Explore this new direction broadly — cover multiple "
                f"angles and variables."
            )
        else:
            commitment_instruction = (
                "The strategic review is holding commitment on the current arc. "
                "Focus all questions on advancing the current investigation."
            )

        phase_prompt = self.engine.prompts.ideas_explorer_auto.format(
            commitment_instruction=commitment_instruction,
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
        questions = self._parse_questions(questions_response)

        return questions, questions_response

    def _select_best_questions(self, questions, context_hint=None, num_to_select=1):
        """Use LLM to select the most promising questions from a pool."""
        if not questions or self.kill_signal:
            return []

        num_to_select = min(num_to_select, len(questions))

        formatted_questions = [f"{i+1}. {q}" for i, q in enumerate(questions)]

        pool_section = ""
        if self.question_pool:
            pool_lines = []
            for j, pq in enumerate(self.question_pool[-5:]):
                pool_idx = len(questions) + j + 1
                pool_lines.append(f"{pool_idx}. [POOL] {pq['question']}")
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

        # Build commitment context for selector
        if self.strategic_next_direction:
            commitment_ctx = f"**Current commitment:** Pivoting to new territory — prefer breadth and diversity."
        elif self.current_arc_direction:
            arc_short = self.current_arc_direction[:80]
            commitment_ctx = f"**Current commitment:** Holding on current arc — {arc_short}"
        else:
            commitment_ctx = "**Current commitment:** Early exploration — prefer breadth."

        selection_prompt = self.engine.prompts.question_selector.format(
            exploration_history=exploration_history,
            questions=chr(10).join(formatted_questions) + pool_section,
            context_hint=context_section,
            num_to_select=num_to_select,
            research_model=research_model_context,
            commitment_context=commitment_ctx,
        )

        select_messages = [{"role": "user", "content": selection_prompt}]
        selection = self._call_agent_with_retry(select_messages, 'Question Selector')

        # Parse selection
        selected_questions = []
        selected_indices = []

        try:
            matches = re.findall(r'\d+', selection.strip())
            for match in matches:
                idx = int(match) - 1
                if idx < len(questions):
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        selected_questions.append(questions[idx])
                else:
                    pool_idx = idx - len(questions)
                    pool_subset = self.question_pool[-5:]
                    if 0 <= pool_idx < len(pool_subset):
                        pq = pool_subset[pool_idx]
                        selected_questions.append(pq['question'])
                        self.question_pool.remove(pq)
                if len(selected_questions) >= num_to_select:
                    break
        except (AttributeError, ValueError):
            pass

        # Fallback
        if not selected_questions:
            selected_indices = list(range(min(num_to_select, len(questions))))
            selected_questions = [questions[i] for i in selected_indices]

        # Pad if needed
        if len(selected_questions) < num_to_select:
            for i in range(len(questions)):
                if i not in selected_indices and len(selected_questions) < num_to_select:
                    selected_indices.append(i)
                    selected_questions.append(questions[i])

        # Store unselected in pool
        for i, q in enumerate(questions):
            if i not in selected_indices:
                self.question_pool.append({
                    'question': q,
                    'iteration_added': getattr(self, '_current_iteration', 0),
                })
        self._trim_question_pool()

        return selected_questions

    def _parse_questions(self, questions_response):
        """Parse numbered questions from LLM response."""
        questions = []

        if not questions_response or not questions_response.strip():
            return []

        # Pattern to strip LLM-generated tags like [EXPLORE], [EXPLOIT], [REGIME-TEST], etc.
        _tag_pattern = re.compile(
            r'\[(?:EXPLORE|EXPLOIT|MAPPING|PURSUING|CONNECTION|'
            r'DETECTED|QUANTIFIED|DECOMPOSED?|REGIME[- ]TEST(?:ED)?|COMPLETE)\]\s*',
            re.IGNORECASE
        )

        lines = questions_response.split('\n')
        current_q = []

        for line in lines:
            stripped = line.strip()
            is_numbered = re.match(r'^(?:[-•*#]*\s*)?([1-9])[.):]', stripped)

            if is_numbered:
                if current_q:
                    question_text = _tag_pattern.sub('', ' '.join(current_q)).strip()
                    if len(question_text) > 10:
                        questions.append(question_text)
                cleaned = re.sub(r'^(?:[-•*#]*\s*)?[1-9][.):][\s]*', '', stripped)
                current_q = [cleaned] if cleaned else []
            elif current_q and stripped and stripped != '---':
                current_q.append(stripped)

        if current_q:
            question_text = _tag_pattern.sub('', ' '.join(current_q)).strip()
            if len(question_text) > 10:
                questions.append(question_text)

        return questions

    # ══════════════════════════════════════════════
    # RESEARCH MODEL INTERPRETER
    # ══════════════════════════════════════════════

    def _interpret_and_update_model(self, winning_solution, quality_score):
        """Interpret the latest result and update the research model.

        Returns:
            (updated_model, model_impact, contradiction, arc_exhausted, result_digest, method_used)
        """
        if self.kill_signal:
            return self.research_model, "LOW", False, False, "", ""

        result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"
        current_model = self.research_model if self.research_model else "(No model yet — this is the first result. Initialize the model from scratch.)"

        # Column list for Exploration Health cross-reference (always available)
        column_list = self.engine._get_column_list() if hasattr(self.engine, '_get_column_list') else "(columns not available)"

        interpret_prompt = self.engine.prompts.research_model_updater.format(
            seed_question=self.seed_question,
            current_model=current_model,
            question=winning_solution['question'],
            score=quality_score,
            result_summary=result_summary,
            column_list=column_list,
        )

        interpret_messages = [{"role": "user", "content": interpret_prompt}]
        response = self._call_agent_with_retry(interpret_messages, 'Research Interpreter')

        # Parse structured fields
        model_impact = self._parse_field(response, 'MODEL_IMPACT', default='MEDIUM',
                                          valid={'HIGH', 'MEDIUM', 'LOW'})
        contradiction = 'YES' in self._parse_field(response, 'CONTRADICTION', default='NO')
        arc_exhausted = 'YES' in self._parse_field(response, 'ARC_EXHAUSTED', default='NO')

        # ── NEW: Parse MATURITY_ADVANCE (logged for observability) ──
        maturity_advance = self._parse_field(response, 'MATURITY_ADVANCE', default='NONE')
        if maturity_advance and maturity_advance != 'NONE':
            logger.info(f"Finding maturity advanced: {maturity_advance}")

        # Parse RESULT_DIGEST
        result_digest = ""
        if 'RESULT_DIGEST:' in response:
            digest_text = response.split('RESULT_DIGEST:')[1]
            for end_marker in ['METHOD_USED:', 'UPDATED_MODEL:', '\n\n\n']:
                if end_marker in digest_text:
                    digest_text = digest_text.split(end_marker)[0]
                    break
            result_digest = digest_text.strip()

        # Parse METHOD_USED
        method_used = ""
        if 'METHOD_USED:' in response:
            method_text = response.split('METHOD_USED:')[1].split('\n')[0].strip()
            if method_text and method_text.upper() not in ('', 'NONE', 'N/A'):
                method_used = method_text

        # Extract updated model
        updated_model = self._extract_and_validate_model(
            response, interpret_messages)

        return updated_model, model_impact, contradiction, arc_exhausted, result_digest, method_used

    @staticmethod
    def _extract_model_text(response):
        """Extract the UPDATED_MODEL text from an RI response."""
        if 'UPDATED_MODEL:' not in response:
            return ""
        model_text = response.split('UPDATED_MODEL:')[1]
        if 'END_MODEL' in model_text:
            model_text = model_text.split('END_MODEL')[0]
        return model_text.strip()

    @staticmethod
    def _count_content_sections(text):
        """Count research model sections, excluding Strategic Trajectory
        (which is managed separately via extract/re-splice)."""
        return sum(1 for line in text.split('\n')
                   if line.startswith('## ') and 'Strategic Trajectory' not in line)

    def _extract_and_validate_model(self, response, messages):
        """Extract UPDATED_MODEL from response, validate structure, retry on truncation."""
        model_text = self._extract_model_text(response)
        if not model_text:
            return self.research_model

        current_sections = self._count_content_sections(self.research_model)
        new_sections = self._count_content_sections(model_text)

        # First iteration — no baseline to compare against
        if current_sections == 0:
            return model_text

        # Model is structurally intact
        if new_sections >= current_sections:
            return model_text

        # Truncated — retry with code model
        logger.warning(
            f"Research model truncated: {new_sections} sections "
            f"vs {current_sections} — retrying with code model")

        fallback = self._get_fallback_model('Research Interpreter')
        if not fallback:
            logger.warning("No fallback model available — keeping existing model")
            return self.research_model

        retry_response = self._call_agent_with_retry(
            messages, 'Research Interpreter', model_override=fallback)
        retry_text = self._extract_model_text(retry_response or "")

        if retry_text and self._count_content_sections(retry_text) >= current_sections:
            logger.info(f"Code model retry succeeded: "
                        f"{self._count_content_sections(retry_text)} sections")
            return retry_text

        logger.warning(
            f"Code model retry also incomplete — keeping existing model")
        return self.research_model

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
    # STRATEGIC REVIEW (premium model)
    # ══════════════════════════════════════════════

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

    def _build_review_context(self):
        """Build the recent iteration context for the strategic review."""
        if not self.insight_tree:
            return "(No iterations yet)"

        active_nodes = sorted(
            [n for n in self.insight_tree.values() if n['status'] == 'active'],
            key=lambda n: n['chain_id'],
        )
        recent = active_nodes[-5:]

        parts = []
        for node in recent:
            score = node['quality_score']
            question = node['question']
            digest = node.get('result_digest', '')
            method = node.get('method_used', '')

            entry = f"- [{score}/10] Q: {question}"
            if digest:
                entry += f"\n  Finding: {digest}"
            if method:
                entry += f"\n  Method: {method}"
            parts.append(entry)

        return '\n'.join(parts)

    def _run_strategic_review(self, iteration):
        """Run premium model strategic review.

        Called every iteration after model update. The premium model:
        1. Checks whether the current commitment should hold/pivot/abandon
        2. Identifies missed opportunities and untested connections
        3. Rewrites the Strategic Trajectory section of the research model
        """
        if self.kill_signal:
            return

        logger.info(f"Running strategic review (iteration {iteration})")
        self.last_review_iteration = iteration

        recent_context = self._build_review_context()

        max_iters = getattr(self, '_max_iterations', 100)
        remaining = max(0, max_iters - iteration - 1)

        prompt = self.engine.prompts.strategic_review.format(
            seed_question=self.seed_question,
            iteration=iteration + 1,
            max_iterations=max_iters,
            remaining_iterations=remaining,
            data_profile=self.data_profile if self.data_profile else "(No profile available)",
            research_model=self.research_model,
            recent_context=recent_context,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_agent_with_retry(
            messages, 'Strategic Review', model_override=self.premium_model)

        if not response or not response.strip():
            logger.warning("Strategic review returned empty response")
            return

        # ── Parse COMMITMENT ──
        commitment_action = self._parse_field(response, 'COMMITMENT', default='HOLD',
                                               valid={'HOLD', 'PIVOT', 'ABANDON'})
        # Extract the reason text after the action (e.g., "ABANDON — Arc is complete...")
        commitment_reason = ""
        if 'COMMITMENT:' in response:
            reason_line = response.split('COMMITMENT:')[1].split('\n')[0]
            commitment_reason = reason_line.strip()

        # ── Parse NEXT_DIRECTION ──
        next_direction = ""
        if 'NEXT_DIRECTION:' in response:
            nd_text = response.split('NEXT_DIRECTION:')[1]
            # Take until next field marker
            for end_marker in ['PROBE_NEEDED:', 'ARC_COMPLETE:', 'MISSED:', 'UPDATED_TRAJECTORY:']:
                if end_marker in nd_text:
                    nd_text = nd_text.split(end_marker)[0]
                    break
            nd_text = nd_text.strip()
            if nd_text and nd_text.upper() not in ('UNCHANGED', 'NONE', 'N/A'):
                next_direction = nd_text[:500]  # cap length

        # ── Parse PROBE_NEEDED ──
        probe_needed = False
        if 'PROBE_NEEDED:' in response.upper():
            probe_line = response.upper().split('PROBE_NEEDED:')[1].split('\n')[0].strip()
            probe_needed = 'YES' in probe_line

        # ── Parse ARC_COMPLETE ──
        arc_complete = False
        if 'ARC_COMPLETE:' in response.upper():
            ac_line = response.upper().split('ARC_COMPLETE:')[1].split('\n')[0].strip()
            arc_complete = 'YES' in ac_line
        self._last_arc_complete = arc_complete

        # ── Parse EARLY_STOP ──
        self._early_stop_requested = False
        if 'EARLY_STOP:' in response.upper():
            es_line = response.upper().split('EARLY_STOP:')[1].split('\n')[0].strip()
            self._early_stop_requested = 'YES' in es_line
            if self._early_stop_requested:
                logger.info("Strategic review requested EARLY_STOP")

        # ── Parse MISSED ──
        missed = ""
        if 'MISSED:' in response:
            missed_text = response.split('MISSED:')[1]
            if 'UPDATED_TRAJECTORY:' in missed_text:
                missed_text = missed_text.split('UPDATED_TRAJECTORY:')[0]
            missed = missed_text.strip()
            if missed.upper() in ('NONE', 'N/A', ''):
                missed = ""

        # ── Parse UPDATED_TRAJECTORY ──
        updated_trajectory = ""
        if 'UPDATED_TRAJECTORY:' in response:
            traj_text = response.split('UPDATED_TRAJECTORY:')[1]
            if 'END_TRAJECTORY' in traj_text:
                traj_text = traj_text.split('END_TRAJECTORY')[0]
            updated_trajectory = traj_text.strip()

        # ── Apply commitment ──
        completing_arc = self.current_arc_direction  # save before overwrite
        if commitment_action in ('PIVOT', 'ABANDON'):
            if next_direction:
                self.strategic_next_direction = next_direction
                self.current_arc_direction = next_direction  # track what arc we're on
                logger.info(f"Strategic review: {commitment_action} → next direction: {next_direction[:100]}")
            else:
                self.strategic_next_direction = ""
                logger.info(f"Strategic review: {commitment_action} (no next direction provided)")
        else:
            # HOLD — maintain or establish commitment
            self.strategic_next_direction = ""
            logger.info("Strategic review: HOLD commitment")

        # ── REFRAMING PROBE: fires on any commitment when strategic review requests it ──
        reframing_override = False
        if probe_needed and self._should_run_reframing_probe(iteration):
            with style.spinner("Reframing probe"):
                reframing = self._run_reframing_probe(iteration)
            if reframing:
                self.strategic_next_direction = reframing
                self.current_arc_direction = reframing  # reframing starts a new arc
                reframing_override = True
                logger.info("Reframing probe override: setting next direction")
                self._review_events.append(f"Reframing: {reframing[:120]}")
                self.probe_history.append((iteration, reframing[:150]))
            else:
                self._review_events.append("Reframing probe: no alternative framing found")
                self.probe_history.append((iteration, "No reframing found"))

        # ── PERSPECTIVE ROTATION: fires on ABANDON when arc genuinely completed ──
        # Uses completing_arc (the arc that just finished), not current_arc_direction
        # (which was already updated to the next planned arc above).
        # Skip if reframing probe already overrode the direction — reframing takes priority.
        if commitment_action == 'ABANDON' and arc_complete and not reframing_override:
            if completing_arc and completing_arc not in self.completed_original_arcs:
                with style.spinner("Perspective rotation"):
                    perspectives = self._run_perspective_rotation(iteration, completing_arc=completing_arc)
                if perspectives:
                    # Mark the completing arc as rotated
                    self.completed_original_arcs.add(completing_arc)
                    # Force-select top-ranked perspective as next direction
                    selected = perspectives[0]
                    self.strategic_next_direction = selected['question']
                    self.current_arc_direction = selected['question']
                    # Prevent this perspective arc from triggering its own rotation
                    self.completed_original_arcs.add(selected['question'])
                    # Record for dashboard
                    parent = selected.get('parent_arc', '')[:60]
                    self.rotation_history.append((
                        iteration,
                        parent,
                        [{'name': p['name'], 'question': p['question'][:120]} for p in perspectives]
                    ))
                    self._review_events.append(
                        f"Perspective selected: {selected['name']} — {selected['differs'][:80]}")
                    for p in perspectives[1:]:
                        self._review_events.append(
                            f"Perspective (deferred): {p['name']}")

        # ── Record arc change for dashboard heatmap ──
        if self.strategic_next_direction:
            label = self._extract_arc_label(self.strategic_next_direction)
            self.arc_history.append((iteration + 1, label))

        # ── Record commitment for dashboard timeline ──
        self.commitment_history.append((iteration, commitment_action))

        # ── Splice updated trajectory into research model ──
        if updated_trajectory:
            self._update_strategic_trajectory(updated_trajectory)

        # ── Log missed opportunities ──
        if missed:
            logger.info(f"Strategic review missed opportunities: {missed[:200]}")

    def _update_strategic_trajectory(self, new_trajectory):
        """Splice the premium model's trajectory into the research model."""
        section_header = '## Strategic Trajectory'
        if section_header in self.research_model:
            # Find the section and replace everything up to next ## or end
            before = self.research_model.split(section_header)[0]
            after_section = self.research_model.split(section_header)[1]
            # Find next ## header
            next_header_idx = after_section.find('\n## ')
            if next_header_idx > 0:
                after = after_section[next_header_idx:]
            else:
                # Check for END_MODEL marker
                end_idx = after_section.find('END_MODEL')
                after = after_section[end_idx:] if end_idx > 0 else ""
            self.research_model = f"{before}{section_header}\n{new_trajectory}\n{after}"
        else:
            # Append if section doesn't exist yet
            self.research_model += f"\n\n{section_header}\n{new_trajectory}"

    # ══════════════════════════════════════════════
    # REFRAMING PROBE (premium model)
    # Fires when strategic review sets PROBE_NEEDED: YES
    # on any commitment (HOLD, PIVOT, or ABANDON).
    # ══════════════════════════════════════════════

    PROBE_MIN_ITERATION = 5   # don't probe in early exploration

    def _should_run_reframing_probe(self, iteration):
        """Check whether a reframing probe is allowed to fire.
        Called only when strategic review sets PROBE_NEEDED: YES.
        The iteration guard prevents probing during early exploration."""
        if iteration < self.PROBE_MIN_ITERATION:
            return False
        return True

    def _build_probe_context(self):
        """Build full-results context for the reframing probe.
        Returns (arc_summary, why_it_matters, full_results) or None."""
        # Arc summary from current commitment in trajectory
        trajectory = self._extract_model_section('Strategic Trajectory')
        arc_summary = ""
        if trajectory:
            for line in trajectory.split('\n'):
                if 'CURRENT COMMITMENT' in line.upper():
                    arc_summary = line.strip()
                    break
        if not arc_summary:
            arc_summary = "(Current arc not identified)"

        # Why it matters: try recent findings first, then biggest gap, then seed
        why_it_matters = ""
        # Recent established findings give context for positive-finding probes
        findings = self._extract_model_section('Established Findings')
        if findings:
            # Last 3 findings are most relevant to current arc
            finding_lines = [l.strip() for l in findings.split('\n') if l.strip().startswith('-')]
            if finding_lines:
                why_it_matters = "Recent findings:\n" + "\n".join(finding_lines[-3:])
        if not why_it_matters or len(why_it_matters) < 20:
            gap = self._extract_model_section('Biggest Gap')
            if gap and len(gap) > 10:
                why_it_matters = gap
        if not why_it_matters or len(why_it_matters) < 10:
            why_it_matters = self.seed_question

        # Full results from last 3 winning analyses (most recent by chain_id)
        if not self.insight_tree:
            return None

        recent_nodes = sorted(
            [n for n in self.insight_tree.values() if n['status'] == 'active'],
            key=lambda n: n['chain_id'],
        )[-3:]

        full_store = self.engine.message_manager.full_results_store
        results_parts = []
        for node in recent_nodes:
            chain_key = str(node['chain_id'])
            full_result = full_store.get(chain_key, node.get('result_summary', ''))
            if not full_result or len(full_result.strip()) < 20:
                continue
            results_parts.append(
                f"**Analysis: {node['question']}**\n"
                f"Score: {node['quality_score']}/10\n"
                f"Full output:\n{full_result}\n"
                f"{'─' * 40}"
            )

        if not results_parts:
            return None

        full_results = '\n\n'.join(results_parts)
        return arc_summary, why_it_matters, full_results

    def _run_reframing_probe(self, iteration):
        """Run the reframing probe: premium model reads full analytical output
        and looks for patterns the headline tests missed.

        Returns a reframing direction string, or empty string if no reframing found.
        """
        if self.kill_signal:
            return ""

        context = self._build_probe_context()
        if context is None:
            logger.info("Reframing probe: insufficient context, skipping")
            return ""

        arc_summary, why_it_matters, full_results = context
        self.last_probe_iteration = iteration

        prompt = self.engine.prompts.reframing_probe.format(
            seed_question=self.seed_question,
            arc_summary=arc_summary,
            why_it_matters=why_it_matters,
            full_results=full_results,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_agent_with_retry(
            messages, 'Reframing Probe', model_override=self.premium_model)

        if not response or not response.strip():
            logger.info("Reframing probe: empty response")
            return ""

        # Parse REFRAMING_DIRECTION
        direction = ""
        if 'REFRAMING_DIRECTION:' in response:
            rd_text = response.split('REFRAMING_DIRECTION:')[1].strip()
            # Take until end or next obvious section
            for end_marker in ['\n\n\n']:
                if end_marker in rd_text:
                    rd_text = rd_text.split(end_marker)[0]
                    break
            rd_text = rd_text.strip()
            if rd_text and rd_text.upper() not in ('NONE', 'N/A', ''):
                direction = rd_text[:500]

        if direction:
            # Log what the probe found
            hidden = ""
            if 'HIDDEN_PATTERN:' in response:
                hp = response.split('HIDDEN_PATTERN:')[1]
                if 'ALTERNATIVE_FRAMING:' in hp:
                    hp = hp.split('ALTERNATIVE_FRAMING:')[0]
                hidden = hp.strip()
            logger.info(f"Reframing probe found: {hidden}")
            logger.info(f"Reframing direction: {direction[:150]}")
        else:
            logger.info("Reframing probe: no reframing warranted (null result genuine)")

        return direction

    # ══════════════════════════════════════════════
    # PERSPECTIVE ROTATION (premium model)
    # Fires when an original arc completes (ABANDON
    # with arc COMPLETE). Generates alternative
    # analytical lenses. Does NOT fire on perspective
    # arcs spawned by previous rotations.
    # ══════════════════════════════════════════════

    ROTATION_MIN_ITERATION = 8  # don't rotate in early exploration

    def _run_perspective_rotation(self, iteration, completing_arc=None):
        """Generate alternative analytical perspectives on a completed arc.

        Args:
            completing_arc: the arc direction that just finished (not the next planned arc).

        Returns a list of perspective dicts [{name, differs, question}] or empty list.
        """
        if self.kill_signal:
            return []

        arc_name = completing_arc or self.current_arc_direction
        if not arc_name:
            return []

        if iteration < self.ROTATION_MIN_ITERATION:
            return []

        # Determine arc start iteration from arc_history
        arc_start = 0
        for start_iter, _label in self.arc_history:
            if start_iter <= iteration:
                arc_start = start_iter

        # Filter nodes to this arc's iteration range
        arc_nodes = [
            node for node in sorted(self.insight_tree.values(), key=lambda n: n['chain_id'])
            if node.get('iteration_added', 0) >= arc_start
            and node.get('iteration_added', 0) <= iteration
            and node['status'] == 'active'
        ]

        # Arc-specific methods (only from this arc's analyses)
        methods = []
        for node in arc_nodes:
            m = node.get('method_used', '')
            if m and m not in methods:
                methods.append(m)
        arc_methods = '; '.join(methods) if methods else "(not recorded)"

        # Arc-specific findings (result digests from arc nodes, not global Established Findings)
        findings_parts = []
        for node in arc_nodes:
            digest = node.get('result_digest', '')
            if digest:
                findings_parts.append(
                    f"[{node['quality_score']}/10] {node['question'][:80]}\n  {digest}"
                )
        arc_findings = '\n'.join(findings_parts) if findings_parts else "(No findings)"
        if len(arc_findings) > 2500:
            arc_findings = arc_findings[:2500]

        # Build column list from data profile (compact)
        columns = ""
        if hasattr(self.engine, 'df') and self.engine.df is not None:
            columns = ', '.join(self.engine.df.columns.tolist())
        if not columns:
            columns = "(columns not available)"

        # Build list of previously selected perspective names
        prior_perspectives = []
        for _, _, perspectives in self.rotation_history:
            if perspectives:
                prior_perspectives.append(perspectives[0]['name'])
        prior_perspectives_text = ', '.join(prior_perspectives) if prior_perspectives else "(none yet)"

        prompt = self.engine.prompts.perspective_rotation.format(
            seed_question=self.seed_question,
            arc_name=arc_name,
            arc_methods=arc_methods,
            arc_findings=arc_findings,
            available_columns=columns,
            previously_selected=prior_perspectives_text,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_agent_with_retry(
            messages, 'Perspective Rotation', model_override=self.premium_model)

        if not response or not response.strip():
            logger.info("Perspective rotation: empty response")
            return []

        if response.strip().upper() == 'NONE':
            logger.info("Perspective rotation: no alternative perspectives warranted")
            return []

        # Parse perspectives
        perspectives = []
        for i in range(1, 4):
            marker = f'PERSPECTIVE_{i}:'
            if marker not in response:
                continue
            block = response.split(marker)[1]
            # Take until next PERSPECTIVE_ or end
            for end in [f'PERSPECTIVE_{i+1}:', 'NONE']:
                if end in block:
                    block = block.split(end)[0]
                    break

            name = block.split('\n')[0].strip()
            differs = ""
            question = ""
            for line in block.split('\n'):
                if line.strip().startswith('DIFFERS:'):
                    differs = line.strip()[len('DIFFERS:'):].strip()
                elif line.strip().startswith('QUESTION:'):
                    question = line.strip()[len('QUESTION:'):].strip()

            if name and question:
                perspectives.append({
                    'name': name[:80],
                    'differs': differs[:200],
                    'question': question[:500],
                    'parent_arc': arc_name,
                    'spawned_at': iteration,
                })

        if perspectives:
            logger.info(f"Perspective rotation produced {len(perspectives)} perspectives: "
                       f"{', '.join(p['name'] for p in perspectives)}")
        else:
            logger.info("Perspective rotation: no valid perspectives parsed")

        return perspectives

    # ══════════════════════════════════════════════
    # AUTO-STOP DETECTION
    # ══════════════════════════════════════════════

    _MIN_ITERATIONS_BEFORE_STOP = 15  # don't stop before the investigation has matured
    _ABANDON_STREAK_THRESHOLD = 8     # consecutive ABANDONs to trigger mechanical backstop
    _SCORE_THRESHOLD = 6.0            # mean score below this during streak triggers stop

    def _should_stop(self, iteration):
        """Determine whether the investigation should stop early.

        Two independent triggers (either is sufficient):
        1. Strategic review explicitly requested EARLY_STOP: YES
        2. Mechanical backstop: last N consecutive commitments are ABANDON
           and the mean score during that streak is below threshold

        Returns True if auto-stop should trigger.
        """
        if iteration < self._MIN_ITERATIONS_BEFORE_STOP:
            return False

        # Signal 1: Strategic review explicit request
        if getattr(self, '_early_stop_requested', False):
            return True

        # Signal 2: Mechanical backstop — sustained ABANDON streak with low scores
        if len(self.commitment_history) < self._ABANDON_STREAK_THRESHOLD:
            return False

        recent = self.commitment_history[-self._ABANDON_STREAK_THRESHOLD:]
        if not all(action == 'ABANDON' for _, action in recent):
            return False

        # Check scores during the streak
        active = sorted(
            [n for n in self.insight_tree.values() if n['status'] == 'active'],
            key=lambda n: n['chain_id']
        )
        if len(active) < self._ABANDON_STREAK_THRESHOLD:
            return False

        streak_scores = [n['quality_score'] for n in active[-self._ABANDON_STREAK_THRESHOLD:]]
        mean_score = sum(streak_scores) / len(streak_scores)

        if mean_score < self._SCORE_THRESHOLD:
            logger.info(f"Auto-stop mechanical backstop: {self._ABANDON_STREAK_THRESHOLD} "
                         f"consecutive ABANDONs, mean score {mean_score:.1f}")
            return True

        return False

    # ══════════════════════════════════════════════
    # SEED DECOMPOSITION (premium model)
    # ══════════════════════════════════════════════

    def _decompose_seed_question(self, seed_question, max_iterations=5):
        """Use the premium model to decompose a broad research agenda into a focused
        first question and an initial Strategic Trajectory.

        Called once after orientation, before the main loop.
        Returns (focused_question, initial_trajectory) or (seed_question, "") on failure.
        """
        if self.kill_signal:
            return seed_question, ""

        profile = self.data_profile if self.data_profile else "(No profile available)"

        prompt = self.engine.prompts.seed_decomposition.format(
            seed_question=seed_question,
            data_profile=profile,
            max_iterations=max_iterations,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_agent_with_retry(
            messages, 'Seed Decomposition', model_override=self.premium_model)

        if not response or not response.strip():
            logger.warning("Seed decomposition returned empty — using original question")
            return seed_question, ""

        # Parse FIRST_QUESTION
        focused_question = seed_question  # fallback
        if 'FIRST_QUESTION:' in response:
            fq_text = response.split('FIRST_QUESTION:')[1]
            for end_marker in ['INITIAL_TRAJECTORY:', 'END_TRAJECTORY']:
                if end_marker in fq_text:
                    fq_text = fq_text.split(end_marker)[0]
                    break
            fq_text = fq_text.strip()
            if fq_text and len(fq_text) > 20:
                focused_question = fq_text[:1000]

        # Parse INITIAL_TRAJECTORY
        initial_trajectory = ""
        if 'INITIAL_TRAJECTORY:' in response:
            traj_text = response.split('INITIAL_TRAJECTORY:')[1]
            if 'END_TRAJECTORY' in traj_text:
                traj_text = traj_text.split('END_TRAJECTORY')[0]
            initial_trajectory = traj_text.strip()

        logger.info(f"Seed decomposed: {focused_question[:100]}")
        return focused_question, initial_trajectory

    # ══════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════

    def run(self, seed_question, initial_image=None, max_iterations=5, num_parallel_solutions=2,
            interactive=False, resumed_state=None, orientation=True, auto_stop=False):
        """Run the autonomous exploration loop.

        Args:
            auto_stop: If True, the system may terminate before max_iterations
                when the strategic review determines the investigation is complete.
                Default False — the full iteration budget is always used.
        """

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

                logger.info(f"Resuming from iteration {start_iteration}, "
                            f"running {max_iterations - start_iteration} more")
            else:
                current_questions = [seed_question]
                last_solution_chain = None
                last_follow_up_angle = None

                self.insight_tree = {}
                self.question_pool = []
                self.research_model = ""
                self.seed_question = seed_question
                self.commitment_history = []
                self.model_impact_history = []
                self.evaluator_score_history = []
                self.biggest_gap_history = []
                self.stagnation_count = 0
                self.data_profile = ""
                self.last_review_iteration = 0
                self.strategic_next_direction = ""
                self.current_arc_direction = ""
                self.last_probe_iteration = 0
                self.probe_history = []
                self.completed_original_arcs = set()
                self.rotation_history = []
                self.arc_history = []

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

            # ── SEED DECOMPOSITION (fresh runs only) ──
            if not resumed_state and not self.kill_signal:
                with style.spinner("Decomposing research agenda"):
                    focused_question, initial_trajectory = self._decompose_seed_question(
                        seed_question, max_iterations=max_iterations)
                if focused_question != seed_question:
                    current_questions = [focused_question]
                    logger.info(f"Seed decomposed into focused first question")
                if initial_trajectory:
                    self._initial_trajectory = initial_trajectory
                    logger.info(f"Initial trajectory: {len(initial_trajectory)} chars")
                self.arc_history.append((1, self._extract_arc_label(focused_question)))
                self.current_arc_direction = focused_question

            # ══════════════════════════════════════════════
            # MAIN ITERATION LOOP
            # ══════════════════════════════════════════════

            while iteration < max_iterations:
                self._current_iteration = iteration
                self._max_iterations = max_iterations

                if self.kill_signal:
                    break

                self.engine._iteration = iteration + 1
                self.engine._max_iterations = max_iterations

                # Determine questions to process
                if iteration == 0:
                    questions_to_process = current_questions[:1]
                elif iteration == 1:
                    questions_to_process = current_questions[:num_parallel_solutions + 1]
                else:
                    questions_to_process = current_questions[:num_parallel_solutions]

                # Commitment posture for iteration bar
                if self.commitment_history:
                    last_action = self.commitment_history[-1][1]
                    posture = last_action if last_action in ('HOLD', 'PIVOT', 'ABANDON') else 'EXPLORING'
                else:
                    posture = 'EXPLORING'
                print(style.iteration_bar(iteration + 1, max_iterations, posture))

                # ── PROCESS QUESTIONS ──
                solutions_data = []

                for q_idx, question in enumerate(questions_to_process):
                    if self.kill_signal:
                        break

                    self.chain_id = int(time.time()) + q_idx
                    solution_chain_id = self.chain_id

                    print()
                    print(style.question_display(q_idx + 1, len(questions_to_process), "", question))

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
                    })

                    self.engine.message_manager.reset_non_cumul_messages()
                    time.sleep(1)

                self.chain_id = int(time.time())
                supporting_chain_id = self.chain_id

                # ── EVALUATE ──
                if len(solutions_data) == 1:
                    selected_index = 0
                    scores = [7]
                    reason = "Seed question — establishing baseline"
                    follow_up_angle = None
                    summaries = ['']
                    _is_seed = True
                else:
                    with style.spinner("Evaluating results"):
                        (selected_index, scores, _, _,
                         reason, follow_up_angle, summaries) = \
                            self._evaluate_results_comparative(
                                solutions_data,
                                remaining_iterations=max_iterations - iteration,
                                max_iterations=max_iterations,
                            )
                    _is_seed = False

                self.evaluator_score_history.append(scores[selected_index])
                if len(self.evaluator_score_history) > 15:
                    self.evaluator_score_history = self.evaluator_score_history[-15:]

                # ── ENRICH QA PAIRS WITH FINDING SUMMARIES ──
                # Attach evaluator-generated summaries to qa_pairs so the code
                # generator gets high-quality context via format_qa_pairs.
                for i, sol in enumerate(solutions_data):
                    if i < len(summaries) and summaries[i]:
                        self.engine.message_manager.update_finding_summary(
                            sol['chain_id'], summaries[i])

                # ── UPDATE TREE ──
                winning_solution = solutions_data[selected_index]
                result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"

                new_node_id = self._add_node_to_tree(
                    question=winning_solution['question'],
                    result_summary=result_summary,
                    quality_score=scores[selected_index],
                    chain_id=winning_solution['chain_id'],
                )
                if selected_index < len(summaries) and summaries[selected_index]:
                    self.insight_tree[new_node_id]['finding_summary'] = summaries[selected_index]

                last_solution_chain = winning_solution['chain_id']
                last_follow_up_angle = follow_up_angle

                # ── INTERPRET & UPDATE MODEL ──
                # Save Strategic Trajectory before update — cheap model must not corrupt it
                _saved_trajectory = self._extract_model_section('Strategic Trajectory')
                if _saved_trajectory and '<<< DO NOT MODIFY' in _saved_trajectory:
                    _saved_trajectory = ""  # placeholder, not real content yet

                with style.spinner("Updating research model"):
                    updated_model, model_impact, contradiction, arc_exhausted, result_digest, method_used = \
                        self._interpret_and_update_model(
                            winning_solution,
                            scores[selected_index],
                        )

                if new_node_id in self.insight_tree and result_digest:
                    self.insight_tree[new_node_id]['result_digest'] = result_digest
                if new_node_id in self.insight_tree and method_used:
                    self.insight_tree[new_node_id]['method_used'] = method_used

                self.research_model = updated_model

                # ── PROTECT STRATEGIC TRAJECTORY ──
                # The model updater (cheap model) regenerates the full research model.
                # It may mangle the Strategic Trajectory despite "DO NOT MODIFY" instructions.
                # Structurally re-splice the trajectory we saved before the update.
                if _saved_trajectory:
                    self._update_strategic_trajectory(_saved_trajectory)

                # On iteration 0, splice the initial trajectory from seed decomposition
                if self._initial_trajectory:
                    self._update_strategic_trajectory(self._initial_trajectory)
                    self._initial_trajectory = ""  # consumed
                if len(self.research_model) > 6000:
                    self.research_model += (
                        "\n\n**NOTE: Model getting long — consolidate on next update.**"
                    )
                self.model_impact_history.append(model_impact)
                if len(self.model_impact_history) > 10:
                    self.model_impact_history = self.model_impact_history[-10:]

                self._update_gap_stability(updated_model)

                # ── STRATEGIC REVIEW (premium model) ──
                if iteration > 0 and not self.kill_signal:
                    self._review_events = []
                    with style.spinner("Strategic review"):
                        self._run_strategic_review(iteration)
                    for event in self._review_events:
                        print(style.branch_event(event))

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
                    )
                    if idx < len(summaries) and summaries[idx]:
                        self.insight_tree[node_id]['finding_summary'] = summaries[idx]
                    self.insight_tree[node_id]['status'] = 'runner_up'

                # ── WRITE ITERATION SUMMARY ──
                self.engine.output_manager.write_iteration_summary(
                    iteration=iteration + 1,
                    solutions_data=solutions_data,
                    scores=scores,
                    selected_index=selected_index,
                    model_impact=model_impact,
                    contradiction=contradiction,
                    arc_exhausted=arc_exhausted,
                )

                iteration += 1
                self._save_checkpoint(iteration, last_solution_chain, last_follow_up_angle)
                self._write_dashboard(iteration, max_iterations)

                if iteration >= max_iterations:
                    print(style.pipeline_summary(
                        selected_q=selected_index + 1,
                        selected_score=scores[selected_index],
                        reason=reason,
                        model_impact=model_impact,
                        n_questions=0,
                        n_selected=0,
                        is_seed=_is_seed,
                    ))
                    break
                if self.kill_signal:
                    break

                # ── AUTO-STOP CHECK ──
                if auto_stop and self._should_stop(iteration):
                    print(f"\n  {style.GREEN}●{style.RESET} {style.BOLD}Investigation complete "
                          f"— proceeding to synthesis{style.RESET} "
                          f"{style.DIM}(iteration {iteration} of {max_iterations}){style.RESET}")
                    break

                # ── ARC TRANSITION ──
                if getattr(self, '_last_arc_complete', False):
                    print(style.branch_event("Arc complete, new territory"))
                    self._last_arc_complete = False

                # ── GENERATE NEW QUESTIONS ──
                # Use strategic next_direction as the primary hint when set (after PIVOT/ABANDON)
                question_hint = self.strategic_next_direction or last_follow_up_angle
                with style.spinner("Generating questions"):
                    new_questions, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                        follow_up_hint=question_hint,
                    )

                # Retry once if parsing failed
                if not new_questions:
                    logger.info(f"Question generation empty. Raw: {raw_response[:200] if raw_response else '(empty)'}")
                    print(style.error_msg("Question generation failed, retrying..."))
                    new_questions, raw_response = self._generate_branching_questions(
                        use_chain_id=supporting_chain_id,
                        follow_up_hint=question_hint,
                    )

                if not new_questions:
                    print(style.error_msg("Could not generate questions. Exploration complete."))
                    break

                # Clear one-shot strategic direction after it's been consumed
                if self.strategic_next_direction:
                    self.strategic_next_direction = ""

                # ── SELECT QUESTIONS ──
                num_to_select = num_parallel_solutions + 1 if iteration == 1 else num_parallel_solutions
                with style.spinner("Selecting questions"):
                    selected_questions = self._select_best_questions(
                        new_questions,
                        context_hint=question_hint,
                        num_to_select=num_to_select,
                    )
                if not selected_questions:
                    break

                print(style.pipeline_summary(
                    selected_q=selected_index + 1,
                    selected_score=scores[selected_index],
                    reason=reason,
                    model_impact=model_impact,
                    n_questions=len(new_questions),
                    n_selected=len(selected_questions),
                    is_seed=_is_seed,
                ))

                current_questions = selected_questions
                current_image = None

            # ── POST-LOOP ──
            with style.spinner("Generating synthesis report"):
                synthesis_text = self._generate_synthesis(seed_question)

            if not synthesis_text:
                print(f"  {style.YELLOW}✗ Synthesis failed{style.RESET}")
                synthesis_text = ""

            # Generate charts for key findings (premium model)
            if synthesis_text and self.premium_model:
                synthesis_text = self._generate_synthesis_charts(synthesis_text)

            print(style.exploration_tree(self.insight_tree,
                                         arc_history=self.arc_history,
                                         total_iterations=iteration))

            active_nodes = [n for n in self.insight_tree.values() if n['status'] == 'active']
            avg_score = sum(n['quality_score'] for n in active_nodes) / len(active_nodes) if active_nodes else 0

            print(style.final_box(
                iterations=iteration,
                analyses=len(active_nodes),
                avg=avg_score,
                n_arcs=len(self.arc_history),
                cost_str=self.engine.cost_tracker.report(),
                output_dir=self.engine.output_dir,
            ))

            self.engine.output_manager.write_final_outputs(
                research_model=self.research_model,
                synthesis_text=synthesis_text,
                cost_tracker=self.engine.cost_tracker,
                run_logger=self.engine.run_logger,
            )

            # Final dashboard write — captures synthesis and charting costs
            self._write_dashboard(iteration, max_iterations)
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

    def _generate_synthesis_charts(self, synthesis_text):
        """Generate one publication-quality chart per key finding section.

        For each finding section, extracts the chain_id citations, retrieves
        the original analysis code that produced those numbers, and passes it
        to the premium model to adapt into a publication chart. This ensures
        charts use the exact same methodology as the original analysis.

        Returns updated synthesis text with embedded chart references.
        """
        if not synthesis_text or not self.premium_model:
            return synthesis_text

        # Sections that should NOT get charts
        skip_keywords = [
            'rejected', 'tested and rejected', 'caveats', 'caveat',
            'limitation', 'methodology', 'conclusion', 'next step',
            'recommended', 'open question', 'what is stable',
            'stable, what', 'executive summary', 'cross-cutting',
        ]

        # Parse sections
        lines = synthesis_text.split('\n')
        sections = []
        current_title = None
        current_lines = []

        for line in lines:
            if line.startswith('## ') and not line.startswith('### '):
                if current_title is not None:
                    sections.append((current_title, '\n'.join(current_lines)))
                current_title = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_title is not None:
            sections.append((current_title, '\n'.join(current_lines)))

        if not sections:
            return synthesis_text

        # Filter to chart-worthy sections
        chart_sections = []
        for title, text in sections:
            if any(kw in title.lower() for kw in skip_keywords):
                continue
            if len(text) < 200:
                continue
            chart_sections.append((title, text))

        if not chart_sections:
            return synthesis_text

        # Create charts directory
        charts_dir = os.path.join(self.engine.output_dir, "synthesis_charts")
        os.makedirs(charts_dir, exist_ok=True)

        schema = self.engine._get_df_schema()
        chart_map = {}  # title → relative image path

        for idx, (title, section_text) in enumerate(chart_sections):
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:40]
            chart_filename = f"{idx+1:02d}_{slug}.png"
            chart_path = os.path.join(charts_dir, chart_filename)

            # Extract chain_ids cited in this section [[chain_id]]
            cited_ids = re.findall(r'\[\[(\d+)\]\]', section_text)

            # Retrieve original code from the primary cited analysis
            original_code = self._get_analysis_code(cited_ids)

            if not original_code:
                logger.info(f"Synthesis chart: no source code found for '{title[:40]}'")
                continue

            # Build prompt with original code
            error_hints = self.engine.message_manager.format_error_patterns(
                static_hints=self.engine._load_pitfalls())
            prompt = self.engine.prompts.synthesis_chart.format(
                finding_text=section_text[:2000],
                original_code=original_code[:4000],
                schema=schema,
            )
            if error_hints:
                prompt = prompt + "\n\n" + error_hints
            messages = [
                {"role": "system", "content": self.engine.prompts.code_generator_system},
                {"role": "user", "content": prompt},
            ]

            try:
                with style.spinner(f"Charting: {title[:40]}"):
                    code, llm_response = self.engine._call_llm_for_code(
                        messages, self.premium_model, agent="Synthesis Chart")
                    if not code:
                        logger.info(f"Synthesis chart: no code for '{title[:40]}'")
                        continue

                    results, error, plots = self.engine.executor.execute(
                        code, self.engine.df, charts_dir)

                    # Error correction: up to 2 retries
                    retries = 0
                    while error and retries < 2:
                        retries += 1
                        logger.info(f"Synthesis chart retry {retries} for '{title[:40]}': "
                                     f"{error.strip().split(chr(10))[-1][:100]}")
                        fix_msg = (
                            f"The chart code produced an error:\n{error}\n\n"
                            f"Fix the code. Return the COMPLETE corrected code "
                            f"within ```python``` blocks. Chart only, no prints."
                        )
                        messages.append({"role": "assistant", "content": llm_response or ""})
                        messages.append({"role": "user", "content": fix_msg})
                        llm_response = self._call_agent_with_retry(
                            messages, 'Synthesis Chart', model_override=self.premium_model)
                        from executor import extract_code
                        code = extract_code(llm_response or "")
                        if code:
                            results, error, plots = self.engine.executor.execute(
                                code, self.engine.df, charts_dir)
                        else:
                            break

                    if error:
                        logger.info(f"Synthesis chart failed for '{title[:40]}' after "
                                     f"{retries + 1} attempts")
                        continue

                    if plots:
                        import shutil
                        shutil.move(plots[0], chart_path)
                        chart_map[title] = f"synthesis_charts/{chart_filename}"
                        logger.info(f"Synthesis chart saved: {chart_filename}")

            except Exception as e:
                logger.info(f"Synthesis chart failed for '{title[:40]}': {e}")
                continue

        if not chart_map:
            return synthesis_text

        # Embed chart references into synthesis text
        updated = synthesis_text
        for title, img_path in chart_map.items():
            header = f"## {title}"
            img_md = f"\n\n![{title}]({img_path})\n"
            header_pos = updated.find(header)
            if header_pos < 0:
                continue
            para_end = updated.find('\n\n', header_pos + len(header))
            if para_end < 0:
                para_end = header_pos + len(header)
            updated = updated[:para_end] + img_md + updated[para_end:]

        print(f"  {style.GREEN}✓{style.RESET} {len(chart_map)} synthesis charts generated")
        return updated

    def _get_analysis_code(self, chain_ids):
        """Retrieve original Python code from analysis.md files for given chain_ids.

        Searches the exploration directory for analysis files matching the cited
        chain_ids and extracts the code block. Returns the code from the first
        (primary) citation found.
        """
        if not chain_ids:
            return ""

        import glob
        exploration_dir = os.path.join(self.engine.output_dir, "exploration")

        for cid in chain_ids[:3]:  # try first 3 cited chain_ids
            # Glob for the analysis.md containing this chain_id
            pattern = os.path.join(exploration_dir, "*", str(cid), "analysis.md")
            matches = glob.glob(pattern)
            if not matches:
                continue

            try:
                with open(matches[0]) as f:
                    content = f.read()
                # Extract code block between ```python and ```
                code_start = content.find('```python')
                if code_start < 0:
                    continue
                code_start += len('```python\n')
                code_end = content.find('```', code_start)
                if code_end < 0:
                    continue
                code = content[code_start:code_end].strip()
                if code and len(code) > 30:
                    return code
            except Exception:
                continue

        return ""

    # ══════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════

    @staticmethod
    def _extract_arc_label(direction_text):
        """Extract a short label from a strategic direction for the dashboard heatmap."""
        label = direction_text.strip()
        # Take up to first sentence boundary
        for sep in ['. ', '? ', '.\n', '?\n', '\n']:
            idx = label.find(sep)
            if idx > 0:
                label = label[:idx]
                break
        label = label[:40].strip()
        if not label:
            label = 'New arc'
        return label

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

    def _add_node_to_tree(self, question, result_summary, quality_score, chain_id):
        node_id = self._generate_node_id()
        if result_summary:
            start_marker = "###RESULTS_START###"
            end_marker = "###RESULTS_END###"
            s_idx = result_summary.find(start_marker)
            e_idx = result_summary.find(end_marker)
            if s_idx >= 0 and e_idx > s_idx:
                result_summary = result_summary[s_idx + len(start_marker):e_idx].strip()
        self.insight_tree[node_id] = {
            'question': question,
            'result_summary': result_summary,
            'quality_score': quality_score,
            'status': 'active',
            'chain_id': chain_id,
            'finding_summary': '',
            'result_digest': '',
            'method_used': '',
            'iteration_added': getattr(self, '_current_iteration', 0),
        }
        return node_id

    def _get_exploration_history(self, max_entries=40, full_detail_count=15):
        """Build exploration history with tiered compaction."""
        if not self.insight_tree:
            return ""
        active_nodes = sorted(
            [n for n in self.insight_tree.values() if n['status'] == 'active'],
            key=lambda n: n['chain_id'],
        )
        recent = active_nodes[-max_entries:]

        if len(recent) > full_detail_count:
            compact_nodes = recent[:-full_detail_count]
            full_nodes = recent[-full_detail_count:]
        else:
            compact_nodes = []
            full_nodes = recent

        history_parts = ["**Exploration History:**"]

        if compact_nodes:
            history_parts.append(f"\n*Earlier analyses ({len(compact_nodes)} entries):*")
            for node in compact_nodes:
                score = node['quality_score']
                question = node['question']
                summary = node.get('finding_summary', '')
                if not summary:
                    finding = node['result_summary'] if node['result_summary'] else "No results"
                    summary = finding.split('\n')[0][:200]
                history_parts.append(
                    f"- [{score}/10] Q: {question}\n  → {summary}"
                )

        if full_nodes:
            if compact_nodes:
                history_parts.append(f"\n*Recent analyses ({len(full_nodes)} entries):*")
            for node in full_nodes:
                score = node['quality_score']
                question = node['question']
                digest = node.get('result_digest', '')

                if digest:
                    history_parts.append(
                        f"- [{score}/10] Q: {question}\n  Finding: {digest}"
                    )
                else:
                    summary = node.get('finding_summary', '')
                    if not summary:
                        finding = node['result_summary'] if node['result_summary'] else "No results"
                        summary = finding.split('\n')[0][:200]
                    history_parts.append(
                        f"- [{score}/10] Q: {question}\n  → {summary}"
                    )

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
            "version": 5,
            "iterations_completed": iteration,
            "explorer": {
                "insight_tree": self.insight_tree,
                "question_pool": self.question_pool,
                "research_model": self.research_model,
                "seed_question": self.seed_question,
                "commitment_history": self.commitment_history,
                "model_impact_history": self.model_impact_history,
                "evaluator_score_history": self.evaluator_score_history,
                "biggest_gap_history": self.biggest_gap_history,
                "stagnation_count": self.stagnation_count,
                "node_counter": AutoExplorer._node_counter,
                "data_profile": self.data_profile,
                "last_review_iteration": self.last_review_iteration,
                "strategic_next_direction": self.strategic_next_direction,
                "current_arc_direction": self.current_arc_direction,
                "last_probe_iteration": self.last_probe_iteration,
                "probe_history": self.probe_history,
                "completed_original_arcs": list(self.completed_original_arcs),
                "rotation_history": self.rotation_history,
                "arc_history": self.arc_history,
            },
            "message_manager": {
                "qa_pairs": self.engine.message_manager.qa_pairs,
                "all_questions": self.engine.message_manager.all_questions,
                "full_results_store": self.engine.message_manager.full_results_store,
                "error_patterns": self.engine.message_manager.error_patterns,
            },
            "last_solution_chain": last_solution_chain,
            "last_follow_up_angle": last_follow_up_angle,
        }

        path = os.path.join(self.engine.output_dir, "state.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, path)

    def _write_dashboard(self, iteration, max_iterations):
        """Write live dashboard HTML (best-effort, failures are silent)."""
        try:
            from dashboard import write_dashboard
            write_dashboard(self.engine.output_dir, self, self.engine,
                            iteration, max_iterations)
        except Exception as e:
            logger.debug(f"Dashboard write skipped: {e}")

    def _restore_checkpoint(self, state):
        """Restore exploration state from a loaded checkpoint dict."""
        ex = state['explorer']
        self.insight_tree = ex['insight_tree']
        self.question_pool = ex['question_pool']
        self.research_model = ex['research_model']
        self.seed_question = ex.get('seed_question', '')
        self.commitment_history = ex.get('commitment_history', [])
        self.model_impact_history = ex['model_impact_history']
        self.evaluator_score_history = ex.get('evaluator_score_history', [])
        self.biggest_gap_history = ex.get('biggest_gap_history', [])
        self.stagnation_count = ex.get('stagnation_count', 0)
        AutoExplorer._node_counter = ex.get('node_counter', 0)
        self.data_profile = ex.get('data_profile', '')
        self.last_review_iteration = ex.get('last_review_iteration',
                                             ex.get('last_connection_iteration', 0))
        self.strategic_next_direction = ex.get('strategic_next_direction', '')
        self.current_arc_direction = ex.get('current_arc_direction', '')
        self.last_probe_iteration = ex.get('last_probe_iteration', 0)
        self.probe_history = ex.get('probe_history', [])
        self.completed_original_arcs = set(ex.get('completed_original_arcs', []))
        self.rotation_history = ex.get('rotation_history', [])
        self.arc_history = ex.get('arc_history', [])
        self.engine.data_profile = self.data_profile

        mm = state['message_manager']
        self.engine.message_manager.qa_pairs = mm['qa_pairs']
        self.engine.message_manager.all_questions = mm['all_questions']
        self.engine.message_manager.full_results_store = mm['full_results_store']
        self.engine.message_manager.error_patterns = mm.get('error_patterns', [])


# ══════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════

def format_synthesis_input(insight_tree, full_results_store, research_model,
                           seed_question, data_profile=''):
    """Build structured synthesis input from exploration state.

    Section order: Context → Findings Index → Research Model → Evidence.
    The Research Model comes before Evidence so the synthesis model reads
    the strategic narrative before drilling into raw numbers.

    Evidence is score-gated:
      8+: full results (ground truth for key findings)
      6-7: finding_summary only (context without noise)
      ≤5:  omitted (kept in Findings Index for completeness)
    """
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

    # Section B: Findings Index (all analyses, one line each)
    parts.append("═══════════════════════════════════════")
    parts.append("SECTION B: COMPLETE FINDINGS INDEX")
    parts.append("═══════════════════════════════════════\n")

    for n in active:
        fs = n.get('finding_summary', '') or '(no summary)'
        parts.append(f"[{n['quality_score']}] [[{n['chain_id']}]] {fs}")

    # Section C: Research Model (read before evidence — provides the map)
    parts.append("\n═══════════════════════════════════════")
    parts.append("SECTION C: RESEARCH MODEL")
    parts.append("═══════════════════════════════════════\n")
    parts.append(research_model or "(No research model available)")

    # Section D: Evidence (score-gated)
    high = [n for n in active if n['quality_score'] >= 8]
    mid = [n for n in active if 6 <= n['quality_score'] <= 7]

    parts.append("\n═══════════════════════════════════════")
    parts.append(f"SECTION D: EVIDENCE (score 8+: full results, score 6-7: summaries)")
    parts.append("═══════════════════════════════════════\n")

    # Score 8+: full results
    for n in high:
        chain_key = str(n['chain_id'])
        result_text = full_results_store.get(
            chain_key, n.get('result_summary', 'Results not available')
        )
        if len(result_text) > 4000:
            result_text = result_text[:2000] + "\n[...truncated...]\n" + result_text[-2000:]

        parts.append(
            f"[[{n['chain_id']}]] Score: {n['quality_score']}/10\n"
            f"Question: {n['question']}\n"
            f"Results:\n{result_text}\n"
            f"{'─' * 5}"
        )

    # Score 6-7: finding summary only
    if mid:
        parts.append(f"\n{'─' * 20}\nScore 6-7 analyses (summaries only — see Findings Index for IDs):\n")
        for n in mid:
            fs = n.get('finding_summary', '')
            if not fs:
                fs = n.get('result_digest', '')
            if not fs:
                fs = n.get('result_summary', 'No summary')
                if len(fs) > 200:
                    fs = fs[:200] + '...'
            parts.append(f"[[{n['chain_id']}]] [{n['quality_score']}/10] {n['question']}\n  → {fs}")

    return '\n'.join(parts)
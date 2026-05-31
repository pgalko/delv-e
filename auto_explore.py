"""
Auto-explore module for autonomous recursive exploration.
Handles the exploration loop, result evaluation, and adaptive branching.

Key mechanisms:
  - Research model: 6 sections, 4 RI-maintained (Established Findings with
    STATUS tags, Active Hypotheses, Attention Flags, Exploration Health) and
    2 Opus-protected (Strategic Trajectory, Structural Landscape). Both
    protected sections use save-before/re-splice-after to survive RI rewrites.
  - STATUS tags on Established Findings: [ESTABLISHED] / [PROVISIONAL] /
    [SHRINKS] / [CONTRADICTED], assigned at RI level and updated as evidence
    accumulates. The briefing renders §2 directly from these tags.
  - Structural Landscape: Opus-protected section accumulating identifiability,
    coverage, foreclosed directions, and open questions. Seeded by orientation,
    extended by strategic review as structural discoveries arrive. The briefing
    renders §1/§3/§4 directly from this section.
  - Strategic Review: premium model runs every iteration to enforce commitment,
    detect missed opportunities, and maintain both Strategic Trajectory and
    Structural Landscape. UPDATED_STRUCTURAL_LANDSCAPE is emitted only when
    structural change has accumulated; absence preserves existing landscape.
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

import concurrent.futures
import json
import os
import re
import sys
import time
import traceback

import style

from logger_config import get_logger
logger = get_logger(__name__)


# Number of top-scored winning analyses to include with full untruncated
# stdout in the synthesis input. Remaining winners with score >= 6 are
# included via digests only. Lowering reduces context pressure at synthesis
# time; raising gives the briefing generator more raw evidence to cite.
TOP_K_FULL_RAW = 10


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

        # Orientation data profile
        self.data_profile = ""

        # Causal Substrate (Phase 3 of orientation). Lives in research model
        # as ## Causal Substrate section. Cheap agents read a compacted view
        # of the substrate (TYPE line + CANDIDATE MATCHING AXES table)
        # extracted by _research_model_for_cheap_agent(); premium agents read
        # the full block via direct research_model access. Set at orientation
        # time; may be refined by Strategic Review (rare).

        # ── Strategic review state ──
        self.last_review_iteration = 0
        self.strategic_next_direction = ""   # set by premium model on PIVOT/ABANDON
        self.current_arc_direction = ""      # the arc currently being pursued
        self._initial_trajectory = ""        # set by seed decomposition, consumed on iteration 0
        self.last_probe_iteration = 0        # tracks when last probe ran (observability)
        self.probe_history = []              # [(iteration, brief_result)] for dashboard
        self._last_probe_suspicion = ""      # set by SR when probe is adversarial; consumed
                                             # by the probe call at the next iteration; cleared
                                             # implicitly on next SR (which re-emits or omits)
        self.completed_original_arcs = set() # arc directions that have been rotated (no recursion)
        self.rotation_history = []           # [(iteration, parent_arc, [{name, question}])] for dashboard
        self.arc_history = []                # [(start_iter, label)] for dashboard heatmap

        # ── Arc reference code (implementation consistency) ──
        self._arc_reference_code = ""       # winning code from best score-8+ in current arc
        self._arc_reference_score = 0       # score of the reference code

        # ── Literature search ──
        self.search_model = None            # set from run.py; None = search disabled
        self.search_history = []            # [(iteration, query, summary)] for dashboard
        self._published_entries = []        # [PUBLISHED] entries, protected from RI
        self._last_search_calibration = ""  # most recent CALIBRATION assessment from
                                            # literature_integration: ALIGNED / NOVEL /
                                            # SUSPECT — passed to next SR call

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

    def _reset_state(self, seed_question):
        """Reset all exploration state for a fresh run.

        Mirrors __init__'s state initialization. Single source of truth so
        a new state variable is added in ONE place (here plus __init__).
        Called from run() when a fresh-run state.json doesn't exist.
        """
        self.insight_tree = {}
        self.question_pool = []
        self.research_model = ""
        self.seed_question = seed_question
        self.commitment_history = []
        self.data_profile = ""
        self.last_review_iteration = 0
        self.strategic_next_direction = ""
        self.current_arc_direction = ""
        self._initial_trajectory = ""
        self.last_probe_iteration = 0
        self.probe_history = []
        self._last_probe_suspicion = ""
        self.completed_original_arcs = set()
        self.rotation_history = []
        self.arc_history = []
        self._arc_reference_code = ""
        self._arc_reference_score = 0
        self.search_history = []
        self._published_entries = []
        self._last_search_calibration = ""

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

        # Cheap-agent read: substitute Causal Substrate block with compacted view
        rm_for_cheap = self._research_model_for_cheap_agent()
        research_model_context = rm_for_cheap if rm_for_cheap else "(No model yet)"
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

        # Most recent active node's tested_estimand — for seed-relevance scoring.
        # The evaluator caps scores at 6 when a sophisticated analysis tests a
        # narrowed estimand without seed-relevance gain.
        recent_estimand = ""
        if self.insight_tree:
            active_recent = sorted(
                [n for n in self.insight_tree.values() if n['status'] == 'active'],
                key=lambda n: n['chain_id'],
            )
            if active_recent:
                recent_estimand = active_recent[-1].get('tested_estimand', '')
        if not recent_estimand:
            recent_estimand = "(none recorded yet — first iteration or earlier RI did not emit field)"

        eval_prompt = self.engine.prompts.result_evaluator.format(
            seed_question=self.seed_question,
            recent_estimand=recent_estimand,
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

        # Cheap-agent read: substitute Causal Substrate block with one-liner guidance
        rm_for_cheap = self._research_model_for_cheap_agent()
        model_context = rm_for_cheap if rm_for_cheap else "(No model yet — first iteration)"

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
            seed_question=self.seed_question,
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

        # Cheap-agent read: substitute Causal Substrate block with one-liner guidance
        rm_for_cheap = self._research_model_for_cheap_agent()
        research_model_context = rm_for_cheap if rm_for_cheap else "(No model yet)"
        if self.strategic_next_direction:
            commitment_ctx = f"**Current commitment:** Pivoting to new territory — prefer breadth and diversity."
        elif self.current_arc_direction:
            arc_short = self.current_arc_direction[:80]
            commitment_ctx = f"**Current commitment:** Holding on current arc — {arc_short}"
        else:
            commitment_ctx = "**Current commitment:** Early exploration — prefer breadth."

        selection_prompt = self.engine.prompts.question_selector.format(
            seed_question=self.seed_question,
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
                    'iteration_added': getattr(self, '_current_iteration', 0) + 1,
                })
        self._trim_question_pool()

        return selected_questions

    def _parse_questions(self, questions_response):
        """Parse numbered questions from LLM response."""
        questions = []

        if not questions_response or not questions_response.strip():
            return []

        # Pattern to strip LLM-generated tags (e.g., [EXPLORE], [PURSUING]) that
        # some models add despite instructions not to. Maturity-stage tags
        # (DETECTED/QUANTIFIED/DECOMPOSED/REGIME-TESTED/COMPLETE) were relevant
        # under the old Finding Maturity system and are no longer emitted by any
        # current prompt; kept in the pattern as defensive insurance against
        # old-prompt contamination on --continue from v5 state.
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

        [PUBLISHED] entries, Strategic Trajectory, and Structural Landscape are
        all protected from RI modification. [PUBLISHED] is extracted/re-spliced
        here. Strategic Trajectory and Structural Landscape are extracted/re-spliced
        at the caller (_interpret_and_update_model is invoked within a
        save-before/re-splice-after block in run()).

        Returns:
            (updated_model, model_impact, contradiction, arc_exhausted,
             result_digest, method_used, tested_estimand)

        tested_estimand is a one-sentence description of what THIS analysis
        actually estimates and for which subset of the data — used by the
        Strategic Review's scope-drift check.
        """
        if self.kill_signal:
            return self.research_model, "LOW", False, False, "", "", ""

        # ── EXTRACT [PUBLISHED] ENTRIES (protect from RI) ──
        self._extract_published_snapshot()

        result_summary = str(winning_solution['results']) if winning_solution['results'] else "No results"
        # Cheap-agent read: substitute Causal Substrate block with one-liner guidance.
        # Note: RI's output goes through _splice_protected_section re-splice for
        # Causal Substrate (alongside Trajectory and Landscape), so its view of
        # the substrate being simplified doesn't propagate — the full substrate
        # is re-spliced into self.research_model after this call.
        rm_for_cheap = self._research_model_for_cheap_agent()
        current_model = rm_for_cheap if rm_for_cheap else "(No model yet — this is the first result. Initialize the model from scratch.)"

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

        # Parse structured fields. Every field has a sensible default so a
        # truncated or malformed RI response falls back gracefully rather
        # than wiping state.
        model_impact = self._parse_field(response, 'MODEL_IMPACT', default='MEDIUM',
                                          valid={'HIGH', 'MEDIUM', 'LOW'})
        contradiction = 'YES' in self._parse_field(response, 'CONTRADICTION', default='NO')
        arc_exhausted = 'YES' in self._parse_field(response, 'ARC_EXHAUSTED', default='NO')

        # Parse RESULT_DIGEST (3-5 lines of key numbers)
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

        # Parse TESTED_ESTIMAND (one sentence describing what THIS analysis
        # actually estimates and for which subset). Used by SR's scope-drift
        # check to detect when the tested question has narrowed from the seed.
        tested_estimand = ""
        if 'TESTED_ESTIMAND:' in response:
            te_text = response.split('TESTED_ESTIMAND:')[1]
            # Take until the next field marker, decorative task separator, or
            # blank-line boundary. The ═ catches LLM-emitted "═══ TASK X ═══"
            # banners that leak from multi-task prompts into adjacent fields.
            for end_marker in ['═', 'UPDATED_MODEL:', 'METHOD_USED:',
                               'RESULT_DIGEST:', '\n\n\n']:
                if end_marker in te_text:
                    te_text = te_text.split(end_marker)[0]
                    break
            te_text = te_text.strip()
            if te_text and te_text.upper() not in ('', 'NONE', 'N/A'):
                tested_estimand = te_text[:500]  # cap defensively

        # Extract updated model (validates structure, retries via fallback if truncated)
        updated_model = self._extract_and_validate_model(
            response, interpret_messages)

        # ── SPLICE [PUBLISHED] ENTRIES BACK IN ──
        updated_model = self._splice_published_entries(updated_model)

        return updated_model, model_impact, contradiction, arc_exhausted, result_digest, method_used, tested_estimand

    def _extract_published_snapshot(self):
        """Extract [PUBLISHED] entries from current research model and store them.
        Called before RI runs to protect entries from modification."""
        if not hasattr(self, '_published_entries'):
            self._published_entries = []

        if not self.research_model:
            self._published_entries = []
            return

        published = []
        for line in self.research_model.split('\n'):
            if '[PUBLISHED]' in line:
                # Keep original formatting (bullet, status, etc)
                published.append(line.rstrip())
        self._published_entries = published

    def _splice_published_entries(self, model_text):
        """Strip any [PUBLISHED] entries from model_text and re-insert the stored
        canonical set into the Established Findings section.

        This enforces that only the integration call can modify [PUBLISHED] entries
        — the RI's output is stripped of them regardless of what it wrote.
        """
        if not hasattr(self, '_published_entries') or not self._published_entries:
            # No stored entries — just strip any [PUBLISHED] the RI may have invented
            lines = model_text.split('\n')
            result_lines = [l for l in lines
                            if '[PUBLISHED]' not in l
                            and not l.strip().startswith('STATUS:')
                            and not l.strip().startswith('- STATUS:')]
            return '\n'.join(result_lines)

        # Strip existing [PUBLISHED] lines and orphaned STATUS lines
        lines = model_text.split('\n')
        stripped = []
        for line in lines:
            if '[PUBLISHED]' in line:
                continue
            if line.strip().startswith('STATUS:'):
                continue
            if line.strip().startswith('- STATUS:'):
                continue
            stripped.append(line)
        model_text = '\n'.join(stripped)

        # Insert stored entries at end of Established Findings section
        if '## Established Findings' not in model_text:
            # Section missing — append at end (degenerate case)
            return model_text + '\n\n## Established Findings\n' + '\n'.join(self._published_entries)

        lines = model_text.split('\n')
        result_lines = []
        in_ef = False
        inserted = False

        for line in lines:
            if '## Established Findings' in line:
                in_ef = True
                result_lines.append(line)
                continue
            if line.startswith('## ') and in_ef:
                # Insert published entries before next section
                for pub in self._published_entries:
                    result_lines.append(pub)
                result_lines.append('')
                in_ef = False
                inserted = True
            result_lines.append(line)

        # If Established Findings was the last section
        if in_ef and not inserted:
            for pub in self._published_entries:
                result_lines.append(pub)

        return '\n'.join(result_lines)

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
    def _extract_orientation_landscape(profile_text):
        """Extract the ###STRUCTURAL_LANDSCAPE_START### block from orientation.

        The orientation prompt requires this block on every run; it seeds the
        research model's Structural Landscape section. Returns empty string
        if the block is missing (e.g., when running with a legacy orientation
        prompt or if orientation output was mangled).
        """
        if not profile_text or '###STRUCTURAL_LANDSCAPE_START###' not in profile_text:
            return ""
        after = profile_text.split('###STRUCTURAL_LANDSCAPE_START###', 1)[1]
        if '###STRUCTURAL_LANDSCAPE_END###' in after:
            block = after.split('###STRUCTURAL_LANDSCAPE_END###', 1)[0]
        else:
            # Fallback: take until next marker or PROFILE_END
            for end_marker in ('###KEY_CONSTRAINTS_END###', '###PROFILE_END###', '\n\n\n'):
                if end_marker in after:
                    block = after.split(end_marker, 1)[0]
                    break
            else:
                block = after
        return block.strip()

    @staticmethod
    def _strip_landscape_from_profile(profile_text):
        """Remove the ###STRUCTURAL_LANDSCAPE_*### block from a profile.

        Called after _extract_orientation_landscape has captured the block
        into the research model. Removing it from data_profile prevents
        downstream agents from seeing a stale Landscape alongside the live
        one in the research model — the research model's Landscape is
        updated by Strategic Review as the investigation discovers new
        structural facts, while data_profile is frozen at orientation. Two
        copies of the same section, one of them stale, creates conflicts
        that the LLMs have to reconcile at every call. This helper ensures
        a single source of truth: the research model owns the Landscape;
        data_profile owns static context (KEY CONSTRAINTS + sections 1–6).

        If the markers are not present (legacy profile, mangled output),
        returns the input unchanged.
        """
        if not profile_text or '###STRUCTURAL_LANDSCAPE_START###' not in profile_text:
            return profile_text
        start = profile_text.find('###STRUCTURAL_LANDSCAPE_START###')
        end = profile_text.find('###STRUCTURAL_LANDSCAPE_END###')
        if end < 0 or end < start:
            return profile_text
        end += len('###STRUCTURAL_LANDSCAPE_END###')
        # Trim any leading whitespace after the removal to avoid blank lines
        before = profile_text[:start].rstrip()
        after = profile_text[end:].lstrip('\n')
        if before and after:
            return before + '\n\n' + after
        return before + after

    @staticmethod
    def _migrate_legacy_model(model_text):
        """Strip removed sections from a pre-v6 research model.

        Removes: Cross-Finding Connections, Finding Maturity, Biggest Gap.
        Leaves other sections untouched. Safe to call on a model that
        doesn't contain any of these sections (no-op).

        The next RI call will rewrite the model in the new 4+2 shape;
        this just ensures the RI prompt isn't confused by legacy sections
        during the transition iteration.
        """
        import re as _re
        legacy_sections = [
            'Cross-Finding Connections',
            'Finding Maturity',
            'Biggest Gap',
        ]
        for section_name in legacy_sections:
            section_header = f'## {section_name}'
            if section_header not in model_text:
                continue
            # Find section start
            before = model_text.split(section_header)[0]
            after_section = model_text.split(section_header, 1)[1]
            # Find where this section ends (next ## header or end-of-string)
            next_header_idx = after_section.find('\n## ')
            if next_header_idx > 0:
                after = after_section[next_header_idx:]
            else:
                # Check for END_MODEL marker
                end_idx = after_section.find('END_MODEL')
                after = after_section[end_idx:] if end_idx > 0 else ""
            model_text = before + after
            # Tidy up possible double newlines
            model_text = _re.sub(r'\n{3,}', '\n\n', model_text)
        return model_text.strip()

    @staticmethod
    def _count_content_sections(text):
        """Count RI-maintained research model sections, excluding
        Opus-protected sections (Strategic Trajectory, Structural Landscape,
        Causal Substrate) which are managed separately via extract/re-splice."""
        return sum(1 for line in text.split('\n')
                   if line.startswith('## ')
                   and 'Strategic Trajectory' not in line
                   and 'Structural Landscape' not in line
                   and 'Causal Substrate' not in line)

    # Expected RI-maintained content sections (excluding Opus-protected
    # Strategic Trajectory, Structural Landscape, and Causal Substrate,
    # which are spliced in separately):
    #   Established Findings, Active Hypotheses, Attention Flags, Exploration Health
    _EXPECTED_MODEL_SECTIONS = 4

    def _extract_and_validate_model(self, response, messages):
        """Extract UPDATED_MODEL from response, validate structure, retry on truncation."""
        model_text = self._extract_model_text(response)
        if not model_text:
            return self.research_model

        new_sections = self._count_content_sections(model_text)

        # First iteration — no baseline to compare against
        current_sections = self._count_content_sections(self.research_model)
        if current_sections == 0:
            return model_text

        # Compare against expected structure, not current model
        # (current may be inflated by rogue sections from integration calls)
        threshold = min(current_sections, self._EXPECTED_MODEL_SECTIONS)

        # Model is structurally intact
        if new_sections >= threshold:
            return model_text

        # Truncated — retry with code model
        logger.warning(
            f"Research model truncated: {new_sections} sections "
            f"vs {threshold} expected — retrying with code model")

        fallback = self._get_fallback_model('Research Interpreter')
        if not fallback:
            logger.warning("No fallback model available — keeping existing model")
            return self.research_model

        retry_response = self._call_agent_with_retry(
            messages, 'Research Interpreter', model_override=fallback)
        retry_text = self._extract_model_text(retry_response or "")

        if retry_text and self._count_content_sections(retry_text) >= threshold:
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
        """Build the recent iteration context for the strategic review.

        Includes per-node TESTED_ESTIMAND so SR can detect scope drift —
        the estimand the analysis actually addressed may have narrowed
        from the seed without anyone noticing.
        """
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
            estimand = node.get('tested_estimand', '')

            entry = f"- [{score}/10] Q: {question}"
            if digest:
                entry += f"\n  Finding: {digest}"
            if method:
                entry += f"\n  Method: {method}"
            if estimand:
                entry += f"\n  Tested estimand: {estimand}"
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

        # Latest tested estimand for the scope-drift check. The most recent
        # active node's tested_estimand is the relevant comparison point
        # against the seed.
        recent_estimand = ""
        if self.insight_tree:
            active_recent = sorted(
                [n for n in self.insight_tree.values() if n['status'] == 'active'],
                key=lambda n: n['chain_id'],
            )
            if active_recent:
                recent_estimand = active_recent[-1].get('tested_estimand', '')
        if not recent_estimand:
            recent_estimand = "(not recorded; treat as equivalent to seed)"

        # Most recent literature CALIBRATION assessment, if any. Drives
        # SR's Task 5 decision on whether contradicting findings should be
        # treated as "potentially novel" or as a methodological flag.
        search_cal = self._last_search_calibration if self._last_search_calibration else "(no recent search)"

        prompt = self.engine.prompts.strategic_review.format(
            seed_question=self.seed_question,
            iteration=iteration + 1,
            max_iterations=max_iters,
            remaining_iterations=remaining,
            data_profile=self.data_profile if self.data_profile else "(No profile available)",
            research_model=self.research_model,
            recent_context=recent_context,
            recent_estimand=recent_estimand,
            search_calibration=search_cal,
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

        # ── Parse PROBE_NEEDED + PROBE_SUSPICION ──
        # PROBE_SUSPICION is set when SR triggers an adversarial probe
        # (case (e) in PROBE_NEEDED). It's a one-sentence statement naming
        # the target finding's chain_id and the specific suspicion. Empty
        # for non-adversarial probes (cases a-d) and for PROBE_NEEDED: NO.
        probe_needed = False
        if 'PROBE_NEEDED:' in response.upper():
            probe_line = response.upper().split('PROBE_NEEDED:')[1].split('\n')[0].strip()
            probe_needed = 'YES' in probe_line
        self._last_probe_suspicion = ""
        if 'PROBE_SUSPICION:' in response:
            ps_text = response.split('PROBE_SUSPICION:')[1].split('\n')[0].strip()
            if ps_text and ps_text.upper() not in ('NONE', 'N/A', ''):
                self._last_probe_suspicion = ps_text[:500]

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

        # ── Parse SEARCH_NEEDED ──
        search_query = ""
        if 'SEARCH_NEEDED:' in response:
            sq_text = response.split('SEARCH_NEEDED:')[1].split('\n')[0].strip()
            # SR may append explanation: "NONE — already have relevant entries"
            sq_keyword = sq_text.split('—')[0].split('-')[0].strip().upper()
            if sq_keyword not in ('NONE', 'N/A', ''):
                search_query = sq_text

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

        # ── Parse UPDATED_STRUCTURAL_LANDSCAPE (optional — omit preserves existing) ──
        # Strategic Review emits this block ONLY when structural information
        # has accumulated since the last review. Absence of the block means
        # "no change" — preserve the existing landscape, do not wipe.
        updated_landscape = ""
        if 'UPDATED_STRUCTURAL_LANDSCAPE:' in response:
            ls_text = response.split('UPDATED_STRUCTURAL_LANDSCAPE:')[1]
            if 'END_STRUCTURAL_LANDSCAPE' in ls_text:
                ls_text = ls_text.split('END_STRUCTURAL_LANDSCAPE')[0]
            updated_landscape = ls_text.strip()

        # ── Parse UPDATED_CAUSAL_SUBSTRATE (optional — rare refinement) ──
        # Causal Substrate is authored once at orientation (Phase 3). Strategic
        # Review may refine it ONLY when a proxy proves systematically
        # unreliable or a regime assumption is empirically contradicted.
        # Absence of this block (the common case) means "no change" — the
        # original substrate stays. Emitting it is a last-resort correction.
        updated_substrate = ""
        if 'UPDATED_CAUSAL_SUBSTRATE:' in response:
            cs_text = response.split('UPDATED_CAUSAL_SUBSTRATE:')[1]
            if 'END_CAUSAL_SUBSTRATE' in cs_text:
                cs_text = cs_text.split('END_CAUSAL_SUBSTRATE')[0]
            updated_substrate = cs_text.strip()

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
            # ABANDON: clear reference code — new arc, fresh start
            # PIVOT: keep reference — core simulation likely still applies
            if commitment_action == 'ABANDON':
                if self._arc_reference_code:
                    logger.info("Arc reference code cleared (ABANDON)")
                self._arc_reference_code = ""
                self._arc_reference_score = 0
        else:
            # HOLD — maintain or establish commitment
            self.strategic_next_direction = ""
            logger.info("Strategic review: HOLD commitment")

        # ── MID-STREAM LITERATURE SEARCH (when SR requests it) ──
        # Only fire on PIVOT or ABANDON — HOLD means we're deepening a known domain
        # and don't need external context. This naturally limits search frequency
        # since HOLDs are ~60-70% of iterations.
        if search_query and self.search_model:
            if commitment_action == 'HOLD':
                logger.info(f"SR requested search but suppressed during HOLD "
                            f"(search only fires on PIVOT/ABANDON)")
            else:
                logger.info(f"SR requested search ({commitment_action}): {search_query[:80]}")
                print(style.search_status(search_query))
                try:
                    brief_context = self._build_search_context()
                    with style.spinner("Searching and integrating literature"):
                        search_results = self.engine.run_literature_search(
                            search_query, self.search_model,
                            mode='midstream', brief_context=brief_context)
                        if search_results:
                            integrated = self._integrate_search_results(
                                search_results, self.research_model)
                        else:
                            integrated = None
                    if integrated:
                        self.research_model = integrated
                        summary = getattr(self, '_last_search_summary', 'findings integrated')
                        self.search_history.append((iteration, search_query[:80], summary))
                        print(style.search_result(summary))
                        logger.info(f"Mid-stream literature: {summary}")
                    elif search_results:
                        logger.warning("Literature integration returned empty")
                    else:
                        logger.warning("Mid-stream search returned no results")
                except Exception as e:
                    logger.warning(f"Mid-stream literature search failed: {e}")

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

        # ── Splice updated structural landscape into research model ──
        # Emitted only when structural change has accumulated. Absence
        # preserves existing landscape.
        if updated_landscape:
            self._update_structural_landscape(updated_landscape)
            # Write live structural_map.md — lets dashboard surface the
            # landscape as it grows, not only at end-of-run.
            try:
                self._write_live_structural_map()
            except Exception as e:
                logger.debug(f"Live structural_map write failed: {e}")

        # ── Splice updated Causal Substrate (rare refinement path) ──
        # Only fires when SR has explicit evidence that the original
        # substrate is miscalibrated (e.g., a proxy variable is
        # systematically unreliable, a regime assumption is contradicted,
        # an axis previously listed as independent is shown to be subsumed).
        # Absence preserves the original substrate authored at Phase 3.
        # The cheap-agent compacted view is regenerated automatically on the
        # next cheap-agent call via _research_model_for_cheap_agent() — no
        # separate guidance-one-liner update needed.
        if updated_substrate:
            self._update_causal_substrate(updated_substrate)
            logger.info(
                f"Causal Substrate refined by Strategic Review "
                f"({len(updated_substrate)} chars)"
            )

        # ── Log missed opportunities ──
        if missed:
            logger.info(f"Strategic review missed opportunities: {missed[:200]}")

    def _write_live_structural_map(self):
        """Write structural_map.md based on current research_model state.

        Called from strategic review after Structural Landscape updates and
        at end-of-run from _write_briefing_artefacts. Safe to call when
        Structural Landscape is empty (writes a placeholder note).
        """
        landscape = self._extract_model_section('Structural Landscape')
        # Strip the DO-NOT-MODIFY placeholder if that's all we have
        if landscape and '<<< DO NOT MODIFY' in landscape:
            return  # placeholder only, nothing to render yet
        if not landscape:
            return

        map_path = os.path.join(self.engine.output_dir, "structural_map.md")
        with open(map_path, 'w') as f:
            f.write("# Structural Landscape\n\n")
            f.write(
                "*Live view of the investigation's structural terrain — "
                "identifiability, coverage, foreclosed directions, and open "
                "questions. Accumulated during exploration and extended by "
                "strategic review as structural discoveries arrive.*\n\n"
            )
            f.write(landscape)

    def _update_strategic_trajectory(self, new_trajectory):
        """Splice the premium model's trajectory into the research model."""
        self._splice_protected_section('Strategic Trajectory', new_trajectory)

    def _update_structural_landscape(self, new_landscape):
        """Splice new structural-landscape content into the research model.

        Called after orientation (initial seeding) and after each strategic
        review that emitted UPDATED_STRUCTURAL_LANDSCAPE. Uses the same
        save-before/re-splice-after pattern as Strategic Trajectory.
        """
        self._splice_protected_section('Structural Landscape', new_landscape)

    def _update_causal_substrate(self, new_substrate):
        """Splice the Causal Substrate section into the research model.

        Called once at orientation time (Phase 3). Strategic Review may
        refine it rarely via UPDATED_CAUSAL_SUBSTRATE. Same Opus-protected
        splice pattern as Strategic Trajectory and Structural Landscape.

        The substrate lives in the research model so it is visible to
        Opus-reading agents (Strategic Review, reframing probe, perspective
        rotation, briefing generator). Cheap agents read a substituted
        version via _research_model_for_cheap_agent() — the full substrate
        is replaced with the one-liner guidance directive.
        """
        self._splice_protected_section('Causal Substrate', new_substrate)

    def _research_model_for_cheap_agent(self):
        """Return a version of the research model suitable for cheap agents.

        Substitutes the full ## Causal Substrate block with a compacted view
        consisting of the TYPE line, the CANDIDATE MATCHING AXES table, and
        the SELECTION GUIDANCE block (if present). Other substrate sections
        (OUTCOME, REGIMES, ENUMERATED CONFOUNDERS, PROXIES BY ROLE) are
        dropped from the cheap-agent view because they encode reasoning that
        cheap models tend to ignore or pattern-match superficially. The
        full substrate remains available to premium agents (Strategic Review,
        reframing probe, perspective rotation, briefing generator) via
        direct self.research_model access.

        For TYPE=SPARSE or TYPE=DESCRIPTIVE, no axes table exists, so the
        cheap-agent view shows just the TYPE line followed by the RATIONALE
        (one or two sentences). Cheap agents need to know the seed isn't
        causal; they don't need the full enumeration of failed candidates.

        If no substrate section exists in the research model, the model is
        returned unchanged.
        """
        if not self.research_model:
            return self.research_model

        section_header = '## Causal Substrate'
        if section_header not in self.research_model:
            return self.research_model

        # Locate the substrate body
        before, after_section = self.research_model.split(section_header, 1)
        next_header_idx = after_section.find('\n## ')
        if next_header_idx > 0:
            substrate_body = after_section[:next_header_idx]
            after = after_section[next_header_idx:]
        else:
            end_idx = after_section.find('END_MODEL')
            if end_idx > 0:
                substrate_body = after_section[:end_idx]
                after = after_section[end_idx:]
            else:
                substrate_body = after_section
                after = ""

        # Substrate is still a placeholder — keep as-is so the RI's "do not
        # modify" placeholder remains visible to the cheap model.
        if '<<< DO NOT MODIFY' in substrate_body:
            return self.research_model

        # Compact the substrate body. Strategy: extract the TYPE line, then
        # extract specific named sections by header. Sections that don't
        # exist in this substrate (e.g., SELECTION GUIDANCE in older
        # substrates) are silently skipped.
        compact = self._compact_substrate(substrate_body)

        if not compact:
            # Compaction failed — fall back to dropping the section entirely
            # rather than show a malformed view.
            if before:
                return f"{before.rstrip()}{after.lstrip(chr(10))}"
            return after.lstrip('\n')

        replacement = f"{section_header}\n{compact}\n"
        return f"{before}{replacement}{after}"

    @staticmethod
    def _compact_substrate(substrate_body):
        """Extract the cheap-agent-relevant slices from a Causal Substrate body.

        Returns: TYPE line + (CANDIDATE MATCHING AXES + SELECTION GUIDANCE
        for FULL) OR (RATIONALE for SPARSE/DESCRIPTIVE).

        Empty string if the body is structurally malformed.
        """
        lines = substrate_body.strip().split('\n')
        if not lines:
            return ""

        # Recognised section headers in the substrate template. A header is
        # an at-column-0 line that starts with one of these tokens (allowing
        # trailing qualifiers in parens or after a colon).
        SECTION_TOKENS = (
            'TYPE:',
            'OUTCOME',
            'REGIMES',
            'ENUMERATED CONFOUNDERS',
            'IDENTIFIABLE CONFOUNDERS',
            'CANDIDATE MATCHING AXES',
            'SELECTION GUIDANCE',
            'PROXIES BY ROLE',
            'RATIONALE',
            'NOTEWORTHY VARIABLES',
        )

        def _is_section_header(line):
            """Header: not indented, starts with one of the recognised tokens."""
            if not line or line[0] in (' ', '\t', '-', '|'):
                return False
            upper = line.upper().lstrip()
            return any(upper.startswith(tok) for tok in SECTION_TOKENS)

        def _matches_token(line, token):
            return line.upper().lstrip().startswith(token.upper())

        # Find TYPE line
        type_line = ""
        for line in lines:
            if line.strip().startswith('TYPE:'):
                type_line = line.strip()
                break
        if not type_line:
            return ""
        type_value = type_line.split(':', 1)[1].strip().upper()

        # Decide which sections to extract
        if type_value == 'FULL':
            wanted = ['CANDIDATE MATCHING AXES', 'SELECTION GUIDANCE']
        else:
            # SPARSE or DESCRIPTIVE
            wanted = ['RATIONALE']

        # Walk the body once, collecting any wanted sections
        body_lines = substrate_body.split('\n')
        sections = {}  # token -> list of lines
        current_token = None
        for line in body_lines:
            if _is_section_header(line):
                # Determine which token this header is
                matched = None
                for tok in SECTION_TOKENS:
                    if _matches_token(line, tok):
                        matched = tok
                        break
                if matched in wanted:
                    current_token = matched
                    sections.setdefault(current_token, []).append(line)
                else:
                    current_token = None
            elif current_token is not None:
                sections[current_token].append(line)

        # Trim trailing empty lines from each section
        for tok, lines_list in sections.items():
            while lines_list and not lines_list[-1].strip():
                lines_list.pop()

        # Assemble in the order specified by `wanted`
        parts = [type_line]
        for tok in wanted:
            if tok in sections and sections[tok]:
                parts.append('')
                parts.extend(sections[tok])

        return '\n'.join(parts)

    def _splice_protected_section(self, section_name, new_content):
        """Generic splicer for Opus-protected sections. Shared by both
        Strategic Trajectory and Structural Landscape. Robust to:
        - section missing → append at end
        - duplicate headers from RI mangling → collapse before splicing
        - trailing END_MODEL marker → preserve
        """
        import re as _re
        section_header = f'## {section_name}'

        # Normalise: collapse any consecutive duplicate headers the RI may have emitted
        self.research_model = _re.sub(
            rf'(## {_re.escape(section_name)}[ \t]*\n)+', section_header + '\n',
            self.research_model
        )

        if section_header in self.research_model:
            before = self.research_model.split(section_header)[0]
            after_section = self.research_model.split(section_header)[1]
            # Find next ## header at start of line
            next_header_idx = after_section.find('\n## ')
            if next_header_idx > 0:
                after = after_section[next_header_idx:]
            else:
                # No subsequent section — check for END_MODEL marker
                end_idx = after_section.find('END_MODEL')
                after = after_section[end_idx:] if end_idx > 0 else ""
            self.research_model = f"{before}{section_header}\n{new_content}\n{after}"
        else:
            # Section doesn't exist yet — append at end
            self.research_model += f"\n\n{section_header}\n{new_content}"

    # ══════════════════════════════════════════════
    # LITERATURE SEARCH HELPERS
    # ══════════════════════════════════════════════

    def _integrate_search_results(self, search_results, current_model):
        """Integrate search results into research model via premium model + code splice.
        
        The premium model reasons about what matters and sets STATUS.
        Code handles the structural insertion — no model rewrites the full document.
        Returns the updated research model, or empty string on failure.
        """
        # Extract existing [PUBLISHED] entries for continuity
        existing_published = []
        for line in current_model.split('\n'):
            if '[PUBLISHED]' in line:
                existing_published.append(line.strip())

        # Extract simulation findings (read-only context for STATUS assessment)
        sim_findings = self._extract_model_section('Established Findings')
        # Filter out [PUBLISHED] lines — only simulation findings for context
        if sim_findings:
            sim_lines = [l for l in sim_findings.split('\n')
                         if l.strip() and '[PUBLISHED]' not in l]
            sim_context = '\n'.join(sim_lines)
        else:
            sim_context = "(No simulation findings yet)"

        arc_direction = self.current_arc_direction or self.seed_question

        prompt = self.engine.prompts.literature_integration.format(
            search_results=search_results,
            existing_published='\n'.join(existing_published) if existing_published else "(None yet)",
            sim_context=sim_context,
            arc_direction=arc_direction[:200],
        )
        messages = [{"role": "user", "content": prompt}]
        model = self.premium_model or self.search_model

        try:
            response = self.engine.llm_client.call(
                messages=messages,
                model=model,
                max_tokens=4000,
                temperature=0,
                agent="Literature Integration",
            )
            if not response or not response.strip():
                return ""

            # Extract [PUBLISHED] entries from response
            new_published = []
            for line in response.split('\n'):
                stripped = line.strip()
                if '[PUBLISHED]' in stripped:
                    # Ensure bullet format
                    if not stripped.startswith('- '):
                        stripped = f'- {stripped}'
                    new_published.append(stripped)

            # Extract SEARCH_SUMMARY for logging
            summary = self._extract_search_summary(response)

            # Extract CALIBRATION assessment (Mitigation 4: literature
            # contradiction is a one-way valve unless we capture this).
            # Stored on explorer state for the next Strategic Review's Task 5.
            calibration = self._extract_calibration(response)
            if calibration:
                self._last_search_calibration = calibration
                logger.info(f"Literature calibration: {calibration[:120]}")

            if not new_published:
                logger.warning("Integration returned no [PUBLISHED] entries")
                return ""

            # ── UPDATE PROTECTED SNAPSHOT ──
            # Integration is the ONLY agent that can modify [PUBLISHED] entries.
            # The RI reads them (for reference) but splice enforces immutability.
            self._published_entries = new_published

            # Splice: remove old [PUBLISHED] lines from Established Findings,
            # insert new ones at the end of the section
            if '## Established Findings' not in current_model:
                # Pre-loop or empty model — create minimal section
                published_block = '\n'.join(new_published)
                if current_model and not current_model.startswith('('):
                    # Append to existing model
                    spliced = f"{current_model}\n\n## Established Findings\n\n{published_block}\n"
                else:
                    # Empty model — just the published entries
                    spliced = f"## Established Findings\n\n{published_block}\n"
                self._last_search_summary = summary
                return spliced

            lines = current_model.split('\n')
            result_lines = []
            in_ef = False
            inserted = False

            for i, line in enumerate(lines):
                if '## Established Findings' in line:
                    in_ef = True
                    result_lines.append(line)
                    continue
                if line.startswith('## ') and in_ef:
                    # Insert new [PUBLISHED] entries before next section
                    for pub in new_published:
                        result_lines.append(pub)
                    result_lines.append('')  # blank line before next section
                    in_ef = False
                    inserted = True
                    result_lines.append(line)
                    continue
                if in_ef and '[PUBLISHED]' in line:
                    continue  # drop old [PUBLISHED] lines
                if in_ef and line.strip().startswith('STATUS:'):
                    continue  # drop orphaned STATUS lines
                if in_ef and line.strip().startswith('- STATUS:'):
                    continue  # drop bullet STATUS lines
                result_lines.append(line)

            # If Established Findings was the last section (no ## after it)
            if in_ef and not inserted:
                for pub in new_published:
                    result_lines.append(pub)

            # Store summary for extraction
            self._last_search_summary = summary
            return '\n'.join(result_lines)

        except Exception as e:
            logger.warning(f"Literature integration failed: {e}")
            return ""

    def _build_search_context(self):
        """Build brief context from research model for mid-stream search."""
        # Extract last few established findings for context
        findings = self._extract_model_section('Established Findings')
        if findings:
            lines = [l.strip() for l in findings.split('\n') if l.strip().startswith('-')]
            return '\n'.join(lines[-5:]) if lines else self.seed_question
        return self.seed_question

    @staticmethod
    def _extract_search_summary(integrated_model):
        """Extract SEARCH_SUMMARY line from the integrated model response."""
        for line in integrated_model.split('\n'):
            if 'SEARCH_SUMMARY:' in line:
                return line.split('SEARCH_SUMMARY:')[1].strip()
        return "findings integrated"

    @staticmethod
    def _extract_calibration(integrated_model):
        """Extract CALIBRATION line from the integrated model response.

        Returns the full calibration text including label and reason
        (e.g., "SUSPECT — 9/13 contradicted with no concrete differentiator")
        or empty string if no CALIBRATION line is present.
        """
        for line in integrated_model.split('\n'):
            stripped = line.strip()
            if stripped.upper().startswith('CALIBRATION:'):
                return stripped.split(':', 1)[1].strip()
        return ""

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

        # Why it matters: recent established findings first, then the Structural
        # Landscape open questions, then fall back to the seed question.
        why_it_matters = ""
        # Recent established findings give context for positive-finding probes
        findings = self._extract_model_section('Established Findings')
        if findings:
            # Last 3 findings are most relevant to current arc
            finding_lines = [l.strip() for l in findings.split('\n') if l.strip().startswith('-')]
            if finding_lines:
                why_it_matters = "Recent findings:\n" + "\n".join(finding_lines[-3:])
        if not why_it_matters or len(why_it_matters) < 20:
            # Fallback: the Structural Landscape's Open Questions section
            # captures what the investigation knows it can't resolve.
            landscape = self._extract_model_section('Structural Landscape')
            if landscape and '<<< DO NOT MODIFY' not in landscape and len(landscape) > 30:
                why_it_matters = landscape[:600]
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

        # Opus agent: receives full Causal Substrate block for Rung-3 reasoning
        # about whether a result stripped signal between the substrate's regimes
        # by over-normalizing on a matching axis.
        substrate = self._extract_model_section('Causal Substrate') or "(no substrate)"

        prompt = self.engine.prompts.reframing_probe.format(
            seed_question=self.seed_question,
            arc_summary=arc_summary,
            why_it_matters=why_it_matters,
            full_results=full_results,
            causal_substrate=substrate,
            probe_suspicion=self._last_probe_suspicion if self._last_probe_suspicion else "NONE",
        )

        if self._last_probe_suspicion:
            logger.info(f"Adversarial probe firing: {self._last_probe_suspicion[:120]}")

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

        # Filter nodes to this arc's iteration range. iteration_added is
        # 1-based (matches arc_history's start_iter and the human-readable
        # iteration count); `iteration` here is the 0-based loop index, so
        # the upper bound is iteration+1 for an inclusive 1-based comparison.
        arc_nodes = [
            node for node in sorted(self.insight_tree.values(), key=lambda n: n['chain_id'])
            if node.get('iteration_added', 0) >= arc_start
            and node.get('iteration_added', 0) <= iteration + 1
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

        # Opus agent: receives full Causal Substrate for Rung-3 reasoning
        # about which causal roles change across perspectives (matching axis
        # vs nuisance vs outcome vs anchor).
        substrate = self._extract_model_section('Causal Substrate') or "(no substrate)"

        prompt = self.engine.prompts.perspective_rotation.format(
            seed_question=self.seed_question,
            arc_name=arc_name,
            arc_methods=arc_methods,
            arc_findings=arc_findings,
            available_columns=columns,
            previously_selected=prior_perspectives_text,
            causal_substrate=substrate,
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
        research_model = self.research_model if self.research_model else "(Research model empty — no orientation Landscape available)"

        prompt = self.engine.prompts.seed_decomposition.format(
            seed_question=seed_question,
            data_profile=profile,
            research_model=research_model,
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
    # PARALLEL QUESTION WORKER
    # ══════════════════════════════════════════════

    def _run_question_buffered(self, question, chain_id, image, q_idx, total):
        """Worker function for parallel question processing.

        Runs in a ThreadPoolExecutor worker. Sets up per-thread output
        buffering before calling _process_question, so all terminal writes
        (the [Code Generator] header line, the result box, the file_ref
        line, error messages) accumulate in a thread-local list instead of
        racing with other workers on stdout. The accumulated text is
        returned to the caller, which flushes buffers in q_idx order after
        all workers complete.

        Side effects on shared state:
          - message_manager.qa_pairs / .full_results_store / .all_questions:
            via _process_question -> append_qa_pair (lock-protected)
          - message_manager.error_patterns: via record_error_pattern
            (lock-protected, only when a retry succeeded)
          - llm_client.cost_tracker / .run_logger: via the LLM calls
            (lock-protected in Step 1)

        Catches exceptions from _process_question internally so the caller's
        future.result() doesn't raise — failed questions return a
        well-formed result dict with results=None and error populated.

        Returns:
            (result_dict, buffered_text, error_occurred) where
            - result_dict is the dict returned by _process_question (or a
              skeleton on exception)
            - buffered_text is the captured stdout of this worker as a
              single string, ready to be written to sys.stdout in order
            - error_occurred is True if _process_question raised
        """
        om = self.engine.output_manager
        om.begin_buffer()

        error_occurred = False
        result = None
        try:
            # Question header lives inside the buffer so the whole block
            # (header → body → file ref) flushes atomically. If we printed
            # the header outside the buffer (i.e. before submit), all
            # headers would appear up-front and then be separated from
            # their bodies by 30+ lines of other questions' output.
            om.print_wrapper("")
            om.print_wrapper(
                style.question_display(q_idx + 1, total, "", question)
            )

            try:
                result = self.engine._process_question(
                    question,
                    chain_id=chain_id,
                    image=image,
                )
            except Exception as e:
                error_occurred = True
                logger.warning(
                    f"Execution error in worker (q_idx={q_idx}): "
                    f"{e}\n{traceback.format_exc()}"
                )
                om.display_error(
                    f"Execution error: {e}", chain_id=chain_id
                )
                result = {
                    'results': None,
                    'code': None,
                    'last_plan': None,
                    'plots': [],
                    'error': str(e),
                }
        finally:
            buffered_text = om.end_buffer()

        return result, buffered_text, error_occurred

    # ══════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════

    def run(self, seed_question, initial_image=None, max_iterations=5, num_parallel_solutions=2,
            interactive=False, resumed_state=None, orientation=True, auto_stop=False,
            backfill_embeddings=True):
        """Run the autonomous exploration loop.

        Args:
            auto_stop: If True, the system may terminate before max_iterations
                when the strategic review determines the investigation is complete.
                Default False — the full iteration budget is always used.
            backfill_embeddings: If True (default), embed any pre-v10 winning
                nodes lacking embeddings on --continue, so geometry has full
                coverage from iteration 1.
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

                # Backfill embeddings for historical winning nodes (pre-v10 or
                # earlier resumes that ran with --no-embeddings). Skipped when
                # backfill_embeddings is False or embeddings are disabled.
                if backfill_embeddings:
                    try:
                        self._backfill_embeddings()
                    except Exception as e:
                        logger.warning(f"Embedding backfill skipped: {e}")
            else:
                current_questions = [seed_question]
                last_solution_chain = None
                last_follow_up_angle = None

                self._reset_state(seed_question)

                start_iteration = 0

            iteration = start_iteration
            self._current_iteration = start_iteration

            # --- Header ---
            if self.engine.df is not None:
                df_shape = f"{self.engine.df.shape[0]:,} rows × {self.engine.df.shape[1]} cols"
            else:
                df_shape = "computation mode (no dataset)"
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
                    # Extract the STRUCTURAL_LANDSCAPE block and inject it
                    # into the research model IMMEDIATELY so that seed
                    # decomposition and iteration 0's agents see it. If we
                    # deferred injection to after the first RI call (the old
                    # behaviour), seed decomposition and iteration 0 would
                    # run against an empty Landscape — losing awareness of
                    # foreclosed directions at the exact moment the initial
                    # strategic trajectory is being chosen.
                    initial_landscape = self._extract_orientation_landscape(self.data_profile)
                    if initial_landscape:
                        logger.info(
                            f"Initial structural landscape: "
                            f"{len(initial_landscape)} chars — seeding research model"
                        )
                        # _update_structural_landscape appends to empty research
                        # model; subsequent RI calls' re-splice-after pattern
                        # (around line 2007) will preserve it.
                        self._update_structural_landscape(initial_landscape)
                    # NOTE: data_profile still contains the Landscape block at
                    # this point. It is deliberately left in place so that
                    # seed decomposition and iteration 0 agents (Code Gen,
                    # Question Gen, Evaluator, RI) see the initial Landscape
                    # — the research model is still empty or unreliable until
                    # the first RI call produces a proper skeleton. The
                    # Landscape is stripped from data_profile at the end of
                    # iteration 0 (once both Landscape and Trajectory are
                    # safely spliced into the research model), so that from
                    # iteration 1 onwards agents see a single source of
                    # truth: research model owns the live Landscape (updated
                    # by Strategic Review); data_profile owns static context.

            # ── PHASE 3: CAUSAL SUBSTRATE (fresh runs only) ──
            # Runs regardless of whether orientation ran or whether a
            # dataset is loaded. The substrate's TYPE field (FULL / SPARSE
            # / DESCRIPTIVE) honestly reports whether the seed supports
            # Rung-3 matching-axis design reasoning. The structured
            # substrate is spliced into the research model as
            # ## Causal Substrate; premium agents read the full block,
            # cheap agents read a compacted view (TYPE + CANDIDATE MATCHING
            # AXES + SELECTION GUIDANCE for FULL; TYPE + RATIONALE for
            # SPARSE/DESCRIPTIVE) extracted at every cheap-agent call by
            # _research_model_for_cheap_agent().
            if not resumed_state and not self.kill_signal:
                logger.info("Running Phase 3: Causal Substrate")
                try:
                    substrate_text = self.engine.run_causal_substrate(
                        seed_question,
                        profile=self.data_profile,
                        model_override=self.premium_model,
                    )
                except Exception as e:
                    logger.warning(f"Causal substrate failed: {e}")
                    substrate_text = ""

                if substrate_text:
                    self._update_causal_substrate(substrate_text)
                    type_line = substrate_text.split('\n', 2)[0].strip()
                    logger.info(
                        f"Causal substrate seeded ({len(substrate_text)} chars, "
                        f"{type_line}); cheap agents read compacted view"
                    )
                else:
                    logger.info(
                        "Causal substrate produced no output — downstream "
                        "will treat as SPARSE (no matching-axis guidance)"
                    )

            # ── PRE-LOOP LITERATURE SEARCH (computation mode, fresh runs only) ──
            if (self.search_model and not resumed_state and not self.kill_signal
                    and self.engine.df is None):
                logger.info("Running pre-loop literature search")
                print(style.search_status(seed_question[:60]))
                try:
                    with style.spinner("Searching published literature"):
                        search_results = self.engine.run_literature_search(
                            seed_question, self.search_model, mode='preloop')
                    if search_results:
                        with style.spinner("Integrating literature"):
                            integrated = self._integrate_search_results(
                                search_results, self.research_model or "(Empty — first iteration)")
                        if integrated:
                            self.research_model = integrated
                            summary = getattr(self, '_last_search_summary', 'findings integrated')
                            self.search_history.append((0, seed_question[:80], summary))
                            print(style.search_result(summary))
                            logger.info(f"Pre-loop literature: {summary}")
                        else:
                            logger.warning("Literature integration returned empty")
                    else:
                        logger.warning("Pre-loop search returned no results")
                except Exception as e:
                    logger.warning(f"Pre-loop literature search failed: {e}")

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
                # Step 2 of the parallelisation refactor: the per-question
                # loop becomes a ThreadPoolExecutor when there's more than
                # one question to process. Each worker runs under per-thread
                # output buffering (set up by _run_question_buffered) so
                # concurrent terminal writes don't interleave. After all
                # workers complete, the buffered blocks are flushed to stdout
                # in question-index order — visually identical to the
                # sequential layout, just with one combined wait instead of
                # N sequential waits.
                #
                # Iteration 0's single seed question keeps the inline path:
                # no pool, no buffering, live spinners. The pool only kicks
                # in from iteration 1 onwards where 2-3 questions run in
                # parallel.
                #
                # _arc_reference_code is per-iteration (set above by strategic
                # review on score-8+ wins) and read-only during the question
                # phase, so we set it once before workers fire.
                n_questions = len(questions_to_process)
                solutions_data = [None] * n_questions
                self.engine._arc_reference_code = self._arc_reference_code

                # Pre-allocate chain_ids: one timestamp per iteration's batch,
                # offset by q_idx. Ensures uniqueness across parallel workers
                # within an iteration.
                base_ts = int(time.time())
                chain_ids = [base_ts + q_idx for q_idx in range(n_questions)]

                if n_questions == 1:
                    # ── Single-question (seed) path — inline, live spinners ──
                    question = questions_to_process[0]
                    print()
                    print(style.question_display(1, 1, "", question))

                    error_occurred = False
                    try:
                        result = self.engine._process_question(
                            question,
                            chain_id=chain_ids[0],
                            image=current_image,
                        )
                    except Exception as e:
                        error_occurred = True
                        logger.warning(f"Execution error: {e}")
                        self.engine.output_manager.display_error(
                            f"Execution error: {e}", chain_id=chain_ids[0])
                        result = {
                            'results': None, 'code': None, 'last_plan': None,
                            'plots': [], 'error': str(e),
                        }

                    solutions_data[0] = {
                        'question': question,
                        'results': result.get('results'),
                        'code': result.get('code'),
                        'text_answer': result.get('last_plan'),
                        'chain_id': chain_ids[0],
                        'error_occurred': error_occurred,
                    }
                else:
                    # ── Multi-question path — parallel workers with buffered output ──
                    buffers = [None] * n_questions
                    label = (f"Processing {n_questions} questions in parallel "
                             f"({self.engine.models.code_model.split(':', 1)[-1][:24]})")

                    with concurrent.futures.ThreadPoolExecutor(
                            max_workers=n_questions,
                            thread_name_prefix='qworker',
                    ) as ex:
                        futures = {}
                        for q_idx in range(n_questions):
                            if self.kill_signal:
                                break
                            futures[q_idx] = ex.submit(
                                self._run_question_buffered,
                                questions_to_process[q_idx],
                                chain_ids[q_idx],
                                # Only the first question gets the image —
                                # matches the pre-refactor behaviour.
                                current_image if q_idx == 0 else None,
                                q_idx,
                                n_questions,
                            )

                        # One combined spinner spans the entire parallel
                        # wait. We collect results in submission order
                        # (sorted q_idx), so .result() blocks naturally
                        # serve as a barrier — by the time we exit this
                        # loop, all workers are done.
                        with style.spinner(label):
                            for q_idx in sorted(futures):
                                try:
                                    result, buffered_text, worker_error = \
                                        futures[q_idx].result()
                                except Exception as e:
                                    # _run_question_buffered catches its own
                                    # exceptions, so a raise here means the
                                    # wrapper itself failed. Degenerate.
                                    logger.warning(
                                        f"Worker for q_idx={q_idx} raised "
                                        f"unhandled: {e}\n{traceback.format_exc()}"
                                    )
                                    result = {
                                        'results': None, 'code': None,
                                        'last_plan': None, 'plots': [],
                                        'error': str(e),
                                    }
                                    buffered_text = ''
                                    worker_error = True

                                buffers[q_idx] = buffered_text
                                solutions_data[q_idx] = {
                                    'question': questions_to_process[q_idx],
                                    'results': result.get('results'),
                                    'code': result.get('code'),
                                    'text_answer': result.get('last_plan'),
                                    'chain_id': chain_ids[q_idx],
                                    'error_occurred': worker_error,
                                }

                    # Outer spinner has exited (cleared its \r-overwritten
                    # line and emitted its checkmark line). Now safe to
                    # flush each question's buffered block to stdout in
                    # order. This is where the user actually sees the
                    # per-question output for the iteration.
                    for buf in buffers:
                        if buf:
                            sys.stdout.write(buf)
                    sys.stdout.flush()

                # Drop any None slots (kill_signal raced mid-submit on the
                # parallel path) so downstream code sees a clean list.
                solutions_data = [s for s in solutions_data if s is not None]

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

                # ── UPDATE ARC REFERENCE CODE ──
                # Save the winning code from score-8+ analyses as the implementation
                # reference for subsequent iterations in this arc. Ensures simulation
                # consistency (generation order, data structures, mechanisms) across
                # HOLD iterations without sharing code literally.
                winning_score = scores[selected_index]
                winning_code = winning_solution.get('code', '')
                if winning_code and winning_score >= 8 and winning_score > self._arc_reference_score:
                    self._arc_reference_code = winning_code
                    self._arc_reference_score = winning_score
                    logger.info(f"Arc reference code updated (score {winning_score}, "
                                f"{len(winning_code)} chars)")

                # ── INTERPRET & UPDATE MODEL ──
                # Save Opus-protected sections before update — cheap model must
                # not corrupt them. All three are spliced back after the update.
                _saved_trajectory = self._extract_model_section('Strategic Trajectory')
                if _saved_trajectory and '<<< DO NOT MODIFY' in _saved_trajectory:
                    _saved_trajectory = ""  # placeholder, not real content yet
                _saved_landscape = self._extract_model_section('Structural Landscape')
                if _saved_landscape and '<<< DO NOT MODIFY' in _saved_landscape:
                    _saved_landscape = ""  # placeholder, not real content yet
                _saved_substrate = self._extract_model_section('Causal Substrate')
                if _saved_substrate and '<<< DO NOT MODIFY' in _saved_substrate:
                    _saved_substrate = ""  # placeholder, not real content yet

                with style.spinner("Updating research model"):
                    (updated_model, model_impact, contradiction, arc_exhausted,
                     result_digest, method_used, tested_estimand) = \
                        self._interpret_and_update_model(
                            winning_solution,
                            scores[selected_index],
                        )

                if new_node_id in self.insight_tree and result_digest:
                    self.insight_tree[new_node_id]['result_digest'] = result_digest
                if new_node_id in self.insight_tree and method_used:
                    self.insight_tree[new_node_id]['method_used'] = method_used
                if new_node_id in self.insight_tree and tested_estimand:
                    self.insight_tree[new_node_id]['tested_estimand'] = tested_estimand

                # Embed the winning node for geometry observability.
                # Non-fatal: failures leave embedding=None and show as a
                # coverage gap in the dashboard's Run Geometry panel.
                self._embed_node(new_node_id)

                self.research_model = updated_model

                # ── PROTECT OPUS-OWNED SECTIONS ──
                # The RI (cheap model) regenerates the full research model. It
                # may mangle protected sections despite the DO-NOT-MODIFY
                # instructions. Re-splice all three sections we saved before
                # the update. Order matters: landscape first, then substrate,
                # then trajectory — this keeps section-boundary resolution
                # stable across successive splices (each splice locates "next
                # ## header" by searching forward from its section header).
                if _saved_landscape:
                    self._update_structural_landscape(_saved_landscape)
                if _saved_substrate:
                    self._update_causal_substrate(_saved_substrate)
                if _saved_trajectory:
                    self._update_strategic_trajectory(_saved_trajectory)

                # On iteration 0, splice the initial trajectory (produced by
                # seed decomposition) into the research model. The initial
                # Structural Landscape was already seeded into the research
                # model at orientation time — it is preserved across the RI
                # call by the _saved_landscape re-splice above.
                if self._initial_trajectory:
                    self._update_strategic_trajectory(self._initial_trajectory)
                    self._initial_trajectory = ""  # consumed

                # End of iteration 0: the research model now carries both the
                # initial Structural Landscape and the initial Strategic
                # Trajectory. It is now safe to strip the Landscape block
                # from data_profile — from iteration 1 onwards the research
                # model is the single source of truth for identifiability,
                # coverage, foreclosed directions, and open questions.
                # Keeping the Landscape in data_profile would pin a frozen
                # copy into every agent's context, competing with the live
                # copy that Strategic Review updates.
                if iteration == 0 and self.data_profile \
                        and '###STRUCTURAL_LANDSCAPE_START###' in self.data_profile:
                    stripped_profile = self._strip_landscape_from_profile(self.data_profile)
                    logger.info(
                        f"End of iteration 0: stripped Landscape block from "
                        f"data_profile ({len(self.data_profile)} → "
                        f"{len(stripped_profile)} chars)"
                    )
                    self.data_profile = stripped_profile
                    self.engine.data_profile = stripped_profile

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
            with style.spinner("Generating briefing"):
                synthesis_text = self._generate_synthesis(seed_question)

            if not synthesis_text:
                print(f"  {style.YELLOW}✗ Briefing generation failed{style.RESET}")
                synthesis_text = ""
            else:
                # Write companion artefacts (findings_index.md,
                # structural_map.md) immediately — before chart generation —
                # so they are available even if chart generation fails.
                self._write_briefing_artefacts(synthesis_text)

            # Generate charts for key findings (premium model).
            # The skip-keywords list in _generate_synthesis_charts has been
            # updated for the new briefing section structure; see that method
            # for details on which sections are chart-eligible.
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
                briefing_text=synthesis_text,
                insight_tree=self.insight_tree,
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
        """Generate final handoff briefing.

        Produces the narrative-free, operational briefing via the premium
        model. The companion artefacts (findings_index.md, structural_map.md)
        are written separately by _write_briefing_artefacts.
        """
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
        prompt = self.engine.prompts.briefing_generation.format(
            today_date=today,
            synthesis_context=synthesis_context,
            task=f"Produce a handoff briefing for the investigation seeded by: {seed_question}",
        )

        messages = [{"role": "user", "content": prompt}]

        # Briefing is the payoff of the entire run — retry indefinitely
        # with exponential backoff (10->20->40->80s cap) until a response arrives.
        attempt = 0
        backoff = 10
        while True:
            attempt += 1
            response = self._call_agent_with_retry(
                messages, 'Briefing Generator', model_override=self.premium_model)
            if response and response.strip():
                if attempt > 1:
                    logger.info(f"Briefing succeeded on attempt {attempt}")
                return response

            logger.warning(f"Briefing attempt {attempt} failed, retrying in {backoff}s...")
            print(f"  {style.YELLOW}⟳ Briefing attempt {attempt} failed — retrying in {backoff}s...{style.RESET}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 80)

    def _generate_synthesis_charts(self, synthesis_text):
        """Generate one publication-quality chart per finding / key section.

        Two paths through the briefing structure:

          - §2 Findings (H2): parsed into H3 sub-sections. Each H3 finding
            becomes a chart candidate, filtered by STATUS tag (CONTRADICTED
            findings are skipped — visualising an overturned claim misleads).
            Charts are inserted BEFORE the `**CONFOUND-STATUS:**` line of
            each finding, visually attaching to the claim they support.

          - Other H2 sections: filtered via `skip_keywords`. Under the new
            briefing structure, all non-§2 H2 sections (Scope, Landscape,
            Foreclosed, Open, Entry Points, Methodological Notes) are
            covered by skip keywords, so they produce no charts. Legacy
            synthesis outputs still work through this path.

        Returns updated briefing text with embedded chart references.
        """
        if not synthesis_text or not self.premium_model:
            return synthesis_text

        # Sections that should NOT get charts at H2 level.
        # Original keywords kept for backward compatibility with legacy
        # synthesis. 'findings' is intentionally NOT in this list: the §2
        # Findings section gets H3-per-finding treatment below, not H2 skip.
        skip_keywords = [
            # legacy synthesis
            'rejected', 'tested and rejected', 'caveats', 'caveat',
            'limitation', 'methodolog', 'conclusion', 'next step',
            'recommended', 'open question', 'what is stable',
            'stable, what', 'executive summary', 'cross-cutting',
            # briefing structure (all non-§2 H2 sections)
            'investigation scope', 'structural landscape',
            'foreclosed', 'suggested entry', 'entry points',
        ]

        # STATUS tags that should NOT get charts at H3 level.
        # CONTRADICTED findings are text-best: visualising the overturned
        # claim misleads; the corrected estimate is the thing to cite.
        # BLOCKED findings are identifiability outcomes (data cannot resolve
        # the seed-relevant question) — they have no key numbers to chart.
        skip_statuses = {'CONTRADICTED', 'BLOCKED'}

        # ── Parse briefing into H2 sections ──
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

        # ── Build list of chart candidates ──
        # Each candidate is (level, title, text) where level is 'h2' or 'h3'.
        # This distinction is needed at insertion time (different anchor
        # patterns and insertion rules per level).

        def is_findings_section(title):
            """Detect §2 Findings — enforced by the briefing prompt."""
            return '§2' in title and 'finding' in title.lower()

        def parse_h3_subsections(section_text):
            """Split an H2 section's text into (h3_title, h3_text) pairs."""
            subs = []
            curr_title = None
            curr_lines = []
            for line in section_text.split('\n'):
                if line.startswith('### '):
                    if curr_title is not None:
                        subs.append((curr_title, '\n'.join(curr_lines)))
                    curr_title = line[4:].strip()
                    curr_lines = [line]
                elif curr_title is not None:
                    curr_lines.append(line)
            if curr_title is not None:
                subs.append((curr_title, '\n'.join(curr_lines)))
            return subs

        def extract_status(finding_text):
            """Read STATUS: TAG from **STATUS: TAG** line."""
            m = re.search(r'\*\*STATUS:\s*([A-Z][A-Z_-]+)\*\*', finding_text)
            return m.group(1) if m else None

        chart_sections = []  # (level, title, text)
        for title, text in sections:
            if is_findings_section(title):
                # §2: iterate H3 findings, filter by STATUS
                for h3_title, h3_text in parse_h3_subsections(text):
                    status = extract_status(h3_text)
                    if status in skip_statuses:
                        logger.info(
                            f"Synthesis chart: skipping '{h3_title[:50]}' "
                            f"(STATUS={status})")
                        continue
                    if len(h3_text) < 200:
                        continue
                    chart_sections.append(('h3', h3_title, h3_text))
            else:
                # Non-§2 H2: apply skip_keywords
                if any(kw in title.lower() for kw in skip_keywords):
                    continue
                if len(text) < 200:
                    continue
                chart_sections.append(('h2', title, text))

        if not chart_sections:
            return synthesis_text

        # ── Generate charts ──
        charts_dir = os.path.join(self.engine.output_dir, "synthesis_charts")
        os.makedirs(charts_dir, exist_ok=True)

        schema = self.engine._get_df_schema()
        chart_map = {}  # (level, title) -> relative image path

        for idx, (level, title, section_text) in enumerate(chart_sections):
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:40]
            chart_filename = f"{idx+1:02d}_{slug}.png"
            chart_path = os.path.join(charts_dir, chart_filename)

            cited_ids = re.findall(r'\[\[(\d+)\]\]', section_text)
            original_code = self._get_analysis_code(cited_ids)

            if not original_code:
                logger.info(
                    f"Synthesis chart: no source code found for '{title[:50]}'")
                continue

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
                    # max_tokens=14000 keeps Opus chart generation under the
                    # Anthropic SDK's 10-minute streaming threshold while
                    # leaving ample headroom for a single chart's code
                    # (typically 60–150 lines). Above ~21K, the SDK refuses
                    # non-streaming submissions for slow models.
                    code, llm_response = self.engine._call_llm_for_code(
                        messages, self.premium_model, agent="Synthesis Chart",
                        max_tokens=14000)
                    if not code:
                        logger.info(f"Synthesis chart: no code for '{title[:50]}'")
                        continue

                    results, error, plots = self.engine.executor.execute(
                        code, self.engine.df, charts_dir)

                    retries = 0
                    while error and retries < 2:
                        retries += 1
                        logger.info(
                            f"Synthesis chart retry {retries} for '{title[:50]}': "
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
                        logger.info(
                            f"Synthesis chart failed for '{title[:50]}' after "
                            f"{retries + 1} attempts")
                        continue

                    if plots:
                        import shutil
                        shutil.move(plots[0], chart_path)
                        chart_map[(level, title)] = f"synthesis_charts/{chart_filename}"
                        logger.info(f"Synthesis chart saved: {chart_filename}")

            except Exception as e:
                logger.info(f"Synthesis chart failed for '{title[:50]}': {e}")
                continue

        if not chart_map:
            return synthesis_text

        # ── Embed chart references into briefing text ──
        # H2 charts: insert after the section's first paragraph (legacy).
        # H3 charts: insert BEFORE the **CONFOUND-STATUS:** line, so the
        # chart visually attaches to the claim it supports.
        updated = synthesis_text

        for (level, title), img_path in chart_map.items():
            img_md = f"\n\n![{title}]({img_path})\n"

            if level == 'h3':
                header = f"### {title}"
                header_pos = updated.find(header)
                if header_pos < 0:
                    continue
                # Bound the search to stay inside this finding (don't leak
                # into the next finding or the next H2 section).
                next_h3 = updated.find('\n### ', header_pos + len(header))
                next_h2 = updated.find('\n## ', header_pos + len(header))
                bounds = [p for p in (next_h3, next_h2) if p >= 0]
                block_end = min(bounds) if bounds else len(updated)
                # Try CONFOUND-STATUS first (preferred insertion point)
                confound_pos = updated.find('**CONFOUND-STATUS:', header_pos)
                if 0 <= confound_pos < block_end:
                    insert_pos = confound_pos
                else:
                    # Fallback: before the '---' separator at end of finding
                    hr_pos = updated.find('\n---', header_pos)
                    if 0 <= hr_pos < block_end:
                        insert_pos = hr_pos
                    else:
                        insert_pos = block_end
                updated = updated[:insert_pos] + img_md + "\n" + updated[insert_pos:]
            else:
                # H2: legacy behaviour — insert after first paragraph
                header = f"## {title}"
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
    # BRIEFING ARTEFACTS
    # After briefing generation, two companion files
    # are written alongside briefing.md:
    #   findings_index.md  — mechanical, from insight_tree
    #   structural_map.md  — §1 sliced out of the briefing
    # HTML renderings (with citation links) are produced
    # by OutputManager via write_final_outputs.
    # ══════════════════════════════════════════════

    @staticmethod
    def _extract_briefing_section(briefing_text, section_number):
        """Extract the content of `## §N. ...` from the briefing text.

        Returns the section body (without the header line itself), or
        empty string if the section can't be found.

        Robust to minor header variations — looks for the `§N.` prefix
        in a `## ` line.
        """
        if not briefing_text:
            return ""

        # Find the start: a line beginning with `## ` containing `§N.`
        pattern_start = re.compile(
            rf'^##\s+§\s*{section_number}\.\s*[^\n]*\n',
            re.MULTILINE,
        )
        start_match = pattern_start.search(briefing_text)
        if not start_match:
            return ""

        body_start = start_match.end()

        # Find the end: next `## ` header at column 0, or end of document
        pattern_next = re.compile(r'^##\s+', re.MULTILINE)
        next_match = pattern_next.search(briefing_text, body_start)
        if next_match:
            body_end = next_match.start()
        else:
            body_end = len(briefing_text)

        return briefing_text[body_start:body_end].strip()

    def _build_findings_index(self):
        """Build findings_index.md content from the insight tree.

        Mechanical — no LLM call. One entry per active (winning) analysis,
        in chronological order by chain_id. Runner-up analyses are NOT
        included.
        """
        active = sorted(
            [n for n in self.insight_tree.values() if n.get('status') == 'active'],
            key=lambda n: n['chain_id'],
        )

        lines = [
            "# Findings Index",
            "",
            "All winning analyses from the investigation, in chronological order. "
            "Each entry shows the evaluator score, analytical method (when recorded), "
            "iteration number, and the chain_id link that opens the full analysis.",
            "",
            f"**Total winning analyses:** {len(active)}",
            "",
            "---",
            "",
        ]

        for n in active:
            score = n.get('quality_score', 0)
            chain_id = n.get('chain_id', '?')
            method = n.get('method_used', '')
            summary = n.get('finding_summary', '') or '(no summary)'
            iter_num = n.get('iteration_added', '')
            question = n.get('question', '')

            # Header line: score, iteration, chain_id link
            header_bits = [f"**[{score}/10]**"]
            if iter_num:
                header_bits.append(f"iter {iter_num}")
            header_bits.append(f"[[{chain_id}]]")
            if method:
                header_bits.append(f"_{method}_")
            lines.append(" · ".join(header_bits))

            # Question (trimmed)
            if question:
                q_trim = question[:200] + ('…' if len(question) > 200 else '')
                lines.append(f"*Q: {q_trim}*")

            # Finding summary
            lines.append(summary)
            lines.append("")

        return '\n'.join(lines)

    def _write_briefing_artefacts(self, briefing_text):
        """Write findings_index.md and structural_map.md alongside briefing.

        findings_index.md is mechanical from the insight tree (winners only,
        no runner-ups).

        structural_map.md comes from the research model's Structural Landscape
        section, which has been accumulated across the run (seeded by
        orientation, extended by strategic review). If the research model
        has no Structural Landscape content (e.g., an old state.json from
        before the refactor, or a run where neither orientation nor strategic
        review contributed), fall back to extracting §1 from the briefing.

        HTML versions are produced by OutputManager via write_final_outputs,
        which reads the MDs written here.
        """
        # 1. findings_index.md — mechanical from insight tree
        index_content = self._build_findings_index()
        index_path = os.path.join(self.engine.output_dir, "findings_index.md")
        try:
            with open(index_path, 'w') as f:
                f.write(index_content)
            logger.info(f"Findings index written to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to write findings_index.md: {e}")

        # 2. structural_map.md — primary source: research model's Structural Landscape
        landscape = self._extract_model_section('Structural Landscape')
        # Discard placeholder banner (present when Opus never wrote content)
        if landscape and '<<< DO NOT MODIFY' in landscape:
            landscape = ""

        if landscape:
            # Write from live research model state
            try:
                self._write_live_structural_map()
                logger.info("Structural map written from research model")
            except Exception as e:
                logger.warning(f"Failed to write structural_map.md from model: {e}")
        else:
            # Fallback: extract §1 from briefing text (old behaviour)
            section_1_body = self._extract_briefing_section(briefing_text, section_number=1)
            if section_1_body:
                map_path = os.path.join(self.engine.output_dir, "structural_map.md")
                try:
                    with open(map_path, 'w') as f:
                        f.write("# Structural Landscape\n\n")
                        f.write(
                            "*Extracted from the briefing §1 (fallback — "
                            "research model had no accumulated Structural "
                            "Landscape).*\n\n"
                        )
                        f.write(section_1_body)
                    logger.info(f"Structural map written to {map_path} (fallback from briefing)")
                except Exception as e:
                    logger.warning(f"Failed to write structural_map.md: {e}")
            else:
                logger.info(
                    "Structural map not written — no Structural Landscape in "
                    "research model and could not locate §1 in briefing text"
                )

        # 3. run_geometry.html — single self-contained interactive page
        #    (silently skipped if too few embeddings to be meaningful)
        try:
            geometry_html = self._build_run_geometry_html()
            if geometry_html:
                geom_path = os.path.join(self.engine.output_dir, "run_geometry.html")
                with open(geom_path, 'w') as f:
                    f.write(geometry_html)
                logger.info(f"Run geometry written to {geom_path}")

                # Clean up legacy artefacts from earlier code versions so the
                # output dir doesn't accumulate stale files. Best-effort; any
                # failure is non-fatal.
                legacy_md = os.path.join(self.engine.output_dir, "run_geometry.md")
                if os.path.exists(legacy_md):
                    try:
                        os.remove(legacy_md)
                        logger.debug(f"Removed legacy {legacy_md}")
                    except Exception:
                        pass
                legacy_dir = os.path.join(self.engine.output_dir, "run_geometry")
                if os.path.isdir(legacy_dir):
                    try:
                        import shutil
                        shutil.rmtree(legacy_dir)
                        logger.debug(f"Removed legacy {legacy_dir}/")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to write run_geometry.html: {e}")

    def _build_run_geometry_html(self):
        """Build run_geometry.html — a single self-contained interactive page.

        Replaces the previous markdown+conceptual_map split. Contains:
          - Topology table (lineage stats + Hz phase shares)
          - Interactive dispersion timeline (hover any point)
          - Interactive lineage-depth strip (hover any cell)
          - Embedded conceptual map (t-SNE + phase shading + hover lineage)
          - Shallow-leaves list (unfollowed early work)

        Soft dependencies (sklearn, Pillow) for the conceptual map. If they're
        missing, the map section is skipped with a notice; the rest still renders.

        Returns '' when fewer than 3 embedded nodes.
        """
        from embeddings import rolling_dispersion, z_score, compute_lineage
        import html as html_mod
        import re

        active = sorted(
            [n for n in self.insight_tree.values() if n.get('status') == 'active'],
            key=lambda n: n['chain_id'],
        )
        if not active:
            return ''

        vectors = [n.get('embedding') for n in active]
        embedded = [v for v in vectors if v is not None]
        if len(embedded) < 3:
            return ''

        H = rolling_dispersion(vectors, window=5)
        H_z = z_score(H)
        parents, par_dists, depths = compute_lineage(vectors)

        scores = [n.get('quality_score', 0) for n in active]
        iters = [n.get('iteration_added', i + 1) for i, n in enumerate(active)]
        n = len(active)
        embedded_count = len(embedded)
        max_iter = max(iters)

        def _clean(text):
            if not text:
                return ''
            return re.split(r'═{2,}', text, maxsplit=1)[0].strip()

        estimands = [_clean(node.get('tested_estimand') or '') for node in active]
        summaries = [(_clean(node.get('finding_summary') or '')
                      or _clean(node.get('result_digest') or '')
                      or _clean(node.get('question') or '')
                      or '(no summary)') for node in active]

        # ── Topology stats ──
        valid_hz = [z for z in H_z if z is not None]
        tight = sum(1 for z in valid_hz if z < -0.5)
        broad = sum(1 for z in valid_hz if z > 0.5)
        mixed = len(valid_hz) - tight - broad
        valid_depths = [d for d in depths if d is not None]
        max_depth = max(valid_depths) if valid_depths else 0
        median_depth = (sorted(valid_depths)[len(valid_depths) // 2]
                        if valid_depths else 0)
        from collections import Counter
        child_counts = Counter(p for p in parents if p is not None)
        branch_points = sum(1 for c in child_counts.values() if c > 1)
        pivot_moves = sum(1 for d in par_dists if d is not None and d >= 0.25)
        leaf_idxs = [i for i in range(n) if i not in child_counts]
        orphan_iters_data = []
        for i in leaf_idxs:
            if depths[i] is not None and depths[i] <= 3 and i != 0:
                orphan_iters_data.append((iters[i], depths[i], scores[i], summaries[i]))
        orphan_iters_data.sort(key=lambda r: r[0])

        # ── Longest score-9 streak (for lede) ──
        longest, cur = [], []
        for it, sc in zip(iters, scores):
            if sc >= 9:
                cur.append(it)
                if len(cur) > len(longest):
                    longest = list(cur)
            else:
                cur = []

        if longest and len(longest) >= 3:
            during_hz = [z for it, z in zip(iters, H_z) if it in longest and z is not None]
            descriptor = ''
            if during_hz:
                mean_during = sum(during_hz) / len(during_hz)
                descriptor = ("tightest semantic focus" if mean_during < 0
                              else "broadest semantic reach")
            lede = (
                f"Investigation produced a lineage tree of depth {max_depth} with "
                f"{branch_points} branch points and {pivot_moves} pivot moves. "
                f"Its highest-quality stretch (iters {longest[0]}–{longest[-1]}, "
                f"{len(longest)} consecutive score-9 findings)"
            )
            if descriptor:
                lede += f" coincided with the trajectory's {descriptor}."
            else:
                lede += "."
        else:
            lede = (
                f"Investigation produced a lineage tree of depth {max_depth} with "
                f"{branch_points} branch points and {pivot_moves} pivot moves."
            )

        seed_q = (self.seed_question or '').strip()[:300]

        # ── Interactive dispersion sparkline ──
        spark_w, spark_h = 920, 110
        spark_pad = 14
        z_max_obs = max((abs(z) for z in valid_hz), default=2.0)
        z_range = max(2.0, z_max_obs)
        spark_inner_w = spark_w - 2 * spark_pad
        spark_inner_h = spark_h - 2 * spark_pad

        def _spark_xy(idx, z):
            x = spark_pad + (idx / max(n - 1, 1)) * spark_inner_w
            y = spark_pad + spark_inner_h / 2 - (z / z_range) * (spark_inner_h / 2 - 4)
            return x, y

        line_pts = []
        for i, z in enumerate(H_z):
            if z is None:
                continue
            x, y = _spark_xy(i, z)
            line_pts.append(f"{x:.1f},{y:.1f}")

        mid_y = spark_pad + spark_inner_h / 2

        # Hover dots — invisible larger circles act as hover targets;
        # visible dots are smaller and stay non-interactive.
        dot_layer = []
        for i, z in enumerate(H_z):
            if z is None:
                continue
            x, y = _spark_xy(i, z)
            sc = scores[i]
            est = estimands[i][:220] + ('…' if len(estimands[i]) > 220 else '')
            tip = f"iter {iters[i]} · score {sc} · Hz {z:+.2f}σ\n\n{est}"
            # Visible dot (score-9+ gets gold, others a thin orange)
            if sc >= 9:
                dot_layer.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#E8C547" '
                    f'stroke="#1a1a1a" stroke-width="0.5" pointer-events="none"/>'
                )
            else:
                dot_layer.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#D87C5A" '
                    f'pointer-events="none"/>'
                )
            # Invisible hover target
            dot_layer.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="transparent" '
                f'class="mtip" data-tip="{html_mod.escape(tip)}"/>'
            )

        dispersion_svg = (
            f'<svg viewBox="0 0 {spark_w} {spark_h}" width="100%" height="{spark_h}" '
            f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" '
            f'class="chart-svg">'
            f'<line x1="{spark_pad}" y1="{mid_y}" x2="{spark_w - spark_pad}" '
            f'y2="{mid_y}" stroke="var(--border2)" stroke-width="0.6" '
            f'stroke-dasharray="2,3"/>'
            f'<polyline points="{" ".join(line_pts)}" fill="none" '
            f'stroke="#D87C5A" stroke-width="1.5"/>'
            f'{"".join(dot_layer)}'
            f'<text x="{spark_w - 4}" y="{spark_pad + 4}" font-size="9" '
            f'text-anchor="end" fill="var(--fg3)" font-family="var(--mono)">'
            f'+{z_range:.0f}σ</text>'
            f'<text x="{spark_w - 4}" y="{spark_h - 4}" font-size="9" '
            f'text-anchor="end" fill="var(--fg3)" font-family="var(--mono)">'
            f'−{z_range:.0f}σ</text>'
            f'</svg>'
        )

        # ── Interactive depth strip ──
        # Width is responsive — outer container is a flex row, cells flex to fill.
        c_low = (228, 233, 240)
        c_high = (28, 52, 89)
        strip_cells_html = []
        for i in range(n):
            d_i = depths[i]
            if d_i is None:
                color = '#9C9B96'
                d_str = '—'
            else:
                t = d_i / max(max_depth, 1)
                rgb = tuple(int(c_low[k] + t * (c_high[k] - c_low[k]))
                            for k in range(3))
                color = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
                d_str = str(d_i)
            par_i = parents[i] if i < len(parents) else None
            par_d = par_dists[i] if i < len(par_dists) else None
            if par_i is None:
                parent_str = '— (root)'
            else:
                par_iter_num = iters[par_i] if par_i < len(iters) else par_i
                par_d_str = f"{par_d:.3f}" if par_d is not None else "—"
                parent_str = f"iter {par_iter_num} (d={par_d_str})"
            tip = (f"iter {iters[i]} · score {scores[i]} · depth {d_str}\n"
                   f"parent: {parent_str}\n\n"
                   f"{summaries[i][:240] + ('…' if len(summaries[i]) > 240 else '')}")
            strip_cells_html.append(
                f'<div class="strip-cell mtip" style="background:{color}" '
                f'data-tip="{html_mod.escape(tip)}"></div>'
            )

        # Tick marks every 5 iters
        tick_labels = []
        for i in range(n):
            if iters[i] % 5 == 0 or iters[i] == 1:
                pct = (i / max(n - 1, 1)) * 100
                tick_labels.append(
                    f'<span style="left:{pct:.2f}%">{iters[i]}</span>'
                )
        strip_ticks_html = ''.join(tick_labels)

        # ── Conceptual map (soft-skipped if deps missing or trajectory degenerate) ──
        conceptual_section = self._build_conceptual_map_section(
            vectors, iters, scores, estimands, parents, max_iter, c_low, c_high
        )

        # ── Shallow leaves section ──
        if orphan_iters_data:
            orphan_items = ''.join(
                f'<li><b>iter {it}</b> (depth {dp}, score {sc}): '
                f'{html_mod.escape(summary[:280] + ("…" if len(summary) > 280 else ""))}</li>'
                for it, dp, sc, summary in orphan_iters_data[:10]
            )
            shallow_html = f"""
  <section>
    <h2>Shallow leaves — unfollowed early work</h2>
    <p class="caption">Iterations attached close to the root that produced no
    children — exploratory directions the loop opened but didn't pursue.</p>
    <ul class="leaf-list">{orphan_items}</ul>
  </section>"""
        else:
            shallow_html = ''

        model_used = next(
            (node.get('embedding_model') for node in active
             if node.get('embedding_model')),
            'unknown',
        )

        topology_rows = ''.join([
            f'<tr><td>Iterations (winning)</td><td>{n}</td></tr>',
            f'<tr><td>Embedded</td><td>{embedded_count}</td></tr>',
            f'<tr><td>Max lineage depth</td><td>{max_depth}</td></tr>',
            f'<tr><td>Median depth</td><td>{median_depth}</td></tr>',
            f'<tr><td>Branch points</td><td>{branch_points}</td></tr>',
            f'<tr><td>Pivot moves (parent dist ≥ 0.25)</td><td>{pivot_moves}</td></tr>',
            f'<tr><td>Tight phase (Hz &lt; −0.5)</td><td>{tight} '
            f'({tight / max(len(valid_hz), 1) * 100:.0f}%)</td></tr>',
            f'<tr><td>Broad phase (Hz &gt; +0.5)</td><td>{broad} '
            f'({broad / max(len(valid_hz), 1) * 100:.0f}%)</td></tr>',
            f'<tr><td>Mixed</td><td>{mixed} '
            f'({mixed / max(len(valid_hz), 1) * 100:.0f}%)</td></tr>',
        ])

        doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shape of the Investigation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  :root {{
    --bg: #FAFAF8; --bg2: #F1F0EC; --bg3: #E8E7E3;
    --fg: #1A1A18; --fg2: #6B6A66; --fg3: #9C9B96;
    --border: #E0DFDB; --border2: #CCCBC6;
    --accent: #D87C5A;
    --font: 'IBM Plex Sans', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #161615; --bg2: #1E1E1C; --bg3: #282826;
      --fg: #E8E7E3; --fg2: #9C9B96; --fg3: #6B6A66;
      --border: #2E2E2B; --border2: #3A3A37;
    }}
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); font-family: var(--font);
         font-size: 14px; line-height: 1.55;
         padding: 32px 24px; max-width: 1280px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px;
       letter-spacing: -0.3px; }}
  h2 {{ font-size: 15px; font-weight: 500; margin: 24px 0 4px;
       letter-spacing: 0.02em; color: var(--fg);
       border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .seed {{ font-size: 12px; color: var(--fg2); margin-bottom: 16px;
          font-style: italic; font-family: var(--mono); }}
  .lede {{ background: var(--bg2); border-left: 3px solid var(--accent);
          padding: 10px 14px; margin: 16px 0 24px; font-size: 14px;
          color: var(--fg); border-radius: 0 4px 4px 0; }}
  .caption {{ font-size: 12px; color: var(--fg2); margin: 6px 0 12px;
             line-height: 1.55; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px;
          font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left;
           border-bottom: 1px solid var(--border); }}
  th {{ font-size: 11px; color: var(--fg2); text-transform: uppercase;
       letter-spacing: 0.08em; font-weight: 500; }}
  td:last-child {{ text-align: right; font-family: var(--mono);
                  font-variant-numeric: tabular-nums; }}
  .chart-svg {{ background: var(--bg2); border: 1px solid var(--border);
               border-radius: 4px; display: block; }}

  /* Depth strip — flex row, responsive width */
  .strip-wrap {{ background: var(--bg2); border: 1px solid var(--border);
                border-radius: 4px; padding: 0; }}
  .strip-cells {{ display: flex; height: 26px; border-radius: 4px 4px 0 0;
                 overflow: hidden; }}
  .strip-cell {{ flex: 1 1 0; min-width: 0; border-right: 1px solid var(--bg2);
                cursor: default; transition: opacity 0.1s; }}
  .strip-cell:last-child {{ border-right: none; }}
  .strip-cell:hover {{ outline: 1.5px solid var(--fg); outline-offset: -1px;
                      z-index: 2; position: relative; }}
  .strip-ticks {{ position: relative; height: 18px; font-size: 9px;
                 color: var(--fg2); font-family: var(--mono);
                 border-top: 1px solid var(--border); }}
  .strip-ticks span {{ position: absolute; top: 4px; transform: translateX(-50%); }}

  /* Floating tooltip — used by dispersion + strip; appears near cursor */
  #floating-tip {{ position: fixed; max-width: 420px; background: var(--bg);
                  border: 1px solid var(--border2); border-radius: 4px;
                  padding: 10px 12px; font-size: 12px; line-height: 1.5;
                  pointer-events: none; display: none; z-index: 100;
                  white-space: pre-line; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                  color: var(--fg); }}

  /* Conceptual map — fixed info panel above the SVG */
  #map-tip {{ background: var(--bg2); border: 1px solid var(--border);
             border-radius: 4px; padding: 12px 16px; font-size: 12px;
             line-height: 1.5; white-space: pre-line; margin: 8px 0 12px;
             height: 110px; overflow-y: auto; color: var(--fg); }}
  #map-tip.empty {{ color: var(--fg3); font-style: italic; }}
  .map-wrap {{ background: var(--bg2); border: 1px solid var(--border);
              border-radius: 4px; padding: 8px; overflow: hidden; }}
  .map-svg {{ display: block; max-width: 100%; height: auto; }}
  .mnode {{ cursor: pointer; transition: stroke-width 0.1s; }}
  .mnode:hover {{ stroke-width: 2.4; }}
  .map-legend {{ font-size: 11px; color: var(--fg2); margin: 10px 0;
                display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
                font-family: var(--mono); }}
  .map-legend svg {{ display: block; }}

  ul.leaf-list {{ list-style: none; padding-left: 0; }}
  ul.leaf-list li {{ font-size: 13px; margin-bottom: 8px; padding-left: 16px;
                    position: relative; }}
  ul.leaf-list li::before {{ content: '·'; position: absolute; left: 0;
                            color: var(--fg3); font-weight: 600; }}

  footer {{ font-size: 11px; color: var(--fg2); margin-top: 36px;
           padding-top: 12px; border-top: 1px solid var(--border);
           font-style: italic; line-height: 1.6; }}
</style>
</head><body>

<h1>Shape of the Investigation</h1>
<div class="seed">seed: {html_mod.escape(seed_q)}</div>

<div class="lede">{html_mod.escape(lede)}</div>

<section>
  <h2>Topology</h2>
  <table>
    <thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
    <tbody>{topology_rows}</tbody>
  </table>
</section>

<section>
  <h2>Dispersion timeline</h2>
  <p class="caption">Hz = z-scored mean pairwise cosine distance over a 5-iteration
  sliding window. Negative = the loop was deepening one conceptual neighbourhood;
  positive = it was spreading across distinct regions. Gold dots mark score-9+
  iterations. <b>Hover any point</b> for iter, score, Hz, and the tested estimand.</p>
  {dispersion_svg}
</section>

<section>
  <h2>Lineage depth · iter 1 → {max_iter}</h2>
  <p class="caption">Each cell is one iteration. Color = depth in the conceptual
  lineage tree — pale = shallow (close to iter 1), dark = deeply built on prior
  work. Each iter's parent is its nearest prior iteration in embedding space.
  <b>Hover any cell</b> for iter, parent, depth, and the finding.</p>
  <div class="strip-wrap">
    <div class="strip-cells">{''.join(strip_cells_html)}</div>
    <div class="strip-ticks">{strip_ticks_html}</div>
  </div>
</section>

{conceptual_section}

{shallow_html}

<footer>
Embeddings: <code>{html_mod.escape(model_used)}</code>.
Hz computed as mean pairwise cosine distance over a 5-iteration sliding window,
then z-scored against the run mean. Lineage tree built by attaching each
iteration to its nearest prior in embedding space — deterministic, no clustering.
This is observability only — no decision in the run was informed by these metrics.
</footer>

<div id="floating-tip"></div>

<script>
const floatTip = document.getElementById('floating-tip');
document.addEventListener('mousemove', e => {{
  const t = e.target.closest('.mtip');
  if (!t) {{ floatTip.style.display = 'none'; return; }}
  floatTip.textContent = t.getAttribute('data-tip');
  floatTip.style.display = 'block';
  const tipWidth = 420;
  floatTip.style.left = Math.min(e.clientX + 14, window.innerWidth - tipWidth - 16) + 'px';
  floatTip.style.top = (e.clientY + 14) + 'px';
}});
</script>
</body></html>"""
        return doc

    def _build_conceptual_map_section(self, vectors, iters, scores, estimands,
                                       parents, max_iter, c_low, c_high):
        """Build the conceptual-map section (t-SNE + phase shading + hover lineage).

        Returns a complete HTML <section>...</section> string, or an empty
        section with a placeholder notice if dependencies are missing or the
        projection fails.
        """
        try:
            import numpy as np
            from sklearn.manifold import TSNE
            from PIL import Image
        except Exception as e:
            return f"""
  <section>
    <h2>Conceptual map</h2>
    <p class="caption">Could not render: missing dependency ({e}).
    Install <code>scikit-learn</code> and <code>Pillow</code> to enable.</p>
  </section>"""

        import base64
        import io
        import html as html_mod

        try:
            embedded_idx = [i for i, v in enumerate(vectors) if v is not None]
            X = np.array([vectors[i] for i in embedded_idx], dtype=float)
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Numerical guards: explicit zero-norm protection, plus a blanket
            # errstate suppression for the matmul itself. Embedding rows that
            # are extreme or degenerate (very large components, near-zero norm
            # after upstream cleanup, etc.) can still trigger cascading
            # divide/overflow/invalid warnings inside the BLAS-backed matmul
            # even with the norm guard in place — the warnings are diagnostic
            # noise, not correctness bugs. The post-matmul `nan_to_num` ensures
            # any NaN/Inf that leaks through gets replaced with sensible
            # defaults (max distance) before t-SNE sees the matrix.
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms = np.where(norms > 1e-12, norms, 1.0)
                Xn = X / norms
                dist_mat = np.clip(1 - Xn @ Xn.T, 0, 2)
            dist_mat = np.nan_to_num(dist_mat, nan=1.0, posinf=2.0, neginf=0.0)
            np.fill_diagonal(dist_mat, 0)

            perplexity = min(10, max(5, len(X) // 5))
            tsne = TSNE(
                n_components=2, metric='precomputed', init='random',
                perplexity=perplexity, learning_rate='auto', random_state=42,
            )
            coords_emb = tsne.fit_transform(dist_mat)

            mn, mx = coords_emb.min(axis=0), coords_emb.max(axis=0)
            coords_emb = (coords_emb - mn) / (mx - mn + 1e-9)
            if coords_emb[0, 1] > 0.5:
                coords_emb[:, 1] = 1 - coords_emb[:, 1]
            coords_emb[:, 0] += (0.5 - coords_emb[0, 0])
            x_lo, x_hi = coords_emb[:, 0].min(), coords_emb[:, 0].max()
            max_offset = max(0.5 - x_lo, x_hi - 0.5)
            if max_offset > 0.45:
                coords_emb[:, 0] = (
                    0.5 + (coords_emb[:, 0] - 0.5) * (0.45 / max_offset)
                )
            coords_emb[:, 1] = coords_emb[:, 1] * 0.90 + 0.05

            coords = np.full((len(vectors), 2), np.nan)
            for k, idx in enumerate(embedded_idx):
                coords[idx] = coords_emb[k]

            iter_norm = np.array(
                [(iters[i] - 1) / max(max_iter - 1, 1)
                 if vectors[i] is not None else 0.0
                 for i in range(len(vectors))],
                dtype=float,
            )
            xs, ys = coords[:, 0], coords[:, 1]

            img_w, img_h = 1280, 720
            pad = 60
            inner_w, inner_h = img_w - 2 * pad, img_h - 2 * pad
            grid_nx, grid_ny = inner_w // 2, inner_h // 2
            sigma = 0.07

            gx = np.linspace(0, 1, grid_nx)
            gy = np.linspace(0, 1, grid_ny)
            X_grid, Y_grid = np.meshgrid(gx, gy)
            mass = np.zeros((grid_ny, grid_nx))
            weighted_sum = np.zeros((grid_ny, grid_nx))
            for k in range(len(vectors)):
                if vectors[k] is None:
                    continue
                d2 = (X_grid - xs[k]) ** 2 + (Y_grid - ys[k]) ** 2
                w = np.exp(-d2 / (2 * sigma ** 2))
                mass += w
                weighted_sum += w * iter_norm[k]
            mean_iter = np.where(
                mass > 0.001, weighted_sum / np.maximum(mass, 1e-9), 0
            )

            c0 = np.array(c_low)
            c1 = np.array(c_high)
            nonzero_mass = mass[mass > 0.01]
            mass_cap = (np.percentile(nonzero_mass, 80)
                        if len(nonzero_mass) else 1.0)
            alpha_map = np.clip(mass / mass_cap, 0, 1) * 0.45

            rgba = np.zeros((grid_ny, grid_nx, 4), dtype=np.uint8)
            valid_mask = mass >= 0.01
            for ch, (lo, hi) in enumerate(zip(c0, c1)):
                rgba[..., ch] = np.where(
                    valid_mask, (lo + mean_iter * (hi - lo)).astype(np.uint8), 0
                )
            rgba[..., 3] = np.where(
                valid_mask, (alpha_map * 255).astype(np.uint8), 0
            )

            img = Image.fromarray(rgba, 'RGBA').resize(
                (inner_w, inner_h), Image.BILINEAR
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

            def _px(i):
                return (pad + coords[i, 0] * inner_w,
                        pad + coords[i, 1] * inner_h)

            def _grad(t):
                return tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))

            svg_parts = [
                f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{inner_h}" '
                f'fill="var(--bg)" stroke="var(--border)" stroke-width="1"/>',
                f'<image href="data:image/png;base64,{img_b64}" '
                f'x="{pad}" y="{pad}" width="{inner_w}" height="{inner_h}" '
                f'preserveAspectRatio="none"/>',
                '<g id="lineage-layer"></g>',
            ]

            first_embedded = embedded_idx[0] if embedded_idx else None

            def path_to_root(i):
                chain = [i]
                while parents[i] is not None:
                    i = parents[i]
                    chain.append(i)
                return chain

            for i in range(len(vectors)):
                if vectors[i] is None:
                    continue
                x, y = _px(i)
                sc = scores[i]
                r = 4 + max(0, sc - 5) * 0.9
                rgb = _grad((iters[i] - 1) / max(max_iter - 1, 1))
                fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                est_excerpt = (estimands[i][:280]
                               + ('…' if len(estimands[i]) > 280 else ''))
                tip = f"iter {iters[i]} · score {sc}\n\n{est_excerpt}"
                chain = path_to_root(i)
                chain_iters = ','.join(str(iters[j]) for j in chain)
                if i == first_embedded:
                    svg_parts.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 6:.1f}" '
                        f'fill="none" stroke="var(--fg)" stroke-width="1.2" '
                        f'stroke-dasharray="2,2" pointer-events="none"/>'
                    )
                    svg_parts.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2:.1f}" '
                        f'fill="{fill}" stroke="var(--fg)" stroke-width="1.3" '
                        f'class="mnode" data-tip="{html_mod.escape(tip)}" '
                        f'data-path="{chain_iters}"/>'
                    )
                    svg_parts.append(
                        f'<text x="{x:.1f}" y="{y - r - 12:.1f}" font-size="11" '
                        f'text-anchor="middle" fill="var(--fg)" font-weight="600" '
                        f'font-family="var(--mono)">iter {iters[i]} (anchor)</text>'
                    )
                else:
                    svg_parts.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                        f'fill="{fill}" stroke="var(--fg)" stroke-width="0.7" '
                        f'class="mnode" data-tip="{html_mod.escape(tip)}" '
                        f'data-path="{chain_iters}"/>'
                    )
                    if sc >= 9:
                        svg_parts.append(
                            f'<text x="{x + r + 3:.1f}" y="{y + 3:.1f}" '
                            f'font-size="10" fill="var(--fg2)" '
                            f'font-family="var(--mono)">{iters[i]}</text>'
                        )

            map_svg = (
                f'<svg viewBox="0 0 {img_w} {img_h}" width="100%" '
                f'preserveAspectRatio="xMidYMid meet" '
                f'xmlns="http://www.w3.org/2000/svg" class="map-svg">'
                + ''.join(svg_parts) + '</svg>'
            )

            iter_xy_pairs = []
            for i in range(len(vectors)):
                if vectors[i] is None:
                    continue
                x, y = _px(i)
                iter_xy_pairs.append(f'"{iters[i]}":[{x:.1f},{y:.1f}]')
            iter_xy_js = '{' + ','.join(iter_xy_pairs) + '}'

            sizes_legend = [(5, 4), (7, 5.8), (9, 7.6), (10, 8.5)]
            size_swatches = ''.join(
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'margin-right:14px"><svg width="22" height="22">'
                f'<circle cx="11" cy="11" r="{r}" fill="var(--fg3)" '
                f'stroke="var(--fg)" stroke-width="0.6"/></svg>'
                f'score {sc}</span>' for sc, r in sizes_legend
            )

            return f"""
  <section>
    <h2>Conceptual map</h2>
    <p class="caption">2D projection of all iterations by conceptual similarity
    (t-SNE on cosine distance). <b>Spatial proximity = conceptual similarity.</b>
    iter 1 anchored at top-centre; the run unfolds downward. Dot color and
    background shading both encode iteration (pale = early, dark = late).
    Dot size = score. <b>Hover any iter</b> to draw its lineage path back to iter 1.</p>
    <div class="map-legend">
      <span style="display:inline-flex;align-items:center;gap:8px">
        iter 1
        <svg width="180" height="12">
          <defs><linearGradient id="g_iter" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stop-color="rgb({c0[0]},{c0[1]},{c0[2]})"/>
            <stop offset="100%" stop-color="rgb({c1[0]},{c1[1]},{c1[2]})"/>
          </linearGradient></defs>
          <rect x="0" y="0" width="180" height="12" fill="url(#g_iter)"
            stroke="var(--fg)" stroke-width="0.5"/>
        </svg>
        iter {max_iter}
      </span>
      <span>{size_swatches}</span>
    </div>
    <div id="map-tip" class="empty">Hover any iteration to see its estimand and lineage</div>
    <div class="map-wrap">{map_svg}</div>
  </section>

<script>
(() => {{
  const mapTip = document.getElementById('map-tip');
  const iterXY = {iter_xy_js};
  const lineageLayer = document.getElementById('lineage-layer');
  const svgNS = 'http://www.w3.org/2000/svg';
  const emptyMsg = 'Hover any iteration to see its estimand and lineage';

  function drawLineage(pathStr) {{
    clearLineage();
    const its = pathStr.split(',');
    if (its.length < 2) return;
    const pts = its.map(it => iterXY[it]).filter(p => p);
    if (pts.length < 2) return;
    const polyline = document.createElementNS(svgNS, 'polyline');
    polyline.setAttribute('points', pts.map(p => p.join(',')).join(' '));
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', '#D87C5A');
    polyline.setAttribute('stroke-width', '1.6');
    polyline.setAttribute('stroke-opacity', '0.85');
    lineageLayer.appendChild(polyline);
    pts.forEach((p, idx) => {{
      if (idx === 0 || idx === pts.length - 1) return;
      const c = document.createElementNS(svgNS, 'circle');
      c.setAttribute('cx', p[0]); c.setAttribute('cy', p[1]);
      c.setAttribute('r', '2.5'); c.setAttribute('fill', '#D87C5A');
      c.setAttribute('stroke', '#1a1a1a'); c.setAttribute('stroke-width', '0.5');
      lineageLayer.appendChild(c);
    }});
  }}
  function clearLineage() {{
    while (lineageLayer.firstChild) lineageLayer.removeChild(lineageLayer.firstChild);
  }}
  // Map nodes drive the map-tip and lineage; suppress the floating tooltip on them.
  document.addEventListener('mousemove', e => {{
    const m = e.target.closest('.mnode');
    if (!m) {{
      mapTip.textContent = emptyMsg;
      mapTip.classList.add('empty');
      clearLineage();
      return;
    }}
    // Also hide floating-tip so it doesn't double up
    const ft = document.getElementById('floating-tip');
    if (ft) ft.style.display = 'none';
    mapTip.textContent = m.getAttribute('data-tip');
    mapTip.classList.remove('empty');
    drawLineage(m.getAttribute('data-path'));
  }});
}})();
</script>"""
        except Exception as e:
            return f"""
  <section>
    <h2>Conceptual map</h2>
    <p class="caption">Could not render the map: {html_mod.escape(str(e))}</p>
  </section>"""

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
            'tested_estimand': '',
            'iteration_added': getattr(self, '_current_iteration', 0) + 1,
            'embedding': None,
            'embedding_model': None,
        }
        return node_id

    # ══════════════════════════════════════════════
    # EMBEDDINGS (run-geometry observability)
    # ══════════════════════════════════════════════

    @staticmethod
    def _embed_text_for_node(node):
        """Pick the text to embed for a node.

        Prefers `tested_estimand` — the formal causal/statistical target
        of the iteration — over `finding_summary` and `result_digest`.
        The latter two lean methodology-heavy ("Solution 1 fits...",
        "HR-binned non-parametric matching..."), which makes embedding
        clusters track *what apparatus was used* rather than *what
        question was investigated*. `tested_estimand` is substantive
        without that bias.

        Falls back to the older fields for legacy (pre-v8) nodes that
        predate `tested_estimand`, and ultimately to `question` as a
        last resort. Returns '' when nothing usable is available.
        """
        text = (node.get('tested_estimand') or
                node.get('finding_summary') or
                node.get('result_digest') or
                node.get('question') or '')
        text = text.strip()
        return text if len(text) >= 20 else ''

    def _embed_node(self, node_id):
        """Embed a winning node's finding text. Non-fatal — failures silent.

        Called from the run() loop after node enrichment with finding_summary
        and result_digest. The dashboard and run_geometry artefact treat a
        missing embedding as a coverage gap.
        """
        client = getattr(self.engine, 'embedding_client', None)
        if client is None:
            return
        node = self.insight_tree.get(node_id)
        if not node:
            return
        text = self._embed_text_for_node(node)
        if not text:
            return
        model = self.engine.embedding_model
        try:
            vector = client.embed(text, model)
            node['embedding'] = vector
            node['embedding_model'] = model
            # Brief CLI confirmation — keeps the user informed without noise.
            try:
                print(
                    f"    {style.DIM}⋯ embedded ({len(vector)}d){style.RESET}"
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Embedding failed for {node_id}: {e}")

    def _backfill_embeddings(self):
        """Embed any winning nodes that lack embeddings (post-resume backfill).

        Called from run() after _restore_checkpoint when --continue is in
        effect and backfill is enabled. Walks active winning nodes that have
        embeddable text but no embedding, and embeds them sequentially.

        Cost is announced before proceeding so the user knows what's
        happening. Failures are logged and skipped — the panel will show
        partial coverage rather than crashing.
        """
        client = getattr(self.engine, 'embedding_client', None)
        if client is None:
            return  # embeddings disabled

        # Find candidates: active winning nodes with embeddable text + no embedding
        candidates = []
        for nid, node in self.insight_tree.items():
            if node.get('status') != 'active':
                continue
            if node.get('embedding') is not None:
                continue
            if self._embed_text_for_node(node):
                candidates.append(nid)

        if not candidates:
            return

        # Rough cost estimate: assume ~400 tokens per finding_summary.
        # text-embedding-3-small is $0.02/M, so ~$0.000008 per node.
        try:
            from llm import compute_cost
            est_tokens = 400 * len(candidates)
            est_cost = compute_cost(self.engine.embedding_model, est_tokens, 0)
        except Exception:
            est_cost = 0.0

        print(
            f"    {style.DIM}Backfilling embeddings for {len(candidates)} "
            f"historical nodes (≈ ${est_cost:.4f})...{style.RESET}"
        )

        embedded = 0
        failed = 0
        with style.spinner(f"Embedding {len(candidates)} nodes"):
            for nid in candidates:
                try:
                    self._embed_node(nid)
                    if self.insight_tree[nid].get('embedding') is not None:
                        embedded += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(f"Backfill failed for {nid}: {e}")
                    failed += 1

        if failed:
            print(
                f"    {style.DIM}Backfill: {embedded} embedded, "
                f"{failed} failed.{style.RESET}"
            )
        else:
            print(
                f"    {style.DIM}Backfill: {embedded} nodes embedded.{style.RESET}"
            )

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
        """Save complete exploration state to disk.

        Version history:
          v6: removed legacy telemetry fields (model_impact_history etc.)
          v7: added causal_substrate_guidance (now removed in v8)
          v8: removed causal_substrate_guidance (cheap-agent view derived
              from substrate body at call time); added
              _last_search_calibration (literature CALIBRATION assessment).
              insight_tree nodes now carry tested_estimand.
          v9: insight-tree iteration_added is now 1-based (matches
              arc_history, chain_ids, briefing/dashboard display).
          v10: insight-tree nodes carry optional `embedding` (list[float])
              and `embedding_model` (str). Both default to None. Used by
              the dashboard's Run Geometry panel and the run_geometry.html
              artifact. Pre-v10 nodes are migrated to None on load; the
              backfill helper called from run() will embed them on
              --continue when backfill_embeddings is True (default).
        """
        state = {
            "version": 10,
            "iterations_completed": iteration,
            "explorer": {
                "insight_tree": self.insight_tree,
                "question_pool": self.question_pool,
                "research_model": self.research_model,
                "seed_question": self.seed_question,
                "commitment_history": self.commitment_history,
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
                "arc_reference_code": self._arc_reference_code,
                "arc_reference_score": self._arc_reference_score,
                "search_history": self.search_history,
                "published_entries": getattr(self, '_published_entries', []),
                "last_search_calibration": self._last_search_calibration,
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
        """Restore exploration state from a loaded checkpoint dict.

        Migrations:
          - pre-v6: strip legacy research-model sections (Cross-Finding
            Connections, Finding Maturity, Biggest Gap).
          - pre-v7: legacy data_profile may carry the Structural Landscape
            block; strip it (now lives in research model only).
          - pre-v8: drop causal_substrate_guidance field if present (the
            cheap-agent view is now derived from the substrate body at
            call time); ensure insight_tree nodes have tested_estimand
            (defaults to empty string for pre-v8 nodes).
        """
        ex = state['explorer']
        self.insight_tree = ex['insight_tree']
        self.question_pool = ex['question_pool']
        self.research_model = ex['research_model']
        self.seed_question = ex.get('seed_question', '')
        self.commitment_history = ex.get('commitment_history', [])
        # Legacy telemetry fields (model_impact_history, evaluator_score_history,
        # stagnation_count, biggest_gap_history) removed in v6 — silently
        # ignore if present in old state.
        AutoExplorer._node_counter = ex.get('node_counter', 0)
        self.data_profile = ex.get('data_profile', '')
        # Migration: older runs stored the full orientation output (including
        # the Structural Landscape block) in data_profile. The Landscape now
        # lives only in the research model; strip any legacy copy here so
        # resumed runs behave identically to fresh ones.
        if self.data_profile and '###STRUCTURAL_LANDSCAPE_START###' in self.data_profile:
            original_len = len(self.data_profile)
            self.data_profile = self._strip_landscape_from_profile(self.data_profile)
            logger.info(
                f"Legacy data_profile: stripped Landscape block "
                f"({original_len} → {len(self.data_profile)} chars)"
            )
        self.last_review_iteration = ex.get('last_review_iteration',
                                             ex.get('last_connection_iteration', 0))
        self.strategic_next_direction = ex.get('strategic_next_direction', '')
        self.current_arc_direction = ex.get('current_arc_direction', '')

        # ── Migrate legacy research model shape (pre-v6) ──
        # Strip removed sections: Cross-Finding Connections, Finding Maturity,
        # Biggest Gap. The new RI prompt emits neither; keeping them in the
        # model would confuse the first RI call after resume. Structural
        # Landscape is allowed to remain empty — strategic review will
        # populate it when structural discoveries arrive.
        version = state.get('version', 0)
        if version < 6 and self.research_model:
            self.research_model = self._migrate_legacy_model(self.research_model)

        # v7 → v8 migration: ensure insight_tree nodes have the new
        # tested_estimand field. Pre-v8 nodes don't have it; default to
        # empty string. Strategic Review's scope-drift check treats empty
        # as "equivalent to seed" (i.e., conservative — no drift detected
        # for legacy nodes).
        if version < 8:
            for node in self.insight_tree.values():
                node.setdefault('tested_estimand', '')
            # causal_substrate_guidance is dropped silently from legacy state.
            # The cheap-agent view is now extracted from the substrate body
            # in research_model at every cheap-agent call.
            if 'causal_substrate_guidance' in ex:
                logger.info(
                    "Legacy v7 checkpoint: dropping causal_substrate_guidance "
                    "(cheap-agent view now derived from substrate body)"
                )

        # v8 → v9 migration: iteration_added stored the 0-based loop index;
        # v9+ stores the 1-based iteration count to match arc_history,
        # chain_ids, and briefing/dashboard display. Bump every node by +1
        # so resumed runs and dashboards show consistent iteration numbers.
        if version < 9:
            for node in self.insight_tree.values():
                node['iteration_added'] = node.get('iteration_added', 0) + 1
            logger.info(
                f"Legacy checkpoint: migrated {len(self.insight_tree)} "
                f"insight-tree nodes to 1-based iteration_added"
            )

        # v9 → v10 migration: ensure insight_tree nodes have the embedding
        # fields (default None). Pre-v10 nodes never had embeddings; the
        # backfill helper called from run() will populate them when
        # backfill_embeddings is True. The geometry panel handles None
        # entries gracefully (treats them as coverage gaps).
        if version < 10:
            added = 0
            for node in self.insight_tree.values():
                if 'embedding' not in node:
                    node['embedding'] = None
                    node['embedding_model'] = None
                    added += 1
            if added:
                logger.info(
                    f"Legacy checkpoint: added embedding fields to "
                    f"{added} insight-tree nodes (None)"
                )

        self.last_probe_iteration = ex.get('last_probe_iteration', 0)
        self.probe_history = ex.get('probe_history', [])
        self.completed_original_arcs = set(ex.get('completed_original_arcs', []))
        self.rotation_history = ex.get('rotation_history', [])
        self.arc_history = ex.get('arc_history', [])
        self._arc_reference_code = ex.get('arc_reference_code', '')
        self._arc_reference_score = ex.get('arc_reference_score', 0)
        self.search_history = ex.get('search_history', [])
        self._published_entries = ex.get('published_entries', [])
        self._last_search_calibration = ex.get('last_search_calibration', '')
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

    Section order: Context -> Findings Index -> Research Model -> Evidence.
    The Research Model comes before Evidence so the synthesis model reads
    the strategic narrative before drilling into raw numbers.

    Evidence is structured as:
      - Top TOP_K_FULL_RAW winning analyses by score: full untruncated
        stdout (head/tail compressed if over 6KB). These are the centrepiece
        findings whose raw numbers the briefing will most likely need to
        cite verbatim.
      - Remaining active winners with score >= 6: finding_summary and
        result_digest only (no raw stdout). Enough for citation-faithful
        briefing when raw detail isn't needed.
      - Score <= 5 winners: omitted from Section D, but remain in Section B
        (Findings Index) for completeness.
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
    parts.append("===========================================")
    parts.append("SECTION A: EXPLORATION CONTEXT")
    parts.append("===========================================\n")
    parts.append(f"**Original question:** {seed_question}\n")
    if data_profile:
        parts.append(f"**Dataset profile:**\n{data_profile}\n")
    else:
        parts.append("**Mode:** computation-only (no dataset, no orientation profile)\n")
    parts.append(f"**Exploration scope:** {len(active)} winning analyses completed\n")

    # Section B: Findings Index (all active analyses, one line each)
    parts.append("===========================================")
    parts.append("SECTION B: COMPLETE FINDINGS INDEX")
    parts.append("===========================================\n")

    for n in active:
        fs = n.get('finding_summary', '') or '(no summary)'
        method = n.get('method_used', '')
        method_tag = f" [{method}]" if method else ""
        parts.append(f"[{n['quality_score']}]{method_tag} [[{n['chain_id']}]] {fs}")

    # Section C: Research Model (read before evidence - provides the map)
    parts.append("\n===========================================")
    parts.append("SECTION C: RESEARCH MODEL")
    parts.append("===========================================\n")
    parts.append(research_model or "(No research model available)")

    # Section D: Evidence (top-K full raw + mid-tier digests)
    # Pick top-K by score, break ties by higher chain_id (more recent).
    top_k = sorted(
        active,
        key=lambda n: (n['quality_score'], n['chain_id']),
        reverse=True,
    )[:TOP_K_FULL_RAW]
    top_k_ids = {n['chain_id'] for n in top_k}

    # Mid-tier: remaining active winners with score >= 6 not in top-K.
    # Threshold matches the previous mid band so reductions in context
    # pressure come from cutting top-tier volume, not mid-tier coverage.
    mid_tier = [
        n for n in active
        if n['quality_score'] >= 6 and n['chain_id'] not in top_k_ids
    ]

    parts.append("\n===========================================")
    parts.append(
        f"SECTION D: EVIDENCE "
        f"(top {len(top_k)} by score: full results; "
        f"remaining score >= 6: digests only)"
    )
    parts.append("===========================================\n")

    # Top-K: full results (head/tail compressed if long)
    # Present in chronological order within the top-K band so the reader
    # can follow the investigation's arc.
    top_k_chron = sorted(top_k, key=lambda n: n['chain_id'])
    for n in top_k_chron:
        chain_key = str(n['chain_id'])
        result_text = full_results_store.get(
            chain_key, n.get('result_summary', 'Results not available')
        )
        if len(result_text) > 6000:
            result_text = result_text[:3000] + "\n[...truncated...]\n" + result_text[-3000:]

        parts.append(
            f"[[{n['chain_id']}]] Score: {n['quality_score']}/10\n"
            f"Question: {n['question']}\n"
            f"Results:\n{result_text}\n"
            f"{'-' * 5}"
        )

    # Mid-tier: digest-only entries (finding_summary + result_digest)
    if mid_tier:
        parts.append(
            f"\n{'-' * 20}\n"
            f"Digest-only evidence "
            f"(score >= 6, not in top-{TOP_K_FULL_RAW} - cite via Findings Index IDs):\n"
        )
        mid_chron = sorted(mid_tier, key=lambda n: n['chain_id'])
        for n in mid_chron:
            fs = n.get('finding_summary', '')
            digest = n.get('result_digest', '')
            # Combine finding summary and result digest if both present.
            # Both are LLM-curated and compact.
            if fs and digest:
                body = f"{fs}\n  digest: {digest}"
            elif digest:
                body = digest
            elif fs:
                body = fs
            else:
                # Fallback only - should be rare for score >= 6 winners
                body = n.get('result_summary', 'No summary')
                if len(body) > 200:
                    body = body[:200] + '...'
            parts.append(
                f"[[{n['chain_id']}]] [{n['quality_score']}/10] {n['question']}\n  -> {body}"
            )

    return '\n'.join(parts)
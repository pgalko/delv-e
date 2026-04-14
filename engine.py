"""
ExplorationEngine: minimal runtime that satisfies the self.engine.* interface
expected by auto_explore.py. Replaces BambooAI for standalone operation.
"""

import json
import os
import re
import shutil
import time
import threading
import pandas as pd

import style
from output import OutputManager
from prompts import PromptManager
from llm import LLMClient, CostTracker, RunLogger
from executor import CodeExecutor, extract_code

from logger_config import get_logger
logger = get_logger(__name__)

DEFAULT_AGENT_MODEL = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_CODE_MODEL = "anthropic:claude-haiku-4-5-20251001"
STDOUT_TRUNCATE_LINES = 25

# Agents that use the code model (heavier, more capable)
CODE_AGENTS = {"Code Generator", "Error Corrector", "Synthesis Generator"}


class SimpleModelManager:
    """Maps agent names to models. Two tiers: code model (heavy) and agent model (light)."""

    def __init__(self, agent_model=None, code_model=None):
        self.agent_model = agent_model or DEFAULT_AGENT_MODEL
        self.code_model = code_model or DEFAULT_CODE_MODEL

    def get_model_name(self, agent):
        """Returns (model_name, provider) for an agent."""
        if agent in CODE_AGENTS:
            return self.code_model, "anthropic"
        return self.agent_model, "anthropic"


class SimpleMessageManager:
    """
    Minimal message manager. Stores code execution results and QA pairs.
    No conversation history — context comes from QA pairs + research model.
    """

    def __init__(self):
        self.code_exec_results = None
        self.last_code = None
        self.last_plan = None
        self.qa_pairs = []
        # Complete log of ALL questions ever asked.
        # Older questions are summarised to 120-char snippets to control context growth.
        self.all_questions = []
        # Full stdout per chain_id — used ONLY by synthesis (never truncated).
        # Loop agents use the capped tree node result_summary instead.
        self.full_results_store = {}
        # auto_explore saves/restores this in run() setup/teardown
        self.select_analyst_messages = [{"content": ""}]
        # Error patterns from successfully-corrected code — prevents repeat failures
        self.error_patterns = []

    def reset_non_cumul_messages(self):
        self.code_exec_results = None
        self.last_code = None
        self.last_plan = None

    def append_qa_pair(self, question, result, chain_id=None):
        self.qa_pairs.append({
            "question": question,
            "result": result,
            "chain_id": chain_id,
        })
        self.all_questions.append(question)
        # Store full untruncated result for synthesis
        if chain_id and result:
            self.full_results_store[str(chain_id)] = result

    def update_finding_summary(self, chain_id, summary):
        """Enrich a qa_pair with its evaluator-generated finding summary.

        Called after evaluation, before the next iteration's code generation.
        The summary is used by format_qa_pairs for compact context.
        """
        if not summary or not chain_id:
            return
        chain_key = str(chain_id)
        # Scan recent pairs (most likely near the end)
        for qa in reversed(self.qa_pairs):
            if str(qa.get('chain_id')) == chain_key:
                qa['finding_summary'] = summary
                return

    def format_qa_pairs(self, max_qa_pairs=40, include_chain_id=False):
        """Format QA pairs for code generator context.

        Uses evaluator-generated finding_summary for compact, high-quality
        context. Falls back to first 2 result lines if no summary available
        (e.g. seed iteration which bypasses the evaluator).
        """
        if not self.qa_pairs:
            return "(No previous analyses)"
        pairs = self.qa_pairs[-max_qa_pairs:]
        parts = []
        for qa in pairs:
            ref = f" [[{qa['chain_id']}]]" if include_chain_id and qa.get('chain_id') else ""
            summary = qa.get('finding_summary', '')
            if summary:
                parts.append(f"Q: {qa['question']}{ref}\n→ {summary}")
            else:
                # Fallback: extract headline from raw results
                result = str(qa.get('result') or 'No results')
                headline = self._extract_headline(result)
                parts.append(f"Q: {qa['question']}{ref}\n→ {headline}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_headline(result):
        """Extract a compact headline from raw results (fallback when no finding_summary)."""
        start_marker = "###RESULTS_START###"
        end_marker = "###RESULTS_END###"
        s = result.find(start_marker)
        e = result.find(end_marker)
        if s >= 0 and e > s:
            block = result[s + len(start_marker):e].strip()
        else:
            block = result.strip()
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        return ' | '.join(lines[:2])[:300] if lines else 'No results'

    # Generic types whose AttributeErrors are context-dependent, not systematic
    _GENERIC_TYPES = {
        'numpy.ndarray', 'str', 'list', 'dict', 'int', 'float', 'tuple',
        'set', 'NoneType', 'bool', 'Series', 'DataFrame', 'ndarray',
    }

    def record_error_pattern(self, error_text):
        """Record a successfully-corrected error for future CG context.

        Only records errors with clear, non-ambiguous lessons:
        - ModuleNotFoundError / ImportError (module not available)
        - AttributeError on library-specific classes (API changes)

        Skips context-dependent errors (wrong type passed to a method,
        KeyError on column names, generic type mismatches) — these would
        confuse the CG in later iterations where the context differs.
        """
        if not error_text:
            return
        lines = [l.strip() for l in error_text.strip().split('\n')
                 if l.strip() and '[...truncated]' not in l]
        if not lines:
            return
        error_line = lines[-1][:200]
        if len(error_line) < 15:
            return

        # Always record: missing modules
        is_module_error = any(k in error_line for k in
                              ('ModuleNotFoundError', 'ImportError'))

        # Record AttributeError only for library-specific classes (not generic types)
        is_useful_attr_error = False
        if 'AttributeError' in error_line:
            is_useful_attr_error = not any(
                f"'{t}'" in error_line for t in self._GENERIC_TYPES)

        if not (is_module_error or is_useful_attr_error):
            return

        # Deduplicate
        for existing in self.error_patterns:
            if existing == error_line:
                return
        self.error_patterns.append(error_line)
        if len(self.error_patterns) > 15:
            self.error_patterns = self.error_patterns[-15:]

    def format_error_patterns(self, static_hints=None):
        """Format known pitfalls for CG context.

        Combines static hints (from pitfalls.txt) with runtime-discovered
        error patterns into a single section.
        """
        all_pitfalls = list(static_hints or []) + self.error_patterns
        if not all_pitfalls:
            return ""
        lines = [f"  - {p}" for p in all_pitfalls]
        return ("**Known pitfalls (avoid these):**\n"
                + "\n".join(lines))

    def format_all_questions(self, recent_full=30):
        """Question log with two-tier summarization.

        Last `recent_full` questions in full text for exact deduplication.
        Older questions as 120-char snippets (captures target variable + method).
        """
        if not self.all_questions:
            return ""

        total = len(self.all_questions)
        if total <= recent_full:
            lines = [f"  {i+1}. {q}" for i, q in enumerate(self.all_questions)]
            return ("**All questions investigated so far (do NOT repeat these):**\n"
                    + "\n".join(lines))

        older = self.all_questions[:-recent_full]
        recent = self.all_questions[-recent_full:]

        parts = [f"**All questions investigated ({total} total — do NOT repeat):**"]
        parts.append(f"  *Earlier questions ({len(older)}, summarised):*")
        for i, q in enumerate(older):
            snippet = q[:120].rstrip()
            parts.append(f"    {i+1}. {snippet}{'...' if len(q) > 120 else ''}")
        parts.append(f"  *Recent questions (full text):*")
        for i, q in enumerate(recent):
            parts.append(f"  {len(older)+i+1}. {q}")

        return "\n".join(parts)


class SimpleLogManager:
    """No-op log manager — satisfies interface without doing anything."""
    def write_to_log(self, *args, **kwargs): pass
    def charge_for_completed_query(self, *args, **kwargs): pass
    def consolidate_logs(self, *args, **kwargs): pass
    def print_summary_to_terminal(self, *args, **kwargs): pass


class ExplorationEngine:
    """
    Minimal runtime for AutoExplorer. Provides the self.engine.* interface.
    
    Usage:
        engine = ExplorationEngine(df, output_dir="output")
        from auto_explore import AutoExplorer
        explorer = AutoExplorer(engine)
        explorer.run("What patterns exist?", max_iterations=5)
    """

    def __init__(self, df, output_dir="output", agent_model=None, code_model=None,
                 continue_run=False):
        """
        Args:
            df: pandas DataFrame to analyze
            output_dir: directory for all output files
            agent_model: model for agents (evaluator, theorist, selector, etc.) — default: Haiku
            code_model: model for code generation and error correction — default: Opus
            continue_run: if True, preserve existing output directory and append
        """
        # DataFrame
        self.df = df.copy()
        self.df_id = "main"
        self.execution_mode = "local"
        self.api_client = None
        self.auxiliary_datasets = []

        # Output
        self.output_dir = output_dir
        self.output_manager = OutputManager(output_dir)
        self.webui = False

        # Models & LLM
        self.cost_tracker = CostTracker()
        self.run_logger = RunLogger(os.path.join(output_dir, "run_log.json"),
                                    append=continue_run)
        self.llm_client = LLMClient(cost_tracker=self.cost_tracker, run_logger=self.run_logger)
        self.models = SimpleModelManager(agent_model=agent_model, code_model=code_model)
        self.prompts = PromptManager()
        self.reasoning_models = []

        # State
        self.chain_id = int(time.time())
        self.thread_id = int(time.time())
        self.kill_signal = False
        self._stop_event = threading.Event()

        # Settings saved/restored by auto_explore run()
        self.user_feedback = False
        self.MAX_ERROR_CORRECTIONS = 3

        # Sub-components
        self.message_manager = SimpleMessageManager()
        self.log_and_call_manager = SimpleLogManager()
        self.executor = CodeExecutor()

        # Iteration context — set by auto_explore before each _process_question
        self._iteration = 0
        self._max_iterations = 0
        self._question = ""
        self.data_profile = ""  # Set by auto_explore after orientation

        # Output directory setup
        if continue_run:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "exploration"), exist_ok=True)
        else:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, "exploration"))
            # Save DataFrame for potential --continue later
            try:
                self.df.to_parquet(os.path.join(output_dir, "dataframe.parquet"))
            except Exception:
                # Mixed-type columns (e.g. mutation cols with int 0 and str '0') —
                # coerce object columns to string and retry
                for col in self.df.select_dtypes(include=['object']).columns:
                    self.df[col] = self.df[col].astype(str).replace('nan', None)
                self.df.to_parquet(os.path.join(output_dir, "dataframe.parquet"))

    # ──────────────────────────────────────────────
    # LLM interface (called by auto_explore agents)
    # ──────────────────────────────────────────────

    def llm_stream(self, prompts, log_and_call_manager, output_manager,
                   messages, agent=None, chain_id=None, tools=None,
                   reasoning_models=None, reasoning_effort="medium",
                   stop_event=None, model_override=None):
        """
        LLM call matching the signature auto_explore expects.
        Streams through the output_manager (which handles silent mode).
        """
        model = model_override or self.models.get_model_name(agent)[0]
        max_tokens = 24000 if agent == "Synthesis Generator" else 16000

        response = self.llm_client.stream(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            output_manager=output_manager,
            chain_id=chain_id,
            agent=agent,
        )
        return response

    # ──────────────────────────────────────────────
    # Shared LLM helpers (used by _process_question and run_orientation)
    # ──────────────────────────────────────────────

    def _get_fallback_model(self, agent):
        """Get the alternate model for cross-model fallback.

        If the agent normally uses the agent_model, returns the code_model,
        and vice versa. Returns None if both models are the same.
        """
        agent_model = self.models.agent_model
        code_model = self.models.code_model
        if agent_model == code_model:
            return None
        normal_model = self.models.get_model_name(agent)[0]
        return code_model if normal_model == agent_model else agent_model

    def _call_llm_for_code(self, messages, model, agent="Code Generator"):
        """Call LLM and extract code with retry + model fallback.

        Pattern: try → retry same model with nudge → fallback alternate model.
        Returns (code, llm_response) where code may be None if all attempts fail.
        """
        # Attempt 1: primary model
        try:
            llm_response = self.llm_client.call(
                messages=messages, model=model,
                max_tokens=16000, temperature=0, agent=agent,
            )
        except Exception as e:
            logger.warning(f"Code gen LLM call failed ({agent}): {e}")
            llm_response = ""

        code = extract_code(llm_response or "")
        if code:
            return code, llm_response

        # Attempt 2: retry same model with nudge
        retry_msg = (
            "Your previous response did not contain a ```python``` code block. "
            "Please return ONLY executable Python code inside ```python``` markers. "
            "The DataFrame `df` is pre-loaded. Do not explain — just provide the code."
        )
        retry_messages = messages + [
            {"role": "assistant", "content": llm_response or ""},
            {"role": "user", "content": retry_msg},
        ]
        try:
            llm_response = self.llm_client.call(
                messages=retry_messages, model=model,
                max_tokens=16000, temperature=0, agent=agent,
            )
        except Exception as e:
            logger.warning(f"Code gen retry failed ({agent}): {e}")
            llm_response = ""

        code = extract_code(llm_response or "")
        if code:
            return code, llm_response

        # Attempt 3: fallback to alternate model
        fallback_model = self._get_fallback_model(agent)
        if fallback_model:
            logger.info(f"Code gen falling back to {fallback_model}")
            self.output_manager.print_wrapper(
                style.error_msg(f"Retrying with fallback model..."),
                chain_id=self.chain_id,
            )
            try:
                llm_response = self.llm_client.call(
                    messages=messages, model=fallback_model,
                    max_tokens=16000, temperature=0, agent=agent,
                )
            except Exception as e:
                logger.warning(f"Code gen fallback failed ({agent}): {e}")
                llm_response = ""
            code = extract_code(llm_response or "")

        return code, llm_response or ""

    # ──────────────────────────────────────────────
    # _process_question (called by auto_explore per iteration)
    # ──────────────────────────────────────────────

    def _process_question(self, question, image=None, user_code=None, replay=None):
        """
        Simplified pipeline: question → code gen → exec → store results.
        Code generation runs silently. Only results and status are shown.
        """
        schema = self._get_df_schema()
        qa_context = self.message_manager.format_qa_pairs()

        # ── Code Generation (silent — user sees status, not raw LLM output) ──
        system_msg = self.prompts.code_generator_system
        user_msg = self.prompts.code_generator_user.format(
            schema=schema,
            qa_pairs=qa_context,
            error_patterns=self.message_manager.format_error_patterns(
                static_hints=self._load_pitfalls()),
            question=question,
        )
        # Inject analytical profile from orientation (if available)
        if self.data_profile:
            user_msg = user_msg.replace(
                "Previous findings from this exploration:",
                f"**Analytical Profile (from orientation — use for group sizes and confounders):**\n{self.data_profile}\n\nPrevious findings from this exploration:",
            )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        model, _ = self.models.get_model_name("Code Generator")
        self.output_manager.print_wrapper(
            style.agent("Code Generator", model), chain_id=self.chain_id
        )

        with style.spinner("Generating code"):
            code, llm_response = self._call_llm_for_code(messages, model)

        if not code:
            self.output_manager.print_wrapper(
                style.error_msg("No executable code generated"),
                chain_id=self.chain_id,
            )
            self.message_manager.code_exec_results = "Code generation produced no executable code."
            self.message_manager.last_code = ""
            # Still write an analysis.md marking the failure
            analysis_dir = self._analysis_dir_for_chain(self.chain_id)
            os.makedirs(analysis_dir, exist_ok=True)
            self.output_manager.write_analysis_md(analysis_dir, question, "(none)", None, "No code generated", [], self._iteration, self._max_iterations, self.chain_id)
            self.message_manager.append_qa_pair(question, "Code generation failed — no executable code produced.", chain_id=self.chain_id)
            return

        code_lines = len(code.strip().split('\n'))

        # ── Build analysis directory ──
        analysis_dir = self._analysis_dir_for_chain(self.chain_id)
        os.makedirs(analysis_dir, exist_ok=True)

        # ── Execute ──
        with style.spinner("Executing code"):
            results, error, plots = self.executor.execute(code, self.df, analysis_dir)

        # ── Error Correction Loop (silent) ──
        initial_error = error  # preserve for error pattern recording
        retries = 0
        while error and retries < self.MAX_ERROR_CORRECTIONS:
            retries += 1
            self.output_manager.print_wrapper(
                style.error_msg(f"Retry {retries}/{self.MAX_ERROR_CORRECTIONS}: {error.strip().split(chr(10))[-1][:120]}"),
                chain_id=self.chain_id,
            )

            # Trim old attempts after 2 retries
            if retries > 2 and len(messages) >= 6:
                del messages[2]
                del messages[2]

            if retries == 1:
                fix_msg = self.prompts.error_corrector.format(error=error, schema=schema)
            else:
                fix_msg = (
                    f"Still failing. Error:\n{error}\n\n"
                    "Return the complete corrected code within ```python``` blocks. "
                    "The DataFrame `df` is pre-loaded. Include all imports."
                )

            messages.append({"role": "assistant", "content": llm_response or ""})
            messages.append({"role": "user", "content": fix_msg})

            with style.spinner("Fixing code"):
                llm_response = self.llm_client.call(
                    messages=messages,
                    model=model,
                    max_tokens=16000,
                    temperature=0,
                    agent="Error Corrector",
                )
            code = extract_code(llm_response or "")
            if code:
                code_lines = len(code.strip().split('\n'))
                results, error, plots = self.executor.execute(code, self.df, analysis_dir)
            else:
                break

        # Record successfully-corrected error patterns for future CG context
        if retries > 0 and results and initial_error:
            self.message_manager.record_error_pattern(initial_error)

        # ── Status line ──
        if results:
            self.output_manager.print_wrapper(
                style.success(f"{code_lines} lines, executed OK"),
                chain_id=self.chain_id,
            )
        elif error:
            self.output_manager.print_wrapper(
                style.error_msg(f"Failed after {retries + 1} attempt(s)"),
                chain_id=self.chain_id,
            )

        # ── Store results ──
        self.message_manager.code_exec_results = results if results else f"Execution failed: {error}"
        self.message_manager.last_code = code

        # ── Terminal display ──
        if results:
            # Try to show just the summary block
            start_marker = "###RESULTS_START###"
            end_marker = "###RESULTS_END###"
            s_idx = results.find(start_marker)
            e_idx = results.find(end_marker)

            if s_idx >= 0 and e_idx > s_idx:
                # Show the summary block
                summary = results[s_idx + len(start_marker):e_idx].strip()
                self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)
                for line in summary.split('\n'):
                    self.output_manager.print_wrapper(style.result_line(line), chain_id=self.chain_id)
                self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)
            else:
                # No markers — show truncated raw output
                lines = results.strip().split('\n')
                self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)
                if len(lines) > STDOUT_TRUNCATE_LINES:
                    for line in lines[:STDOUT_TRUNCATE_LINES]:
                        self.output_manager.print_wrapper(style.result_line(line), chain_id=self.chain_id)
                    self.output_manager.print_wrapper(
                        style.result_line(style.dim(f"... ({len(lines) - STDOUT_TRUNCATE_LINES} more lines)")),
                        chain_id=self.chain_id,
                    )
                else:
                    for line in lines:
                        self.output_manager.print_wrapper(style.result_line(line), chain_id=self.chain_id)
                self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)

        # ── Write analysis.md ──
        plot_note = f"{len(plots)} plot{'s' if len(plots) != 1 else ''}" if plots else ""
        self.output_manager.write_analysis_md(analysis_dir, question, code, results, error, plots, self._iteration, self._max_iterations, self.chain_id)
        rel_path = os.path.relpath(analysis_dir, ".")
        self.output_manager.print_wrapper(
            style.file_ref(f"{rel_path}/analysis.md", plot_note),
            chain_id=self.chain_id,
        )

        # ── Append QA pair ──
        self.message_manager.append_qa_pair(
            question,
            results if results else f"Execution failed: {str(error)[:300]}",
            chain_id=self.chain_id,
        )

    # ──────────────────────────────────────────────
    # Orientation (data profiling)
    # ──────────────────────────────────────────────

    def run_orientation(self, seed_question, model_override=None):
        """Run the orientation phase: profile the dataset's analytical landscape.

        Returns the data profile string, or empty string on failure.
        Uses the same code generation + retry + fallback pattern as _process_question.
        """
        schema = self._get_df_schema()

        system_msg = self.prompts.orientation_system
        user_msg = self.prompts.orientation_user.format(
            schema=schema,
            seed_question=seed_question,
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        model = model_override or self.models.get_model_name("Code Generator")[0]
        self.output_manager.print_wrapper(
            style.agent("Orientation", model), chain_id=self.chain_id
        )

        # Code generation: try → retry with nudge → model fallback (shared method)
        with style.spinner("Profiling dataset"):
            code, llm_response = self._call_llm_for_code(messages, model)

        if not code:
            self.output_manager.print_wrapper(
                style.error_msg("Orientation: no code generated"),
                chain_id=self.chain_id,
            )
            return ""

        # Execute
        analysis_dir = os.path.join(self.output_dir, "orientation")
        os.makedirs(analysis_dir, exist_ok=True)
        results, error, plots = self.executor.execute(code, self.df, analysis_dir)

        # Error correction loop (same pattern as _process_question)
        initial_error = error
        retries = 0
        while error and retries < self.MAX_ERROR_CORRECTIONS:
            retries += 1
            self.output_manager.print_wrapper(
                style.error_msg(f"Retry {retries}/{self.MAX_ERROR_CORRECTIONS}: {error.strip().split(chr(10))[-1][:120]}"),
                chain_id=self.chain_id,
            )

            if retries > 2 and len(messages) >= 6:
                del messages[2]
                del messages[2]

            if retries == 1:
                fix_msg = self.prompts.error_corrector.format(error=error, schema=schema)
            else:
                fix_msg = (
                    f"Still failing. Error:\n{error}\n\n"
                    "Return the complete corrected code within ```python``` blocks. "
                    "The DataFrame `df` is pre-loaded. Include all imports."
                )

            messages.append({"role": "assistant", "content": llm_response or ""})
            messages.append({"role": "user", "content": fix_msg})

            with style.spinner("Fixing orientation code"):
                try:
                    llm_response = self.llm_client.call(
                        messages=messages,
                        model=model,
                        max_tokens=16000,
                        temperature=0,
                        agent="Error Corrector",
                    )
                except Exception as e:
                    logger.warning(f"Orientation error correction failed: {e}")
                    break
            code = extract_code(llm_response or "")
            if code:
                results, error, plots = self.executor.execute(code, self.df, analysis_dir)
            else:
                break

        # Record successfully-corrected error patterns
        if retries > 0 and results and initial_error:
            self.message_manager.record_error_pattern(initial_error)

        if not results:
            self.output_manager.print_wrapper(
                style.error_msg("Orientation failed — continuing without data profile"),
                chain_id=self.chain_id,
            )
            return ""

        # Extract PROFILE block
        profile = ""
        start_marker = "###PROFILE_START###"
        end_marker = "###PROFILE_END###"
        s_idx = results.find(start_marker)
        e_idx = results.find(end_marker)

        if s_idx >= 0 and e_idx > s_idx:
            profile = results[s_idx + len(start_marker):e_idx].strip()
        else:
            # Fall back to RESULTS markers or raw output
            s_idx = results.find("###RESULTS_START###")
            e_idx = results.find("###RESULTS_END###")
            if s_idx >= 0 and e_idx > s_idx:
                profile = results[s_idx + len("###RESULTS_START###"):e_idx].strip()
            else:
                profile = results.strip()[:3000]

        # Truncate if excessively long (target ~500-1000 tokens)
        if len(profile) > 4000:
            profile = profile[:4000] + "\n[...truncated]"

        # Display
        code_lines = len(code.strip().split('\n'))
        self.output_manager.print_wrapper(
            style.success(f"Orientation: {code_lines} lines, profile {len(profile)} chars"),
            chain_id=self.chain_id,
        )
        self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)
        for line in profile.split('\n')[:25]:
            self.output_manager.print_wrapper(style.result_line(line), chain_id=self.chain_id)
        if profile.count('\n') > 25:
            self.output_manager.print_wrapper(
                style.result_line(style.dim(f"... ({profile.count(chr(10)) - 25} more lines)")),
                chain_id=self.chain_id,
            )
        self.output_manager.print_wrapper(style.result_border(), chain_id=self.chain_id)

        # Write analysis.md for the orientation
        self.output_manager.write_analysis_md(analysis_dir, "ORIENTATION: Dataset analytical profile", code, results, error, plots, self._iteration, self._max_iterations, self.chain_id)

        return profile

    # ──────────────────────────────────────────────
    # File Writing
    # ──────────────────────────────────────────────

    def _analysis_dir_for_chain(self, chain_id):
        """Build the analysis directory path for a given chain_id."""
        iter_dir = f"{self._iteration:02d}"
        return os.path.join(self.output_dir, "exploration", iter_dir, str(chain_id))

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def _get_df_schema(self):
        """Rich DataFrame schema for LLM context — includes sample values for reliable code gen."""
        parts = []
        parts.append(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")

        parts.append("Columns (with sample values):")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            nulls = self.df[col].isna().sum()
            nunique = self.df[col].nunique()
            null_pct = f", {nulls} nulls ({100*nulls/len(self.df):.0f}%)" if nulls > 0 else ""

            # Show sample non-null values so the LLM knows what the data looks like
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                is_string = dtype == 'object' or 'str' in str(dtype).lower() or str(dtype) == 'category'
                if is_string or nunique <= 20:
                    samples = non_null.unique()[:6].tolist()
                    sample_str = f"  samples: {samples}"
                else:
                    mn, mx = non_null.min(), non_null.max()
                    sample_str = f"  range: [{mn}, {mx}]"
            else:
                sample_str = "  (all null)"

            parts.append(f"  - {col} ({dtype}, {nunique} unique{null_pct})")
            parts.append(f"    {sample_str}")

        # For narrow datasets (≤40 cols), include head() and describe() — they're
        # readable and help the code model understand data format and distributions.
        # For wide datasets (>40 cols), these become unreadable walls of text that
        # consume most of the context window while adding little value — the column
        # schema already provides types, nulls, ranges, and sample values.
        if len(self.df.columns) <= 40:
            parts.append(f"\nFirst 5 rows:\n{self.df.head(5).to_string()}")
            parts.append(f"\nNumeric summary:\n{self.df.describe().to_string()}")
        return "\n".join(parts)

    def _get_df_schema_slim(self):
        """Lightweight schema for non-code agents (QG, Selector, Evaluator).

        Lists column names, types, and categorical values only — no head() or
        describe() output. Roughly 5-6x smaller than the full schema."""
        parts = []
        parts.append(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
        parts.append("Columns:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            nunique = self.df[col].nunique()
            nulls = self.df[col].isna().sum()
            null_pct = f", {nulls} nulls ({100*nulls/len(self.df):.0f}%)" if nulls > 0 else ""
            is_string = dtype == 'object' or str(dtype) == 'category'
            if is_string or nunique <= 20:
                samples = self.df[col].dropna().unique()[:6].tolist()
                parts.append(f"  - {col} ({dtype}, {nunique} unique{null_pct}) values: {samples}")
            else:
                parts.append(f"  - {col} ({dtype}, {nunique} unique{null_pct})")
        return "\n".join(parts)

    def _get_column_list(self):
        """Compact column list for Research Interpreter context.

        For narrow datasets (≤50 cols): comma-separated names.
        For wide datasets: split by coverage tier so the RI can distinguish
        usable columns from sparse ones without a wall of text.
        """
        cols = self.df.columns.tolist()
        if len(cols) <= 50:
            return ', '.join(cols)
        # Wide dataset — group by coverage tier
        high = [c for c in cols if self.df[c].notna().mean() >= 0.5]
        low = [c for c in cols if self.df[c].notna().mean() < 0.5]
        parts = [f"Columns with ≥50% coverage ({len(high)}): {', '.join(high)}"]
        if low:
            parts.append(f"Sparse columns <50% coverage ({len(low)}): {', '.join(low)}")
        return '\n'.join(parts)

    @staticmethod
    def _load_pitfalls(filename='pitfalls.txt'):
        """Load static code hints from pitfalls.txt.

        User-maintained file with one hint per line. Lines starting with #
        are comments. Re-read on each code generation call so edits during
        a run take effect on the next iteration.
        """
        if not os.path.exists(filename):
            return []
        try:
            with open(filename) as f:
                return [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
        except Exception:
            return []

    def store_dataset_details_in_db(self):
        """No-op — no database in minimal version."""
        pass
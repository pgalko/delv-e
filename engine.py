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
        # Complete log of ALL questions ever asked (never truncated).
        # Cheap in tokens (~50 per question) and prevents circular exploration.
        self.all_questions = []
        # Full stdout per chain_id — used ONLY by synthesis (never truncated).
        # Loop agents use the capped tree node result_summary instead.
        self.full_results_store = {}
        # auto_explore saves/restores this in run() setup/teardown
        self.select_analyst_messages = [{"content": ""}]

    def reset_non_cumul_messages(self):
        self.code_exec_results = None
        self.last_code = None
        self.last_plan = None

    def restore_interaction(self, thread_id, chain_id):
        """No-op — context comes from QA pairs + research model."""
        pass

    def store_interaction(self, *args, **kwargs):
        pass

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

    def format_qa_pairs(self, max_qa_pairs=20, include_chain_id=False):
        if not self.qa_pairs:
            return "(No previous analyses)"
        pairs = self.qa_pairs[-max_qa_pairs:]
        parts = []
        for qa in pairs:
            ref = f" [[{qa['chain_id']}]]" if include_chain_id and qa.get('chain_id') else ""
            result = str(qa['result'] or 'No results')

            # Try to extract the delimited summary block
            start_marker = "###RESULTS_START###"
            end_marker = "###RESULTS_END###"
            start_idx = result.find(start_marker)
            end_idx = result.find(end_marker)

            if start_idx >= 0 and end_idx > start_idx:
                # Extract just the summary block (most information-dense)
                result = result[start_idx + len(start_marker):end_idx].strip()
            elif len(result) > 1500:
                # No markers found — keep first + last
                result = result[:750] + "\n[...]\n" + result[-750:]

            parts.append(f"Q: {qa['question']}{ref}\nResult:\n{result}")
        return "\n\n---\n\n".join(parts)

    def format_all_questions(self):
        """Complete list of every question asked. Cheap in tokens, prevents circular exploration."""
        if not self.all_questions:
            return ""
        lines = [f"  {i+1}. {q}" for i, q in enumerate(self.all_questions)]
        return "**All questions investigated so far (do NOT repeat these):**\n" + "\n".join(lines)


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
        self.output_manager = OutputManager()
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
        self.branch_endpoints = []
        self.exploration_trajectory = None

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
        self._phase = "MAPPING"
        self._question = ""

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
        max_tokens = 24000 if agent == "Synthesis Generator" else 10000

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
            question=question,
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
            llm_response = self.llm_client.call(
                messages=messages,
                model=model,
                max_tokens=10000,
                temperature=0,
                agent="Code Generator",
            )

        code = extract_code(llm_response)

        # If no code was extracted, retry with a more explicit prompt
        if not code:
            retry_msg = (
                "Your previous response did not contain a ```python``` code block. "
                "Please return ONLY executable Python code inside ```python``` markers. "
                "The DataFrame `df` is pre-loaded. Do not explain — just provide the code."
            )
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": retry_msg})

            with style.spinner("Retrying code generation"):
                llm_response = self.llm_client.call(
                    messages=messages,
                    model=model,
                    max_tokens=10000,
                    temperature=0,
                    agent="Code Generator",
                )
            code = extract_code(llm_response)

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
            self._write_analysis_md(analysis_dir, question, "(none)", None, "No code generated", [])
            self.message_manager.append_qa_pair(question, "Code generation failed — no executable code produced.", chain_id=self.chain_id)
            return

        code_lines = len(code.strip().split('\n'))

        # ── Build analysis directory ──
        analysis_dir = self._analysis_dir_for_chain(self.chain_id)
        os.makedirs(analysis_dir, exist_ok=True)

        # ── Execute ──
        results, error, plots = self.executor.execute(code, self.df, analysis_dir)

        # ── Error Correction Loop (silent) ──
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

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": fix_msg})

            with style.spinner("Fixing code"):
                llm_response = self.llm_client.call(
                    messages=messages,
                    model=model,
                    max_tokens=10000,
                    temperature=0,
                    agent="Error Corrector",
                )
            code = extract_code(llm_response)
            if code:
                code_lines = len(code.strip().split('\n'))
                results, error, plots = self.executor.execute(code, self.df, analysis_dir)
            else:
                break

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
        self._write_analysis_md(analysis_dir, question, code, results, error, plots)
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
    # File Writing
    # ──────────────────────────────────────────────

    def _analysis_dir_for_chain(self, chain_id):
        """Build the analysis directory path for a given chain_id."""
        iter_dir = f"{self._iteration:02d}_{self._phase}"
        return os.path.join(self.output_dir, "exploration", iter_dir, str(chain_id))

    def _write_analysis_md(self, analysis_dir, question, code, results, error, plots):
        """Write analysis.md with code, output, and embedded plots."""
        md_parts = [f"# {question}\n"]
        md_parts.append(
            f"| Field | Value |\n|-------|-------|\n"
            f"| Iteration | {self._iteration} of {self._max_iterations} |\n"
            f"| Phase | {self._phase} |\n"
            f"| Chain ID | {self.chain_id} |\n"
        )

        md_parts.append("\n## Code\n")
        md_parts.append(f"```python\n{code}\n```\n")

        md_parts.append("\n## Output\n")
        if results:
            md_parts.append(f"```\n{results}\n```\n")
        elif error:
            md_parts.append(f"```\nEXECUTION ERROR:\n{error}\n```\n")
        else:
            md_parts.append("```\nNo output produced.\n```\n")

        if plots:
            md_parts.append("\n## Plots\n")
            for plot_path in plots:
                fname = os.path.basename(plot_path)
                md_parts.append(f"![{fname}]({fname})\n")

        md_path = os.path.join(analysis_dir, "analysis.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_parts))

    def write_iteration_summary(self, iteration, phase, solutions_data, scores,
                                 selected_index, model_impact, contradiction,
                                 thread_completed, new_phase, old_phase,
                                 new_questions=None, selected_questions=None):
        """Write _summary.md for an iteration. Called from auto_explore.run()."""
        iter_dir = f"{iteration:02d}_{phase}"
        summary_dir = os.path.join(self.output_dir, "exploration", iter_dir)
        os.makedirs(summary_dir, exist_ok=True)

        parts = [f"# Iteration {iteration} — {phase}\n"]

        # Solutions table
        parts.append("## Solutions Evaluated\n")
        parts.append("| # | Chain ID | Question | Score |")
        parts.append("|---|----------|----------|-------|")
        for i, sol in enumerate(solutions_data):
            s = scores[i] if i < len(scores) else "?"
            marker = " ✓" if i == selected_index else ""
            cid = sol.get('chain_id', '?')
            parts.append(f"| {i+1}{marker} | [{cid}]({cid}/analysis.md) | {sol['question'][:80]} | {s}/10 |")
        parts.append("")

        # Model update
        parts.append("## Research Model Update\n")
        parts.append(f"- **Model Impact:** {model_impact}")
        parts.append(f"- **Contradiction:** {'Yes' if contradiction else 'No'}")
        parts.append(f"- **Thread Completed:** {'Yes' if thread_completed else 'No'}")
        parts.append("")

        # Phase
        if new_phase != old_phase:
            parts.append(f"## Phase Transition\n\n{old_phase} → {new_phase}\n")
        else:
            parts.append(f"## Phase: {new_phase} (maintained)\n")

        # Questions
        if new_questions:
            parts.append("## Questions Generated\n")
            for i, q in enumerate(new_questions, 1):
                parts.append(f"{i}. {q[:120]}")
            parts.append("")

        if selected_questions:
            parts.append("**Selected for next iteration:** " + ", ".join(
                q[:60] + "..." for q in selected_questions
            ))

        path = os.path.join(summary_dir, "_summary.md")
        with open(path, "w") as f:
            f.write("\n".join(parts))

    def write_final_outputs(self, research_model, phase_history, synthesis_text=None):
        """Write final research model, synthesis report, and cost summary."""
        # Research model
        model_path = os.path.join(self.output_dir, "research_model.md")
        with open(model_path, "w") as f:
            f.write("# Final Research Model\n\n")
            f.write(research_model or "(empty)")
            if phase_history:
                f.write("\n\n## Phase History\n\n")
                for iteration, old, new in phase_history:
                    f.write(f"- Iteration {iteration}: {old} → {new}\n")

        # Synthesis report
        if synthesis_text:
            synth_path = os.path.join(self.output_dir, "synthesis_report.md")
            with open(synth_path, "w") as f:
                f.write(synthesis_text)

        # Cost summary
        cost_path = os.path.join(self.output_dir, "cost.txt")
        with open(cost_path, "w") as f:
            f.write(self.cost_tracker.report() + "\n")
            if self.run_logger:
                agent_summary = self.run_logger.summary()
                if agent_summary:
                    f.write("\n" + agent_summary + "\n")

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

    def store_dataset_details_in_db(self):
        """No-op — no database in minimal version."""
        pass
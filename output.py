"""
Print-based output manager for terminal display.
Satisfies the output_manager interface that auto_explore.py expects.
"""

import sys
import style


class OutputManager:
    def __init__(self):
        self.silent_mode = False
        self._captured_output = []

    def print_wrapper(self, message, end="\n", flush=False, chain_id=None, thought=False):
        message = str(message)
        if self.silent_mode:
            self._captured_output.append(message)
            if end:
                self._captured_output.append(end)
            return
        print(message, end=end, flush=flush)

    def display_system_messages(self, message, chain_id=None):
        if self.silent_mode:
            return
        print(f"  {style.DIM}{message}{style.RESET}")

    def display_tool_start(self, agent_name, model, chain_id=None):
        if self.silent_mode:
            return
        print(style.agent(agent_name, model))

    def display_results(self, chain_id=None, **kwargs):
        if self.silent_mode:
            return
        if 'answer' in kwargs and kwargs['answer']:
            print(kwargs['answer'])

    def display_error(self, error, chain_id=None):
        if self.silent_mode:
            return
        print(style.error_msg(str(error)[:200]), file=sys.stderr)

    def display_tool_info(self, action, action_input, chain_id=None):
        if self.silent_mode:
            return

    def set_silent(self, silent):
        self.silent_mode = silent
        if silent:
            self._captured_output = []

    def get_captured_output(self):
        output = ''.join(self._captured_output)
        self._captured_output = []
        return output

    def send_chain_id(self, *args, **kwargs):
        pass

    def send_synthesis_image(self, *args, **kwargs):
        pass
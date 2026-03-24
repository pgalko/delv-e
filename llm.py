"""
Multi-provider LLM client with streaming, cost tracking, and run logging.

Supports Anthropic, OpenAI, Ollama, and OpenRouter via provider:model syntax:
    anthropic:claude-opus-4-6
    openai:gpt-4o
    ollama:qwen3:30b
    openrouter:google/gemini-2.5-flash

Provider classes each implement call() and stream() with identical signatures.
LLMClient routes to the correct provider based on the model string.
"""

import json
import os
import time

from logger_config import get_logger
logger = get_logger(__name__)


# ══════════════════════════════════════════════════
# PRICING
# ══════════════════════════════════════════════════

# USD per million tokens — update when pricing changes
PRICING = {
    # Anthropic
    "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":            {"input": 5.00, "output": 25.00},
    # OpenAI
    "gpt-5.4":                    {"input": 2.50, "output": 15.00},
    "gpt-5.3-codex":              {"input": 1.75, "output": 14.00},
    "gpt-5-mini":                 {"input": 0.25, "output": 2.00},
    "o4-mini":                    {"input": 1.10, "output": 4.40},
    # OpenRouter — varies by model, add entries as needed
    # Pricing: https://openrouter.ai/models
    "moonshotai/kimi-k2.5":       {"input": 0.45, "output": 2.20},
    "z-ai/glm-5":                 {"input": 0.72, "output": 2.30},
    "deepseek/deepseek-v3.2":     {"input": 0.26, "output": 0.38},
    "qwen/qwen3.5-397b-a17b":     {"input": 0.39, "output": 2.34},
    "minimax/minimax-m2.7":       {"input": 0.30, "output": 1.20},
    # Ollama (local, free)
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def compute_cost(model, input_tokens, output_tokens):
    """Compute USD cost for a single API call."""
    # Strip provider prefix if present (e.g. 'openrouter:moonshotai/kimi-k2.5' → 'moonshotai/kimi-k2.5')
    model_name = model
    if ':' in model:
        parts = model.split(':', 1)
        if parts[0] in ('anthropic', 'openai', 'ollama', 'openrouter'):
            model_name = parts[1]
    pricing = PRICING.get(model_name, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000)


# ══════════════════════════════════════════════════
# PROVIDERS
# ══════════════════════════════════════════════════

class AnthropicProvider:
    """Anthropic Claude API."""

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found. Set it in .env or environment.")
        self.client = anthropic.Client(api_key=api_key)
        self._api_error = anthropic.APIError

    def call(self, messages, model, max_tokens, temperature):
        system_msg, api_messages = self._split_system(messages)
        try:
            response = self.client.messages.create(
                model=model, system=system_msg, messages=api_messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        except self._api_error as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text or ""
        return (
            content,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        system_msg, api_messages = self._split_system(messages)
        collected = []
        try:
            with self.client.messages.stream(
                model=model, system=system_msg, messages=api_messages,
                max_tokens=max_tokens, temperature=temperature,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type') and event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            collected.append(event.delta.text)
                            on_token(event.delta.text)
                final = stream.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens
        except self._api_error as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        return ''.join(collected), input_tokens, output_tokens

    def _split_system(self, messages):
        """Anthropic requires system message separate from conversation."""
        system = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation.append(msg)
        return system, conversation


class OpenAIProvider:
    """OpenAI Responses API (GPT-5.x, Codex, o-series, GPT-4o, etc.)."""

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found. Set it in .env or environment.")
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _split_system(self, messages):
        """Extract system/instructions from messages for the Responses API."""
        instructions = None
        input_messages = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_messages.append(msg)
        return instructions, input_messages

    def call(self, messages, model, max_tokens, temperature):
        instructions, input_messages = self._split_system(messages)
        params = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if instructions:
            params["instructions"] = instructions
        response = self.client.responses.create(**params)
        return (
            response.output_text or "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        instructions, input_messages = self._split_system(messages)
        params = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        collected = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.responses.create(**params)
        for event in stream:
            if event.type == "response.output_text.delta":
                collected.append(event.delta)
                on_token(event.delta)
            elif event.type == "response.completed":
                if event.response and event.response.usage:
                    input_tokens = event.response.usage.input_tokens
                    output_tokens = event.response.usage.output_tokens
        return ''.join(collected), input_tokens, output_tokens


class OllamaProvider:
    """Ollama inference via OpenAI-compatible SDK.

    Supports both local models and cloud models (e.g. gpt-oss:120b-cloud).
    Cloud models are offloaded automatically by the local Ollama server —
    just run 'ollama signin' and 'ollama pull <model>' first.
    """

    def __init__(self):
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")

    def call(self, messages, model, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
        return (
            content,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        collected = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected.append(text)
                on_token(text)
        return ''.join(collected), input_tokens, output_tokens


class OpenRouterProvider:
    """OpenRouter API — access hundreds of models via OpenAI-compatible endpoint.

    Uses the OpenAI SDK pointed at OpenRouter's base URL.
    Model names use OpenRouter's format: 'google/gemini-2.5-flash',
    'deepseek/deepseek-chat-v3', etc.

    Usage: openrouter:google/gemini-2.5-flash
    Docs:  https://openrouter.ai/docs/quickstart
    """

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPEN_ROUTER_API_KEY not found. Set it in .env or environment.")
        base_url = os.environ.get("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/pgalko/delv-e",
                "X-Title": "delv-e",
            },
        )

    def call(self, messages, model, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
        return (
            content,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def stream(self, messages, model, max_tokens, temperature, on_token):
        collected = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected.append(text)
                on_token(text)
        return ''.join(collected), input_tokens, output_tokens


# ══════════════════════════════════════════════════
# PROVIDER REGISTRY
# ══════════════════════════════════════════════════

PROVIDER_CLASSES = {
    "anthropic":    AnthropicProvider,
    "openai":       OpenAIProvider,
    "ollama":       OllamaProvider,
    "openrouter":   OpenRouterProvider,
}

DEFAULT_PROVIDER = "anthropic"


def parse_model_string(model_string):
    """
    Parse 'provider:model' syntax. Returns (provider_name, model_name).
    If no provider prefix, returns (DEFAULT_PROVIDER, model_string).
    
    Examples:
        'anthropic:claude-opus-4-6'             -> ('anthropic', 'claude-opus-4-6')
        'openai:gpt-4o'                         -> ('openai', 'gpt-4o')
        'ollama:qwen3:30b'                      -> ('ollama', 'qwen3:30b')
        'openrouter:google/gemini-2.5-flash'    -> ('openrouter', 'google/gemini-2.5-flash')
        'claude-opus-4-6'                       -> ('anthropic', 'claude-opus-4-6')
    """
    parts = model_string.split(":", 1)
    if len(parts) == 2 and parts[0] in PROVIDER_CLASSES:
        return parts[0], parts[1]
    return DEFAULT_PROVIDER, model_string


# ══════════════════════════════════════════════════
# COST TRACKING & LOGGING (unchanged)
# ══════════════════════════════════════════════════

class CostTracker:
    """Accumulates token counts and cost across all calls in a run."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def record(self, input_tokens, output_tokens, model=None):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += compute_cost(model or "", input_tokens, output_tokens)

    def report(self):
        return (
            f"{self.calls} API calls | "
            f"{self.input_tokens:,} input + {self.output_tokens:,} output tokens | "
            f"${self.total_cost:.4f}"
        )


class RunLogger:
    """Append-only run log. Flushes to disk after each call."""

    def __init__(self, path, append=False):
        self.path = path
        self.entries = []
        if append and os.path.exists(path):
            try:
                with open(path) as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def log(self, agent, model, messages, response,
            input_tokens, output_tokens, elapsed_time):
        cost = compute_cost(model, input_tokens, output_tokens)
        self.entries.append({
            "agent": agent,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "elapsed_time_s": round(elapsed_time, 2),
            "tokens_per_second": round(output_tokens / max(elapsed_time, 0.01), 1),
            "cost_usd": round(cost, 6),
            "input": messages,
            "output": response,
        })
        self._flush()

    def _flush(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.entries, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write run log: {e}")

    def summary(self):
        if not self.entries:
            return ""
        agent_stats = {}
        for e in self.entries:
            a = e["agent"] or "unknown"
            if a not in agent_stats:
                agent_stats[a] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0, "time": 0.0}
            agent_stats[a]["calls"] += 1
            agent_stats[a]["input"] += e["input_tokens"]
            agent_stats[a]["output"] += e["output_tokens"]
            agent_stats[a]["cost"] += e["cost_usd"]
            agent_stats[a]["time"] += e["elapsed_time_s"]

        lines = ["Per-agent breakdown:"]
        for a, s in sorted(agent_stats.items()):
            lines.append(
                f"  {a:25s} {s['calls']:3d} calls | "
                f"{s['input']:>8,} in + {s['output']:>7,} out | "
                f"{s['time']:>6.1f}s | ${s['cost']:.4f}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════
# LLM CLIENT
# ══════════════════════════════════════════════════

class LLMClient:
    """
    Multi-provider LLM client. Routes calls to the correct provider
    based on 'provider:model' syntax in the model string.
    
    Providers are initialized lazily on first use.
    """

    def __init__(self, cost_tracker=None, run_logger=None):
        self.cost_tracker = cost_tracker or CostTracker()
        self.run_logger = run_logger
        self._providers = {}  # lazily initialized

    def _get_provider(self, provider_name):
        """Get or initialize a provider instance."""
        if provider_name not in self._providers:
            if provider_name not in PROVIDER_CLASSES:
                raise ValueError(
                    f"Unknown provider '{provider_name}'. "
                    f"Available: {', '.join(PROVIDER_CLASSES.keys())}"
                )
            self._providers[provider_name] = PROVIDER_CLASSES[provider_name]()
        return self._providers[provider_name]

    def call(self, messages, model, max_tokens=10000, temperature=0, agent=None):
        """Non-streaming call. Returns response text (always a string, never None)."""
        provider_name, model_name = parse_model_string(model)
        provider = self._get_provider(provider_name)
        start_time = time.time()

        try:
            content, input_tokens, output_tokens = provider.call(
                messages, model_name, max_tokens, temperature
            )
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            raise

        # Guarantee string return — providers should already handle this,
        # but belt-and-suspenders against None leaking through.
        content = content or ""

        elapsed = time.time() - start_time
        self.cost_tracker.record(input_tokens, output_tokens, model_name)

        if self.run_logger:
            self.run_logger.log(agent, f"{provider_name}:{model_name}", messages,
                                content, input_tokens, output_tokens, elapsed)
        return content

    def stream(self, messages, model, max_tokens=10000, temperature=0,
               output_manager=None, chain_id=None, agent=None):
        """Streaming call. Returns full response text."""
        provider_name, model_name = parse_model_string(model)
        provider = self._get_provider(provider_name)
        start_time = time.time()

        def on_token(text):
            if output_manager:
                output_manager.print_wrapper(text, end='', flush=True, chain_id=chain_id)

        try:
            content, input_tokens, output_tokens = provider.stream(
                messages, model_name, max_tokens, temperature, on_token
            )
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            raise

        if output_manager and not output_manager.silent_mode:
            output_manager.print_wrapper("", chain_id=chain_id)

        elapsed = time.time() - start_time
        self.cost_tracker.record(input_tokens, output_tokens, model_name)

        if self.run_logger:
            self.run_logger.log(agent, f"{provider_name}:{model_name}", messages,
                                content, input_tokens, output_tokens, elapsed)
        return content
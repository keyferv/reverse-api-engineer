"""DeepSeek SDK implementation for API reverse engineering.

Uses OpenAI-compatible API via DeepSeek's endpoint.
Supports reasoning models (deepseek-reasoner) with reasoning_content.
"""

import os
from typing import Any

from openai import AsyncOpenAI

from .base_engineer import BaseEngineer


class DeepSeekEngineer(BaseEngineer):
    """Uses DeepSeek to analyze HAR files and generate API scripts."""

    def __init__(
        self,
        *args: Any,
        deepseek_model: str = "deepseek-chat",
        **kwargs: Any,
    ) -> None:
        """Initialize DeepSeek engineer.

        Args:
            deepseek_model: Model ID (e.g., "deepseek-chat", "deepseek-reasoner")
        """
        super().__init__(*args, **kwargs)

        self.deepseek_model = deepseek_model
        if not self.model:
            self.model = deepseek_model

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in environment")

        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
        )

    async def analyze_and_generate(self) -> dict[str, Any] | None:
        """Run the reverse engineering analysis with DeepSeek.

        Returns:
            Dict with script_path and usage, or None on error.
        """
        self.ui.header(self.run_id, self.prompt, self.model, self.sdk, mode="engineer")
        self.ui.start_analysis()

        system_prompt, user_message = self._build_prompts()
        self.message_store.save_prompt(user_message)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Track reasoning content for deepseek-reasoner
            reasoning_tokens = 0
            accumulated_reasoning: list[str] = []

            stream = await self._create_completion_with_retry(
                model=self.deepseek_model,
                messages=messages,
            )

            accumulated_content: list[str] = []
            usage_metadata: dict[str, int] = {}

            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        accumulated_content.append(delta.content)
                        self.ui.thinking(delta.content)
                        self.message_store.save_thinking(delta.content)
                    # Capture reasoning_content for deepseek-reasoner
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        accumulated_reasoning.append(delta.reasoning_content)
                        reasoning_tokens += len(delta.reasoning_content.split())

                if chunk.usage:
                    usage_metadata = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }

            # Add reasoning tokens to usage if present
            if reasoning_tokens > 0:
                usage_metadata["reasoning_tokens"] = reasoning_tokens

            full_response = "".join(accumulated_content)

            script_path = self.scripts_dir / self._get_client_filename()
            script_path.write_text(full_response)
            self.message_store.save_result({"script_path": str(script_path)})

            if usage_metadata:
                self.usage_metadata.update(usage_metadata)
                input_tokens = usage_metadata.get("input_tokens", 0)
                output_tokens = usage_metadata.get("output_tokens", 0)

                from .pricing import calculate_cost

                cost = calculate_cost(
                    model_id=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                self.usage_metadata["estimated_cost_usd"] = cost

                self.ui.console.print("  [dim]Usage:[/dim]")
                if input_tokens > 0:
                    self.ui.console.print(f"  [dim]  input: {input_tokens:,} tokens[/dim]")
                if output_tokens > 0:
                    self.ui.console.print(f"  [dim]  output: {output_tokens:,} tokens[/dim]")
                if reasoning_tokens > 0:
                    self.ui.console.print(f"  [dim]  reasoning: {reasoning_tokens:,} tokens[/dim]")
                self.ui.console.print(f"  [dim]  total cost: ${cost:.4f}[/dim]")

            local_path = (
                str(self.local_scripts_dir / self._get_client_filename()) if self.local_scripts_dir else None
            )
            self.ui.success(str(script_path), local_path)

            return {
                "script_path": str(script_path),
                "usage": self.usage_metadata,
            }

        except Exception as e:
            error_msg = str(e)
            self.ui.error(error_msg)
            self.message_store.save_error(error_msg)

            if "401" in error_msg or "unauthorized" in error_msg.lower():
                self.ui.console.print(
                    "\n[dim]Make sure DEEPSEEK_API_KEY is set correctly.[/dim]"
                )
                self.ui.console.print("[dim]Get your API key from https://platform.deepseek.com/api_keys[/dim]")
            elif "429" in error_msg or "rate" in error_msg.lower():
                self.ui.console.print(
                    "\n[dim]Rate limit hit. The engineer retried 3 times with exponential backoff.[/dim]"
                )
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                self.ui.console.print(
                    "\n[dim]Request timed out. Check your network connection.[/dim]"
                )

            return None

    async def _create_completion_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_retries: int = 3,
    ):
        """Create chat completion with retry on 429 errors.

        Args:
            model: Model ID
            messages: Chat messages
            max_retries: Maximum number of retries (default: 3)

        Returns:
            Async stream iterator

        Raises:
            Exception: If all retries are exhausted
        """
        import asyncio

        from openai import RateLimitError

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                return stream
            except RateLimitError as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    self.ui.console.print(f"  [dim]Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})[/dim]")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                # Re-raise non-rate-limit errors immediately
                raise

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error

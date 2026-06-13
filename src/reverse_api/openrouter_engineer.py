"""OpenRouter SDK implementation for API reverse engineering.

Uses OpenAI-compatible API via OpenRouter's endpoint.
"""

import os
from typing import Any

from openai import AsyncOpenAI

from .base_engineer import BaseEngineer


class OpenRouterEngineer(BaseEngineer):
    """Uses OpenRouter to analyze HAR files and generate API scripts."""

    def __init__(
        self,
        *args: Any,
        openrouter_model: str = "anthropic/claude-sonnet-4",
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter engineer.

        Args:
            openrouter_model: Model ID in provider/model format (e.g., "anthropic/claude-sonnet-4")
        """
        super().__init__(*args, **kwargs)

        self.openrouter_model = openrouter_model
        if not self.model:
            self.model = openrouter_model

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def analyze_and_generate(self) -> dict[str, Any] | None:
        """Run the reverse engineering analysis with OpenRouter.

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

            stream = await self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
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

                if chunk.usage:
                    usage_metadata = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }

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
                )
                self.usage_metadata["estimated_cost_usd"] = cost

                self.ui.console.print("  [dim]Usage:[/dim]")
                if input_tokens > 0:
                    self.ui.console.print(f"  [dim]  input: {input_tokens:,} tokens[/dim]")
                if output_tokens > 0:
                    self.ui.console.print(f"  [dim]  output: {output_tokens:,} tokens[/dim]")
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
                    "\n[dim]Make sure OPENROUTER_API_KEY is set correctly.[/dim]"
                )
                self.ui.console.print("[dim]Get your API key from https://openrouter.ai/keys[/dim]")
            elif "not found" in error_msg.lower() or "model" in error_msg.lower():
                self.ui.console.print(
                    f"\n[dim]Model '{self.openrouter_model}' not found. Check available models at https://openrouter.ai/models[/dim]"
                )

            return None

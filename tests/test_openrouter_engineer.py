"""Tests for openrouter_engineer.py - OpenRouterEngineer class."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverse_api.base_engineer import BaseEngineer


class TestOpenRouterEngineerImport:
    """Test that OpenRouterEngineer can be imported."""

    def test_import_class(self):
        """OpenRouterEngineer class should be importable from reverse_api.openrouter_engineer."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        assert OpenRouterEngineer is not None
        assert issubclass(OpenRouterEngineer, BaseEngineer)


class TestOpenRouterEngineerInit:
    """Test OpenRouterEngineer initialization."""

    def test_init_with_default_model(self, tmp_path):
        """Should initialize with default model when not specified."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            engineer = OpenRouterEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
            )
            assert engineer.openrouter_model == "anthropic/claude-sonnet-4"
            assert engineer.model == "anthropic/claude-sonnet-4"

    def test_init_with_custom_model(self, tmp_path):
        """Should initialize with custom model when specified."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            engineer = OpenRouterEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                openrouter_model="anthropic/claude-opus-4",
            )
            assert engineer.openrouter_model == "anthropic/claude-opus-4"
            assert engineer.model == "anthropic/claude-opus-4"

    def test_init_raises_without_api_key(self, tmp_path):
        """Should raise ValueError when OPENROUTER_API_KEY is not set."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {}, clear=True):
            with patch("reverse_api.base_engineer.Path.home", return_value=tmp_path):
                with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set in environment"):
                    OpenRouterEngineer(
                        run_id="test123",
                        har_path=har_path,
                        prompt="test prompt",
                    )

    def test_init_creates_openai_client(self, tmp_path):
        """Should create AsyncOpenAI client with OpenRouter base URL."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("reverse_api.openrouter_engineer.AsyncOpenAI") as mock_openai:
                engineer = OpenRouterEngineer(
                    run_id="test123",
                    har_path=har_path,
                    prompt="test prompt",
                )
                mock_openai.assert_called_once_with(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="test-key",
                )
                assert engineer.client is not None


class TestOpenRouterEngineerDispatch:
    """Test run_reverse_engineering dispatch for openrouter SDK."""

    def test_dispatches_to_openrouter(self, tmp_path):
        """OpenRouter SDK is used when sdk='openrouter'."""
        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("reverse_api.openrouter_engineer.OpenRouterEngineer") as mock_cls:
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                mock_instance.start_sync = MagicMock()
                mock_instance.stop_sync = MagicMock()

                from reverse_api.engineer import run_reverse_engineering

                with patch("reverse_api.engineer.asyncio.run", return_value={"test": True}):
                    result = run_reverse_engineering(
                        run_id="test123",
                        har_path=har_path,
                        prompt="test prompt",
                        sdk="openrouter",
                        openrouter_model="anthropic/claude-sonnet-4",
                    )
                    mock_cls.assert_called_once()
                    # Verify openrouter_model was passed
                    call_kwargs = mock_cls.call_args[1]
                    assert call_kwargs.get("openrouter_model") == "anthropic/claude-sonnet-4"


class TestOpenRouterEngineerAnalyzeAndGenerate:
    """Test analyze_and_generate method."""

    @pytest.mark.asyncio
    async def test_analyze_and_generate_success(self, tmp_path):
        """Should successfully generate API client with streaming."""
        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        # Create scripts_dir
        scripts_dir = tmp_path / "scripts" / "test123"
        scripts_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            engineer = OpenRouterEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                output_dir=str(tmp_path),
            )

            # Mock scripts_dir and _get_client_filename
            engineer.scripts_dir = scripts_dir
            engineer._get_client_filename = MagicMock(return_value="api_client.py")

            # Mock the AsyncOpenAI client streaming response
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock(delta=MagicMock(content="test output"))]
            mock_chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

            async def mock_stream():
                yield mock_chunk

            engineer.client.chat.completions.create = AsyncMock(return_value=mock_stream())

            # Mock _build_prompts
            engineer._build_prompts = MagicMock(return_value=("system", "user"))

            # Mock message_store
            engineer.message_store.save_prompt = MagicMock()
            engineer.message_store.save_result = MagicMock()
            engineer.message_store.save_thinking = MagicMock()

            # Mock ui
            engineer.ui.header = MagicMock()
            engineer.ui.start_analysis = MagicMock()
            engineer.ui.success = MagicMock()
            engineer.ui.thinking = MagicMock()
            engineer.ui.console = MagicMock()

            result = await engineer.analyze_and_generate()
            # Should return a result dict
            assert result is not None
            assert "script_path" in result
            assert "usage" in result

    @pytest.mark.asyncio
    async def test_analyze_and_generate_http_401(self, tmp_path):
        """Should handle HTTP 401 error gracefully."""
        from openai import AuthenticationError

        from reverse_api.openrouter_engineer import OpenRouterEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            engineer = OpenRouterEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                output_dir=str(tmp_path),
            )

            # Mock _build_prompts
            engineer._build_prompts = MagicMock(return_value=("system", "user"))

            # Mock client to raise AuthenticationError
            engineer.client.chat.completions.create = MagicMock(
                side_effect=AuthenticationError(
                    message="Invalid API key",
                    response=MagicMock(status_code=401),
                    body={"error": "Invalid API key"},
                )
            )

            # Mock ui and message_store
            engineer.ui.header = MagicMock()
            engineer.ui.start_analysis = MagicMock()
            engineer.ui.error = MagicMock()
            engineer.message_store.save_error = MagicMock()

            result = await engineer.analyze_and_generate()
            # Should return None on error
            assert result is None
            engineer.ui.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

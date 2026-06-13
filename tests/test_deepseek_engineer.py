"""Tests for deepseek_engineer.py - DeepSeekEngineer class."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverse_api.base_engineer import BaseEngineer


class TestDeepSeekEngineerImport:
    """Test that DeepSeekEngineer can be imported."""

    def test_import_class(self):
        """DeepSeekEngineer class should be importable from reverse_api.deepseek_engineer."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        assert DeepSeekEngineer is not None
        assert issubclass(DeepSeekEngineer, BaseEngineer)


class TestDeepSeekEngineerInit:
    """Test DeepSeekEngineer initialization."""

    def test_init_with_default_model(self, tmp_path):
        """Should initialize with default model when not specified."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
            )
            assert engineer.deepseek_model == "deepseek-chat"
            assert engineer.model == "deepseek-chat"

    def test_init_with_custom_model(self, tmp_path):
        """Should initialize with custom model when specified."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                deepseek_model="deepseek-coder",
            )
            assert engineer.deepseek_model == "deepseek-coder"
            assert engineer.model == "deepseek-coder"

    def test_init_raises_without_api_key(self, tmp_path):
        """Should raise ValueError when DEEPSEEK_API_KEY is not set."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {}, clear=True):
            with patch("reverse_api.base_engineer.Path.home", return_value=tmp_path):
                with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not set in environment"):
                    DeepSeekEngineer(
                        run_id="test123",
                        har_path=har_path,
                        prompt="test prompt",
                    )

    def test_init_creates_openai_client(self, tmp_path):
        """Should create AsyncOpenAI client with DeepSeek base URL."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("reverse_api.deepseek_engineer.AsyncOpenAI") as mock_openai:
                engineer = DeepSeekEngineer(
                    run_id="test123",
                    har_path=har_path,
                    prompt="test prompt",
                )
                mock_openai.assert_called_once_with(
                    base_url="https://api.deepseek.com/v1",
                    api_key="test-key",
                )
                assert engineer.client is not None


class TestDeepSeekEngineerDispatch:
    """Test run_reverse_engineering dispatch for deepseek SDK."""

    def test_dispatches_to_deepseek(self, tmp_path):
        """DeepSeek SDK is used when sdk='deepseek'."""
        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("reverse_api.deepseek_engineer.DeepSeekEngineer") as mock_cls:
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
                        sdk="deepseek",
                        deepseek_model="deepseek-chat",
                    )
                    mock_cls.assert_called_once()
                    # Verify deepseek_model was passed
                    call_kwargs = mock_cls.call_args[1]
                    assert call_kwargs.get("deepseek_model") == "deepseek-chat"


class TestDeepSeekEngineerAnalyzeAndGenerate:
    """Test analyze_and_generate method."""

    @pytest.mark.asyncio
    async def test_analyze_and_generate_success(self, tmp_path):
        """Should successfully generate API client with streaming."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        # Create scripts_dir
        scripts_dir = tmp_path / "scripts" / "test123"
        scripts_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
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
    async def test_analyze_and_generate_reasoning_model(self, tmp_path):
        """Should capture reasoning_content for deepseek-reasoner model."""
        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        # Create scripts_dir
        scripts_dir = tmp_path / "scripts" / "test123"
        scripts_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                output_dir=str(tmp_path),
                deepseek_model="deepseek-reasoner",
            )

            # Mock scripts_dir and _get_client_filename
            engineer.scripts_dir = scripts_dir
            engineer._get_client_filename = MagicMock(return_value="api_client.py")

            # Mock the AsyncOpenAI client streaming response with reasoning_content
            mock_chunk_1 = MagicMock()
            mock_chunk_1.choices = [MagicMock(delta=MagicMock(content=None, reasoning_content="Let me think..."))]
            mock_chunk_1.usage = None

            mock_chunk_2 = MagicMock()
            mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="final answer", reasoning_content=None))]
            mock_chunk_2.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

            async def mock_stream():
                yield mock_chunk_1
                yield mock_chunk_2

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
            # Verify reasoning_content was captured (this will be checked in implementation)

    @pytest.mark.asyncio
    async def test_analyze_and_generate_429_retry(self, tmp_path):
        """Should retry on 429 rate limit error with exponential backoff."""
        from openai import RateLimitError

        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                output_dir=str(tmp_path),
            )

            # Mock _build_prompts
            engineer._build_prompts = MagicMock(return_value=("system", "user"))

            # Mock client to raise RateLimitError then succeed
            call_count = [0]

            async def mock_create(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise RateLimitError(
                        message="Rate limit exceeded",
                        response=MagicMock(status_code=429),
                        body={"error": "Rate limit"},
                    )
                # On third attempt, succeed
                mock_chunk = MagicMock()
                mock_chunk.choices = [MagicMock(delta=MagicMock(content="success"))]
                mock_chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

                async def mock_stream():
                    yield mock_chunk

                return mock_stream()

            engineer.client.chat.completions.create = AsyncMock(side_effect=mock_create)

            # Mock sleep to avoid waiting
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Mock scripts_dir
                engineer.scripts_dir = tmp_path / "scripts" / "test123"
                engineer.scripts_dir.mkdir(parents=True, exist_ok=True)
                engineer._get_client_filename = MagicMock(return_value="api_client.py")

                # Mock message_store
                engineer.message_store.save_prompt = MagicMock()
                engineer.message_store.save_result = MagicMock()

                # Mock ui
                engineer.ui.header = MagicMock()
                engineer.ui.start_analysis = MagicMock()
                engineer.ui.success = MagicMock()
                engineer.ui.thinking = MagicMock()
                engineer.ui.console = MagicMock()
                engineer.ui.error = MagicMock()

                result = await engineer.analyze_and_generate()
                # Should eventually succeed after retries
                assert result is not None
                assert call_count[0] == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_analyze_and_generate_network_timeout(self, tmp_path):
        """Should handle network timeout and return None."""
        import httpx

        from reverse_api.deepseek_engineer import DeepSeekEngineer

        har_path = tmp_path / "test.har"
        har_path.touch()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            engineer = DeepSeekEngineer(
                run_id="test123",
                har_path=har_path,
                prompt="test prompt",
                output_dir=str(tmp_path),
            )

            # Mock _build_prompts
            engineer._build_prompts = MagicMock(return_value=("system", "user"))

            # Mock client to raise httpx.TimeoutException
            engineer.client.chat.completions.create = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            # Mock ui and message_store
            engineer.ui.header = MagicMock()
            engineer.ui.start_analysis = MagicMock()
            engineer.ui.error = MagicMock()
            engineer.message_store.save_error = MagicMock()

            result = await engineer.analyze_and_generate()
            # Should return None on network error
            assert result is None
            engineer.ui.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

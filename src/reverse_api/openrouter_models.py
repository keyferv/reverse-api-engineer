"""OpenRouter model discovery — fetches available models from API."""

import json
import time
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_FILE = Path.home() / ".reverse-api" / "openrouter_models.json"
CACHE_TTL_SECONDS = 3600  # 1 hour


def fetch_openrouter_models(api_key: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch available models from OpenRouter API with caching.

    Args:
        api_key: OpenRouter API key
        force_refresh: If True, ignore cache and fetch fresh

    Returns:
        List of model dicts with 'id', 'name', 'context_length', 'pricing'
    """
    # Check cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            cache_data = json.loads(CACHE_FILE.read_text())
            if time.time() - cache_data.get("timestamp", 0) < CACHE_TTL_SECONDS:
                return cache_data.get("models", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from API
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        models = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            prompt_price = pricing.get("prompt", "0")
            completion_price = pricing.get("completion", "0")

            models.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length"),
                "pricing": {
                    "prompt": float(prompt_price) if prompt_price else 0.0,
                    "completion": float(completion_price) if completion_price else 0.0,
                },
            })

        # Sort by name
        models.sort(key=lambda x: x["name"].lower())

        # Cache the results
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({
            "timestamp": time.time(),
            "models": models,
        }, indent=2))

        return models

    except Exception as e:
        # If fetch fails, return empty list (will fall back to hardcoded)
        return []


def get_openrouter_model_choices(api_key: str) -> list[dict[str, str]]:
    """Get model choices formatted for questionary select.

    Returns:
        List of dicts with 'name' (display) and 'value' (model id)
    """
    models = fetch_openrouter_models(api_key)

    if not models:
        # Fallback to hardcoded if API fails
        return [
            {"name": "Claude Sonnet 4 (via OpenRouter)", "value": "anthropic/claude-sonnet-4"},
            {"name": "Claude Opus 4 (via OpenRouter)", "value": "anthropic/claude-opus-4"},
        ]

    choices = []
    for m in models:
        prompt_price = m["pricing"]["prompt"]
        ctx = m.get("context_length", 0)
        ctx_str = f"{ctx // 1000}k" if ctx else "?"
        price_str = f"${prompt_price:.2f}/M" if prompt_price > 0 else "free"

        display = f"{m['name']} [{ctx_str}] ({price_str})"
        choices.append({"name": display, "value": m["id"]})

    return choices

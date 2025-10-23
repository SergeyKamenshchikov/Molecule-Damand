from __future__ import annotations

import os

import httpx
import pytest
import requests
from dotenv import load_dotenv
from openai import AsyncClient as OpenAIAsyncClient
from openai import Client as OpenAIClient
from zenrows import ZenRowsClient

load_dotenv(dotenv_path=".env", override=True)

OPENAI_PROXY = os.getenv("OPENAI_PROXY_HTTP")
_PROXY_HEALTHY = False


def _http_client_with_proxy(timeout: int = 10) -> httpx.Client:
    if not OPENAI_PROXY:
        raise RuntimeError("OPENAI_PROXY_HTTP is not configured.")
    return httpx.Client(proxy=OPENAI_PROXY, timeout=timeout)


def _async_http_client_with_proxy(timeout: int = 10) -> httpx.AsyncClient:
    if not OPENAI_PROXY:
        raise RuntimeError("OPENAI_PROXY_HTTP is not configured.")
    return httpx.AsyncClient(proxy=OPENAI_PROXY, timeout=timeout)


def _proxy_ready() -> bool:
    return bool(OPENAI_PROXY) and _PROXY_HEALTHY


def test_00_proxy_probe_reaches_checkip():
    global _PROXY_HEALTHY

    try:
        response = requests.get("https://checkip.amazonaws.com", timeout=10)
        response.raise_for_status()
    except requests.HTTPError as exc:
        pytest.fail(
            f"Proxy HTTP error: status={exc.response.status_code}, body={exc.response.text}"
        )
    except requests.RequestException as exc:
        pytest.fail(f"Proxy request failed: {exc!r}")

    ip_address = response.text.strip()
    assert ip_address, "Proxy probe returned an empty IP address"

    if OPENAI_PROXY:
        _PROXY_HEALTHY = True


def _invoke_openai_chat(client: OpenAIClient) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Ответь словом 'активен'."},
        ],
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()


def test_openai_chat_completion_live():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for OpenAI live integration test.")

    try:
        content = _invoke_openai_chat(OpenAIClient(api_key=api_key))
    except Exception:
        if not _proxy_ready():
            raise
        with _http_client_with_proxy() as http_client:
            content = _invoke_openai_chat(
                OpenAIClient(api_key=api_key, http_client=http_client)
            )

    assert "актив" in content or "жив" in content


async def _invoke_perplexity_chat(client: OpenAIAsyncClient) -> str:
    response = await client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with the word 'alive'."},
        ],
        max_tokens=5,
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()


@pytest.mark.asyncio
async def test_perplexity_chat_completion_live():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY is required for Perplexity live integration test.")

    client = OpenAIAsyncClient(
        api_key=api_key,
        base_url="https://api.perplexity.ai",
    )

    try:
        content = await _invoke_perplexity_chat(client)
    except Exception:
        if not _proxy_ready():
            raise
        async with _async_http_client_with_proxy() as http_client:
            client = OpenAIAsyncClient(
                api_key=api_key,
                base_url="https://api.perplexity.ai",
                http_client=http_client,
            )
            content = await _invoke_perplexity_chat(client)

    assert "alive" in content or "ok" in content


def test_zenrows_key_allows_fetching_html():
    api_key = os.getenv("ZENROWS_KEY")
    if not api_key:
        pytest.skip("ZENROWS_KEY is required for ZenRows live integration test.")

    client = ZenRowsClient(api_key)
    response = client.get("https://httpbin.org/html", params={"premium_proxy": "true"}, timeout=10)
    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_scraperapi_key_allows_fetching_html():
    api_key = os.getenv("SCRAPERAPI_KEY")
    if not api_key:
        pytest.skip("SCRAPERAPI_KEY is required for ScraperAPI live integration test.")

    payload = {"api_key": api_key, "url": "https://httpbin.org/html"}
    response = requests.get("https://api.scraperapi.com/", params=payload, timeout=10)
    response.raise_for_status()
    assert "<html" in response.text.lower()

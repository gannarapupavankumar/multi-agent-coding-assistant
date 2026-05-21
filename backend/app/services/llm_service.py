import os

import requests
from fastapi import HTTPException
from groq import Groq

from app.config import (
    DEFAULT_LLM_PROVIDER,
    GROQ_MODEL,
    JSON_SYSTEM_PROMPT,
    OLLAMA_MODEL,
    OLLAMA_URL,
)


class LLMProviderError(HTTPException):
    def __init__(self, status_code: int, detail: str, stop_reason: str):
        super().__init__(status_code=status_code, detail=detail)
        self.stop_reason = stop_reason


def _classify_llm_error(exc: Exception) -> LLMProviderError | None:
    message = str(exc).lower()
    context_markers = (
        "token limit",
        "maximum context",
        "context length",
        "request too large",
        "payload too large",
        "too many tokens",
    )
    rate_markers = ("tpm", "rate_limit_exceeded", "rate limit", "too many requests")

    if any(marker in message for marker in context_markers):
        return LLMProviderError(
            status_code=413,
            detail="LLM context is too large for the configured provider.",
            stop_reason="llm_context_too_large",
        )
    if any(marker in message for marker in rate_markers):
        return LLMProviderError(
            status_code=429,
            detail="LLM provider rate limit was reached.",
            stop_reason="llm_rate_limit",
        )
    return None


def call_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, str]:
    selected_provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    fell_back_from_groq = False

    if selected_provider == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                groq_model = model or GROQ_MODEL
                client = Groq(api_key=groq_api_key)
                completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": JSON_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                content = completion.choices[0].message.content or ""
                return {
                    "model": groq_model,
                    "response": content,
                }
            except Exception as exc:
                classified_error = _classify_llm_error(exc)
                if classified_error is not None:
                    raise classified_error from exc
                selected_provider = "ollama"
                fell_back_from_groq = True
        else:
            selected_provider = "ollama"
            fell_back_from_groq = True

    if selected_provider == "ollama":
        ollama_model = OLLAMA_MODEL if fell_back_from_groq else model or OLLAMA_MODEL
        try:
            ollama_response = requests.post(
                OLLAMA_URL,
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature},
                },
                timeout=120,
            )
            ollama_response.raise_for_status()
            data = ollama_response.json()
        except requests.exceptions.ConnectionError as exc:
            raise HTTPException(
                status_code=503,
                detail="Could not connect to Ollama. Make sure Ollama is running on http://localhost:11434.",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise HTTPException(
                status_code=504,
                detail="Ollama did not respond within 120 seconds.",
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned an error: {exc.response.text}",
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Error calling Ollama: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an invalid JSON response.",
            ) from exc

        return {
            "model": data.get("model", ollama_model),
            "response": data.get("response", ""),
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported LLM provider: {selected_provider}",
    )


def call_ollama_prompt(prompt: str) -> dict[str, str]:
    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        ollama_response.raise_for_status()
        data = ollama_response.json()
    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running on http://localhost:11434.",
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama did not respond within 120 seconds.",
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.text}",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling Ollama: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid JSON response.",
        ) from exc

    return {
        "model": data.get("model", OLLAMA_MODEL),
        "response": data.get("response", ""),
    }

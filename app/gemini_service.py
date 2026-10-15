"""Gemini integration: turns a saved prediction into a plain-language explanation."""
import logging
import os
import time

from google import genai
from google.genai import errors

log = logging.getLogger("uvicorn.error")


class GeminiUnavailable(Exception):
    """Gemini is not configured (no API key)."""


class GeminiBusy(Exception):
    """Gemini is overloaded or out of quota on every model we tried."""


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def models_to_try() -> list[str]:
    """Main model first, then backups (set GEMINI_FALLBACK_MODELS in .env to change)."""
    backups = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite")
    order = [get_model_name()] + [m.strip() for m in backups.split(",") if m.strip()]
    return list(dict.fromkeys(order))  # remove duplicates, keep order


def build_prompt(input_data: dict, result: str, probability: float) -> str:
    details = "\n".join(f"- {k}: {v}" for k, v in input_data.items())
    return (
        "You are helping a retail manager understand a purchase prediction.\n"
        f"A machine learning model predicted: {result} "
        f"(probability of purchase: {probability:.1%}).\n\n"
        f"The product and promotion details were:\n{details}\n\n"
        "Explain this result in 3 to 4 short sentences of plain English for a "
        "non-technical reader. Mention which of the details above are most likely "
        "relevant (for example discount depth, promotions, price, season). "
        "You cannot see the model's internals, so say 'likely' where you are "
        "interpreting, and do not invent numbers that are not given."
    )


def explain_prediction(input_data: dict, result: str, probability: float) -> tuple[str, str]:
    """Returns (explanation, model_used). Retries when busy, then tries backup models."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(input_data, result, probability)

    for model in models_to_try():
        for attempt in range(2):                      # 2 tries per model
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    return text, model
                break                                 # empty answer -> next model
            except errors.ServerError as e:           # 5xx, e.g. 503 "high demand"
                log.warning("Gemini %s try %d: %s %s", model, attempt + 1, e.code, e.status)
                time.sleep(1.5 * (attempt + 1))       # wait a little, then retry
            except errors.APIError as e:
                log.warning("Gemini %s: %s %s", model, e.code, e.status)
                if e.code in (404, 429):              # model missing / quota -> next model
                    break
                raise                                 # bad key etc. -> real error
    raise GeminiBusy("Gemini is very busy right now. Please try again in a minute.")
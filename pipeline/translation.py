from typing import Any

from langdetect import detect
from mlx_lm import generate

from base import set_llama_prompt

def translate_to_english(text: str, model: Any, tokenizer: Any, max_tokens: int = 512) -> str:
    """
    Translate the given text to English.

    Args:
        text (str): The text to translate.

    Returns:
        str: The translated text.
    """
    # no need to translate if the text is already in English
    if detect(text[:250].lower()) == "en":
        return text

    system_prompt = (
        "You are an expert language translation assistant, "
        "tasked with translating online job postings, "
        "from any given language, into English. "
        "Don't provide any explanations or additional notes - just translate the given text."
    )
    translation_prompt = set_llama_prompt(system_prompt, text)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=translation_prompt,
        max_tokens=max_tokens,
    )

"""Connectivity smoke test: validates config and pings both API providers.

Run after changing keys or provider settings: `python scripts/smoke_test.py`
"""

from openai import OpenAI

from rag.config import get_settings


def main() -> None:
    settings = get_settings()  # fails fast here if required keys are missing

    chat = OpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.chat_base_url,
    )
    response = chat.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=5,
    )
    print(f"chat ({settings.chat_model}): OK — {response.choices[0].message.content!r}")

    embeddings = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    result = embeddings.embeddings.create(model=settings.embedding_model, input="ping")
    print(f"embeddings ({settings.embedding_model}): OK — {len(result.data[0].embedding)} dims")


if __name__ == "__main__":
    main()

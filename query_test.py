# query_test.py
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def test_deepseek_chat():
    """Test the DeepSeek chat API."""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello, can you help me?"}],
            max_tokens=50,
        )
        print("Chat API Test Success (DeepSeek):", response.choices[0].message.content)
    except Exception as e:
        print("Chat API Test Failed (DeepSeek):", str(e))


def test_openai_embeddings():
    """Test the OpenAI embeddings API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small", input="Test sentence for embeddings."
        )
        print(
            "Embeddings API Test Success (OpenAI): Generated",
            len(response.data[0].embedding),
            "dimensions",
        )
    except Exception as e:
        print("Embeddings API Test Failed (OpenAI):", str(e))


if __name__ == "__main__":
    print("Testing DeepSeek Chat...")
    test_deepseek_chat()
    print("\nTesting OpenAI Embeddings...")
    test_openai_embeddings()

# prototype.py
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from embeddings import load_or_build_vectorstore

load_dotenv()


def simple_rag():
    """A simple prototype to test RAG with DeepSeek (full cloud)."""
    vectorstore = load_or_build_vectorstore()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens=500,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )

    query = "What is in my documents?"
    result = qa.invoke({"query": query})
    print("Answer:", result["result"])
    if result.get("source_documents"):
        print("Sources:", [doc.metadata.get("source") for doc in result["source_documents"]])


if __name__ == "__main__":
    simple_rag()

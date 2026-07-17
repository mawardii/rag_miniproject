# embeddings.py
import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"


def compute_docs_hash(folder_path: str) -> str:
    """Calculate a hash based on file names and modification times to detect changes."""
    import hashlib

    hasher = hashlib.sha256()
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                hasher.update(f.encode())
                hasher.update(str(os.path.getmtime(p)).encode())
            except FileNotFoundError:
                pass
    return hasher.hexdigest()


def load_or_build_vectorstore(chunk_size=500, chunk_overlap=50, force_rebuild=False):
    """Load or build the FAISS index from documents in DOCS_DIR using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # OpenAI's embedding model
        api_key=os.getenv("OPENAI_API_KEY"),  # Use OpenAI API key
    )

    if not force_rebuild and os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    docs = []
    os.makedirs(DOCS_DIR, exist_ok=True)
    for f in os.listdir(DOCS_DIR):
        p = os.path.join(DOCS_DIR, f)
        name = f.lower()
        if name.endswith(".txt"):
            docs.extend(TextLoader(p, autodetect_encoding=True).load())
        elif name.endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())

    if not docs:
        raise RuntimeError("No documents found in 'docs/'. Add .txt or .pdf files.")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
    )
    split_docs = splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    print(f"Indexed {len(split_docs)} chunks using OpenAI embeddings!")
    return db


if __name__ == "__main__":
    vectorstore = load_or_build_vectorstore()
    print("Vector store built successfully with OpenAI embeddings!")

import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.loaders.load_docs import load_enterprise_documents

INDEX_PATH = "vector_index/faiss_index"

def index_documents():
    """
    Load and chunk enterprise documents, create embeddings, store in FAISS, and
    save the vectorstore locally.
    """
    # Step 1: Load and chunk docs
    documents = load_enterprise_documents()
    print(f"📄 Loaded {len(documents)} chunks with metadata.")
    # Each document is a langchain Document object with text, metadata, and
    # optional fields like page_content

    # Step 2: Create embeddings
    embeddings = OpenAIEmbeddings()  # you can swap this with HuggingFace embeddings
    # We use OpenAI embeddings in this example, but you can use other
    # embedding models like HuggingFace or your own custom model

    # Step 3: Store in FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    # We store the embeddings in a FAISS index, which is a lightweight and
    # efficient library for similarity search

    # Step 4: Save locally
    os.makedirs("vector_index", exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ Vectorstore saved to {INDEX_PATH}")

if __name__ == "__main__":
    index_documents()

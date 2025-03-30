from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document


INDEX_PATH = "vector_index/faiss_index"

def create_retriever(k: int = 3) -> VectorStoreRetriever:
    """
    Creates and returns a configured VectorStoreRetriever for the enterprise knowledge base.
    
    Args:
        k (int): Number of documents to retrieve for each query. Defaults to 3.
        
    Returns:
        VectorStoreRetriever: A configured retriever object ready to use.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Set up retriever
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_kwargs={"k": k},
    )
    return retriever

def get_relevant_chunks(query: str, k: int = 3) -> list[Document]:
    """
    Retrieves and prints k relevant chunks (Documents) from the vectorstore
    given a query string.

    Args:
        query (str): String to query the vectorstore with.
        k (int): Number of documents to retrieve from the vectorstore.
            Defaults to 3.

    Returns:
        list[Document]: A list of k Documents sorted by relevance to the query.
    """
    # Use the centralized retriever creation function
    retriever = create_retriever(k)
    results = retriever.get_relevant_documents(query)

    print(f"\n🔍 Query: {query}")
    for i, doc in enumerate(results):
        print(f"\n--- Result #{i+1} ---")
        print(f"📄 Content:\n{doc.page_content[:300]}...")
        print(f"🧾 Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_query = "How many casual leaves do I get per year?"
    get_relevant_chunks(test_query)

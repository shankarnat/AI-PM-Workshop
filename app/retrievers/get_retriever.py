from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import VectorStoreRetriever

INDEX_PATH = "vector_index/faiss_index"

def get_relevant_chunks(query: str, k: int = 3):
    # Load existing FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Set up retriever
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)

    print(f"\n🔍 Query: {query}")
    for i, doc in enumerate(results):
        print(f"\n--- Result #{i+1} ---")
        print(f"📄 Content:\n{doc.page_content[:300]}...")
        print(f"🧾 Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_query = "How many casual leaves do I get per year?"
    get_relevant_chunks(test_query)

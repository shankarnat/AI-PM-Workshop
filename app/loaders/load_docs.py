import os
import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_enterprise_documents(
    path: str = "/Users/nikitashankar/Users/nikitashankar/Programming/AI-PM-Workshop/enterprise_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Load and chunk documents from the enterprise_docs folder.

    Metadata is inferred from subfolder names (e.g., HR, IT).
    """
    documents = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if not os.path.isdir(category_path):
            continue

        try:
            loader = DirectoryLoader(
                category_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                use_multithreading=True,
                show_progress=True,
            )
            loaded_docs = loader.load()
        except Exception as error:
            print(f"Error loading documents from {category_path}: {error}")
            continue

        for document in loaded_docs:
            document.metadata["category"] = category



        for document in loaded_docs:
            source_path = document.metadata.get("source")
            if source_path:
                abs_path = os.path.join(category_path, source_path)
                if os.path.exists(abs_path):
                    file_metadata = get_file_metadata(abs_path, category)
                    document.metadata.update(file_metadata)
                else:
                    document.metadata["category"] = category
            else:
                document.metadata["category"] = category


        documents.extend(loaded_docs)

    if not documents:
        print("No documents loaded.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunked_documents = text_splitter.split_documents(documents)

    for chunk in chunked_documents:
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_chunk"

    return chunked_documents

def get_file_metadata(file_path: str, category: str) -> dict:
    stat = os.stat(file_path)
    return {
        "source": os.path.basename(file_path),
        "path": file_path,
        "category": category,
        "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

 


if __name__ == "__main__":
    docs = load_enterprise_documents()
    print(f"✅ Loaded {len(docs)} chunks.")
    print("📄 Sample chunk content:\n", docs[0].page_content[:225])
    print("🧾 Metadata:\n", docs[0].metadata)
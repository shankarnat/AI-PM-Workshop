"""
Retrieval QA Chain Module

This module implements a Retrieval QA chain that connects the existing vector store with an LLM (Claude).
It can be used as a standalone agent or as part of the multi-agent system.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.retrievers.get_retriever import create_retriever

# Load environment variables (for API keys)
load_dotenv()

# Define constants
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def create_retrieval_qa_chain():
    """
    Create a Retrieval QA chain that connects the existing vector store with Claude.
    
    Returns:
        RetrievalQA: A configured QA chain ready to answer questions.
    """
    # Step 1: Use the centralized retriever creation function
    print("📚 Creating the retriever...")
    retriever = create_retriever(k=5)  # Retrieve top 5 chunks for each query
    
    # Step 3: Set up Claude as the LLM
    llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307",  # You can change to a different Claude model
        temperature=0.2,
        anthropic_api_key=ANTHROPIC_API_KEY, 
        max_tokens=1024
    )
    
    # Step 4: Create a custom prompt template
    # Load the retrieval prompt from file if it exists
    prompt_template = ""
    prompt_path = "app/prompts/retrieval_prompt.txt"
    
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        # Default prompt template if file doesn't exist
        prompt_template = """
        <context>
        {context}
        </context>
        
        Human: Based on the provided context, please answer the following question:
        {question}
        
        Assistant: """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    
    # Step 5: Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' method passes all documents to LLM at once
        retriever=retriever,
        return_source_documents=True,  # Include source docs in the response
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ Retrieval QA chain created successfully!")
    return qa_chain

def ask_question(qa_chain, question):
    """
    Ask a question to the Retrieval QA chain.
    
    Args:
        qa_chain: The RetrievalQA chain
        question (str): The question to ask
        
    Returns:
        dict: Response containing answer and source documents
    """
    print(f"\n🔍 Question: {question}")
    
    # Get response from the chain
    response = qa_chain({"query": question})
    
    # Print the answer
    print("\n📝 Answer:")
    print(response["result"])
    
    # Print source information
    print("\n📄 Sources:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nSource {i+1}:")
        print(f"Content (excerpt): {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")
    
    return response

if __name__ == "__main__":
    # Example usage
    qa_chain = create_retrieval_qa_chain()
    
    # Ask some test questions
    test_questions = [
        "What is the company's policy on remote work?",
        "How many vacation days do employees get?",
        "What steps should I take during the onboarding process?"
    ]
    
    for question in test_questions:
        ask_question(qa_chain, question)
        print("\n" + "-"*50 + "\n")

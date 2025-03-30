"""
AI-PM-Workshop Main Interface

This is the main entry point for the AI-PM-Workshop application.
It provides a simple command-line interface to interact with the different agents.
"""

import sys
from app.agents.retrieval_qa import create_retrieval_qa_chain, ask_question

def display_banner():
    """Display the application banner."""
    banner = """
    ╭────────────────────────────────────────────────╮
    │                                                │
    │   AI-PM Workshop - Enterprise Knowledge Bot    │
    │                                                │
    ╰────────────────────────────────────────────────╯
    """
    print(banner)

def display_help():
    """Display the help information."""
    help_text = """
    Available commands:
    
    /help       - Display this help information
    /exit       - Exit the application
    /examples   - Show example questions
    
    Just type your question to get an answer from the knowledge base.
    """
    print(help_text)

def display_examples():
    """Display example questions."""
    examples = """
    Example questions:
    
    1. What is the company's remote work policy?
    2. How many vacation days do employees get?
    3. What is the process for requesting time off?
    4. What security protocols should I follow?
    5. How do I set up my development environment?
    """
    print(examples)

def main():
    """Main function to run the application."""
    display_banner()
    print("Initializing the Enterprise Knowledge Assistant...")
    
    # Create the QA chain
    qa_chain = create_retrieval_qa_chain()
    
    print("\nReady to answer your questions! Type /help for available commands.")
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Process commands
        if user_input.lower() == "/exit":
            print("Thank you for using the Enterprise Knowledge Assistant. Goodbye!")
            sys.exit(0)
        elif user_input.lower() == "/help":
            display_help()
        elif user_input.lower() == "/examples":
            display_examples()
        elif user_input.strip():
            # Process the question
            ask_question(qa_chain, user_input)
        else:
            print("Please enter a question or command.")

if __name__ == "__main__":
    main()

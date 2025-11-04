from langchain_agent_chromadb import create_business_agent_chromadb
import argparse
import sys
import os
from pathlib import Path
from gemini_llm import GeminiConfig
from config.api_keys import load_dotenv

# Try to load API keys from .env file
load_dotenv()

def main(model_type="local"):
    print("\n=== Autonomous LangChain Agent Chat ===")
    print(f"Model: {model_type}")
    print("Type your query and press Enter. Type 'exit' to quit.\n")

    # Configure Gemini if needed
    gemini_config = None
    if model_type == "gemini":
        gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)

    # Create agent with specified model type
    agent_executor = create_business_agent_chromadb(
        model_type=model_type,
        gemini_config=gemini_config
    )
    chat_history = ""
    turn = 1
    while True:
        user_input = input(f"[User {turn}]: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        # Show current chat history before agent runs
        print(f"\n[Chat History before turn {turn}]:\n{chat_history}\n")
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        agent_reply = response.get("output", "")
        print(f"[Agent]: {agent_reply}\n")
        # Update chat history
        chat_history += f"User: {user_input}\nAgent: {agent_reply}\n"
        print(f"[Chat History after turn {turn}]:\n{chat_history}\n")
        turn += 1

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LangChain business agent with selectable LLM backend")
    parser.add_argument("--model", "-m", type=str, choices=["local", "gemini"], 
                      default="gemini", help="LLM model to use (default: gemini)")
    
    args = parser.parse_args()
    
    # Run with selected model configuration
    main(model_type=args.model)

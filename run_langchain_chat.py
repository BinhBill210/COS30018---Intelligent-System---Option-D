from langchain_agent_chromadb import create_business_agent_chromadb


def main():
    print("\n=== Autonomous LangChain Agent Chat ===")
    print("Type your query and press Enter. Type 'exit' to quit.\n")

    agent_executor = create_business_agent_chromadb()
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
    main()

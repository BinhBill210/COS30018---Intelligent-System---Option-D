import streamlit as st
import sys
from langchain_agent_chromadb import create_business_agent_chromadb

# Configure page
st.set_page_config(
    page_title="Business Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""
    if "agent_loaded" not in st.session_state:
        st.session_state.agent_loaded = False

# Load agent
@st.cache_resource
def load_agent():
    try:
        return create_business_agent_chromadb()
    except Exception as e:
        st.error(f"Error loading agent: {str(e)}")
        return None

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Business Agent")
        st.markdown("---")
        
        # Agent status
        if st.session_state.agent_executor is None:
            if st.button("Load Agent", type="primary"):
                with st.spinner("Loading agent..."):
                    st.session_state.agent_executor = load_agent()
                    if st.session_state.agent_executor:
                        st.session_state.agent_loaded = True
                        st.success("Agent loaded successfully!")
                        st.rerun()
        else:
            st.success("Agent Ready")
        
        st.markdown("---")
        
        # Chat controls
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = ""
            st.rerun()
        
        if st.button("Restart Agent"):
            st.session_state.agent_executor = None
            st.session_state.agent_loaded = False
            st.session_state.messages = []
            st.session_state.chat_history = ""
            st.rerun()
        
        st.markdown("---")
        
        # Chat statistics
        st.subheader("Chat Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        # Debug info (collapsible)
        with st.expander("Debug Info"):
            st.text(f"Agent loaded: {st.session_state.agent_loaded}")
            st.text(f"Chat history length: {len(st.session_state.chat_history)}")
    
    # Main chat interface
    st.title("Business Agent Chat")
    st.markdown("Ask me anything about business, and I'll help you with insights and analysis.")
    
    # Check if agent is loaded
    if not st.session_state.agent_executor:
        st.warning("Please load the agent first using the sidebar.")
        return
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Thinking..."):
                    # Call the agent
                    response = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    agent_reply = response.get("output", "I apologize, but I couldn't generate a response.")
                    
                    # Display the response
                    message_placeholder.markdown(agent_reply)
                    
                    # Update chat history
                    st.session_state.chat_history += f"User: {prompt}\nAgent: {agent_reply}\n"
                    
                    # Add assistant message to session
                    st.session_state.messages.append({"role": "assistant", "content": agent_reply})
                    
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Show chat history in expander (for debugging)
    if st.session_state.chat_history:
        with st.expander("View Raw Chat History"):
            st.text(st.session_state.chat_history)

if __name__ == "__main__":
    main()
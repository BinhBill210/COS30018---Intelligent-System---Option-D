import streamlit as st
import sys
import logging
import os
from pathlib import Path
from datetime import datetime
import json
from langchain_agent_chromadb import create_business_agent_chromadb, create_llm_instance
from gemini_llm import GeminiConfig
from config.api_keys import APIKeyManager, setup_api_keys_interactive, load_dotenv
from config.logging_config import setup_logging, get_performance_logger

# Try to load .env file at startup
load_dotenv()

# Initialize logging
setup_logging("INFO")

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
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "agent_loaded" not in st.session_state:
        st.session_state.agent_loaded = False
    
    # Model configuration
    if "model_type" not in st.session_state:
        st.session_state.model_type = "local"
    if "current_model_info" not in st.session_state:
        st.session_state.current_model_info = "Local Qwen2.5"
    if "api_key_status" not in st.session_state:
        st.session_state.api_key_status = {}
    if "current_llm_instance" not in st.session_state:
        st.session_state.current_llm_instance = None

# Conversation management functions
def create_new_conversation():
    """Create a new conversation and return its ID"""
    conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": "New Conversation",
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    return conversation_id

def get_current_conversation():
    """Get the current conversation or create one if none exists"""
    if not st.session_state.current_conversation_id:
        conversation_id = create_new_conversation()
        st.session_state.current_conversation_id = conversation_id
    
    return st.session_state.conversations.get(st.session_state.current_conversation_id, {})

def update_conversation_title(conversation_id, first_message):
    """Update conversation title based on first message"""
    if conversation_id in st.session_state.conversations:
        # Take first 30 characters of the first message as title
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        st.session_state.conversations[conversation_id]["title"] = title
        st.session_state.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()

def delete_conversation(conversation_id):
    """Delete a conversation"""
    if conversation_id in st.session_state.conversations:
        del st.session_state.conversations[conversation_id]
        
        # If we deleted the current conversation, switch to another or create new
        if st.session_state.current_conversation_id == conversation_id:
            if st.session_state.conversations:
                # Switch to the most recent conversation
                most_recent = max(st.session_state.conversations.keys(), 
                                key=lambda x: st.session_state.conversations[x]["updated_at"])
                st.session_state.current_conversation_id = most_recent
            else:
                # Create a new conversation if none exist
                st.session_state.current_conversation_id = create_new_conversation()

def switch_conversation(conversation_id):
    """Switch to a different conversation"""
    st.session_state.current_conversation_id = conversation_id

# Load agent
def load_agent(model_type, gemini_config=None):
    try:
        return create_business_agent_chromadb(
            model_type=model_type,
            gemini_config=gemini_config,
            local_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=False,
            max_iterations=15,
            verbose=True
        )
    except Exception as e:
        st.error(f"Error loading agent: {str(e)}")
        return None

# Check API key status
def check_api_key_status():
    api_manager = APIKeyManager()
    status = {}
    
    # Check Gemini API key
    gemini_key = api_manager.get_api_key('gemini')
    status['gemini'] = {
        'available': bool(gemini_key),
        'valid': False
    }
    
    if gemini_key:
        try:
            from gemini_llm import GeminiLLM
            status['gemini']['valid'] = GeminiLLM.test_connection(gemini_key)
        except Exception:
            status['gemini']['valid'] = False
    
    return status

# Model selection UI
def render_model_selection():
    st.subheader("🤖 Model Configuration")
    
    # Check API key status
    api_status = check_api_key_status()
    st.session_state.api_key_status = api_status
    
    # Model type selection
    model_options = ["local", "gemini"]
    
    # Disable Gemini if no valid API key
    if not api_status.get('gemini', {}).get('valid', False):
        if st.session_state.model_type == "gemini":
            st.warning("⚠️ Gemini API key not configured or invalid. Falling back to local model.")
            st.session_state.model_type = "local"
    
    model_type = st.selectbox(
        "Select Model Type",
        options=model_options,
        index=model_options.index(st.session_state.model_type),
        help="Choose between local LLM or Gemini API",
        key="model_type_select"
    )
    
    # Update session state
    if model_type != st.session_state.model_type:
        st.session_state.model_type = model_type
        st.session_state.agent_executor = None
        st.session_state.agent_loaded = False
    
    # API Key management
    with st.expander("🔑 API Key Status"):
        st.write("API keys can be configured via:")
        st.markdown("""
        1. **Environment Variables**: Set `GEMINI_API_KEY` in your environment
        2. **`.env` File**: Create a `.env` file in the project root with `GEMINI_API_KEY=your_key_here`
        """)
        
        # Display Gemini API key status
        gemini_status = "✓ Valid" if api_status['gemini']['valid'] else "✗ Not available or invalid"
        st.info(f"Gemini API Key: {gemini_status}")
        
        # Show where the key was found if it exists
        if api_status['gemini']['available']:
            api_manager = APIKeyManager()
            if 'GEMINI_API_KEY' in os.environ:
                source = "from environment variable"
            elif api_manager.get_api_key('gemini'):
                source = "from system storage"
            else:
                source = "unknown source"
                
            st.text(f"API key loaded {source}")
            
            # Simple hint for users who need to set up API key
            if not api_status['gemini']['valid']:
                st.warning("The API key was found but appears to be invalid")
        else:
            st.warning("No Gemini API key found. Run 'python scripts/create_env_file.py' to set up your API keys.")
    
    return model_type

# Render conversations list in sidebar
def render_conversations_sidebar():
    st.markdown("---")
    
    # Header with new chat button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Chat History")
    with col2:
        if st.button("➕", help="New conversation", key="new_chat"):
            conversation_id = create_new_conversation()
            st.session_state.current_conversation_id = conversation_id
            st.rerun()
    
    # Show conversations list
    if not st.session_state.conversations:
        st.text("No conversations yet...")
        return
    
    # Sort conversations by updated_at (most recent first)
    sorted_conversations = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["updated_at"],
        reverse=True
    )
    
    for conv_id, conv_data in sorted_conversations:
        # Create container for each conversation
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Make conversation title clickable
                is_current = conv_id == st.session_state.current_conversation_id
                
                if st.button(
                    conv_data["title"],
                    key=f"conv_{conv_id}",
                    help=f"Switch to this conversation",
                    type="primary" if is_current else "secondary",
                    use_container_width=True
                ):
                    if not is_current:
                        switch_conversation(conv_id)
                        st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation"):
                    delete_conversation(conv_id)
                    st.rerun()
        
    

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Business Agent")
        
        # Model selection
        current_model_type = render_model_selection()
        
      
        
        # Agent status and loading
        if st.session_state.agent_executor is None:
            if st.button("Load Agent", type="primary", use_container_width=True):
                with st.spinner("Loading agent..."):
                    gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)
                    
                    agent_result = load_agent(
                        model_type=st.session_state.model_type,
                        gemini_config=gemini_config
                    )
                    
                    if agent_result:
                        st.session_state.agent_executor = agent_result
                        st.session_state.agent_loaded = True
                        
                        # Try to store the LLM instance for performance tracking
                        try:
                            llm_instance = create_llm_instance(
                                model_type=st.session_state.model_type,
                                gemini_config=gemini_config
                            )
                            st.session_state.current_llm_instance = llm_instance
                        except Exception:
                            pass
                        
                        # Update model info
                        if st.session_state.model_type == "local":
                            st.session_state.current_model_info = "Local Qwen2.5"
                        elif st.session_state.model_type == "gemini":
                            st.session_state.current_model_info = "Google Gemini"
                        
                        st.success("Agent loaded successfully!")
                        st.rerun()
        else:
            # Display current model
            model_info = st.session_state.current_model_info
            st.success(f"Agent Ready: {model_info}")
            
            # Show performance stats for hybrid models
            if 'current_llm_instance' in st.session_state and st.session_state.current_llm_instance:
                try:
                    llm_instance = st.session_state.current_llm_instance
                    if hasattr(llm_instance, 'get_performance_stats'):
                        stats = llm_instance.get_performance_stats()
                        if stats.get('total_requests', 0) > 0:
                            with st.expander("📊 Performance Stats"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Requests", stats.get('total_requests', 0))
                                    st.metric("Primary Success Rate", f"{stats.get('primary_success_rate', 0):.1%}")
                                with col2:
                                    st.metric("Fallback Usage", stats.get('fallback_uses', 0))
                                    st.metric("Primary Failures", stats.get('primary_failures', 0))
                except Exception:
                    pass
        
        # Render conversations list
        render_conversations_sidebar()
        
        # Global controls at bottom
        if st.button("Clear All Conversations", help="Delete all conversations"):
            st.session_state.conversations = {}
            st.session_state.current_conversation_id = None
            st.rerun()
    
    # Main chat interface
    st.title("Business Agent Chat")
    
    # Display current model in header
    if st.session_state.agent_loaded:
        st.info(f"🤖 Currently using: **{st.session_state.current_model_info}**")
    
    # Check if agent is loaded
    if not st.session_state.agent_executor:
        st.warning("Please load the agent first using the sidebar.")
        st.markdown("""
        ### Getting Started:
        1. **Configure your model** in the sidebar (Local, Gemini, or Hybrid)
        2. **Set up API keys** if using Gemini (optional)
        3. **Load the agent** to start chatting
        4. **Ask questions** about business reviews and analytics
        
        ### Model Options:
        - **Local**: Fast, private Qwen2.5 model running on your machine
        - **Gemini**: Google's powerful API model for complex reasoning
        """)
        return
    
    # Get current conversation
    current_conv = get_current_conversation()
    current_messages = current_conv.get("messages", [])
    
    # Show current conversation title
    if current_conv.get("title", "") != "New Conversation":
        st.markdown("Ask me anything about business, and I'll help you with insights and analysis.")
    
    
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Ensure we have a current conversation
        if not st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = create_new_conversation()
        
        conv_id = st.session_state.current_conversation_id
        
        # Add user message to current conversation
        user_message = {"role": "user", "content": prompt}
        st.session_state.conversations[conv_id]["messages"].append(user_message)
        
        # Update conversation title if this is the first message
        if len(st.session_state.conversations[conv_id]["messages"]) == 1:
            update_conversation_title(conv_id, prompt)
        
        
      
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate agent response
        with st.chat_message("assistant"):
            
                message_placeholder = st.empty()
                
                try:
                    with st.spinner("Thinking..."):
                        # Build chat history from current conversation
                        chat_history = ""
                        for msg in current_messages:
                            role = "User" if msg["role"] == "user" else "Agent"
                            chat_history += f"{role}: {msg['content']}\n"
                        chat_history += f"User: {prompt}\n"
                        
                        # Call the agent
                        response = st.session_state.agent_executor.invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        
                        agent_reply = response.get("output", "I apologize, but I couldn't generate a response.")
                        
                        # Display the response
                        message_placeholder.markdown(agent_reply)
                        
                        # Add assistant message to current conversation
                        assistant_message = {"role": "assistant", "content": agent_reply}
                        st.session_state.conversations[conv_id]["messages"].append(assistant_message)
                        
                       
                        
                        # Rerun to update sidebar
                        st.rerun()
                        
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    message_placeholder.error(error_message)
                    # Add error message to conversation
                    error_msg = {"role": "assistant", "content": error_message}
                    st.session_state.conversations[conv_id]["messages"].append(error_msg)
                    st.rerun()

if __name__ == "__main__":
    main()
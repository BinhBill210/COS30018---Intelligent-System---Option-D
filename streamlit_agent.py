import streamlit as st
import sys
import logging
import os
from pathlib import Path
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""
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

# Load agent
def load_agent(model_type, gemini_config=None):
    try:
        return create_business_agent_chromadb(
            model_type=model_type,
            gemini_config=gemini_config,
            local_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=False,
            max_iterations=15,  # Fixed higher iteration count
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



def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Business Agent")
        st.markdown("---")
        
        # Model selection
        current_model_type = render_model_selection()
        
        st.markdown("---")
        
        # Agent status and loading
        if st.session_state.agent_executor is None:
            if st.button("Load Agent", type="primary"):
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
                            # Extract the LLM from the agent creation process
                            # This will help us access performance stats later
                            llm_instance = create_llm_instance(
                                model_type=st.session_state.model_type,
                                gemini_config=gemini_config
                            )
                            st.session_state.current_llm_instance = llm_instance
                        except Exception:
                            pass  # Not critical if we can't store the LLM instance
                        
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
                    pass  # Ignore stats errors
        
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
            # Clear the stored LLM instance
            if 'current_llm_instance' in st.session_state:
                del st.session_state.current_llm_instance
            st.rerun()
        
        st.markdown("---")
        
        # Chat statistics
        st.subheader("📈 Chat Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        # Model status
        st.subheader("🔧 Model Status")
        if st.session_state.api_key_status:
            if st.session_state.api_key_status.get('gemini', {}).get('valid', False):
                st.success("✓ Gemini API Ready")
            else:
                st.warning("⚠️ Gemini API Not Available")
        
        # Database status
        st.subheader("🗄️ Database Status")
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent))
            from database.db_manager import get_db_manager
            
            db_manager = get_db_manager()
            
            # Test database connectivity
            business_count = db_manager.execute_query("SELECT COUNT(*) as count FROM businesses")
            
            if not business_count.empty and business_count.iloc[0, 0] > 0:
                st.success("✓ DuckDB Connected")
                st.metric("Businesses", f"{business_count.iloc[0, 0]:,}")
                
                # Try to get review count (optional)
                try:
                    review_count = db_manager.execute_query("SELECT COUNT(*) as count FROM reviews")
                    if not review_count.empty and review_count.iloc[0, 0] > 0:
                        st.metric("Reviews", f"{review_count.iloc[0, 0]:,}")
                    else:
                        st.info("Reviews: Not loaded")
                except:
                    st.info("Reviews: Not available")
                
                # Get performance stats
                stats = db_manager.get_performance_stats()
                if stats.get('total_queries', 0) > 0:
                    st.metric("DB Queries", stats['total_queries'])
                
                st.metric("DB Size", f"{stats.get('database_size_mb', 0):.1f} MB")
                
            else:
                st.warning("⚠️ DuckDB Not Set Up")
                st.info("💡 Run: `python migration/setup_database.py`")
                
        except Exception as e:
            st.error(f"❌ DuckDB Setup Needed")
            st.info("💡 Run: `python migration/setup_database.py`")
        
        # Debug info (collapsible)
        with st.expander("ℹ️ System Info"):
            st.text(f"Agent loaded: {st.session_state.agent_loaded}")
            st.text(f"Model type: {st.session_state.model_type}")
            st.text(f"Chat history length: {len(st.session_state.chat_history)}")
    
    # Main chat interface
    st.title("Business Agent Chat")
    
    # Display current model in header
    if st.session_state.agent_loaded:
        st.info(f"🤖 Currently using: **{st.session_state.current_model_info}**")
    
    st.markdown("Ask me anything about business, and I'll help you with insights and analysis.")
    
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
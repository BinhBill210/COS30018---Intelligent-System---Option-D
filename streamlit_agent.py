import streamlit as st
import sys
import os
from pathlib import Path
from langchain_agent_duckdb import create_enhanced_business_agent

# Configure page
st.set_page_config(
    page_title="Enhanced Business Agent Chat",
    page_icon="🔥",
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
    if "duckdb_ready" not in st.session_state:
        st.session_state.duckdb_ready = False

# Check DuckDB setup
def check_duckdb_setup():
    """Check if DuckDB database exists and is ready."""
    db_path = Path("business_analytics.duckdb")
    if not db_path.exists():
        return False, "DuckDB database not found"
    
    try:
        # Quick validation
        import duckdb
        conn = duckdb.connect(str(db_path))
        business_count = conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
        review_count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        conn.close()
        
        if business_count > 0 and review_count > 0:
            return True, f"✅ DuckDB ready: {business_count:,} businesses, {review_count:,} reviews"
        else:
            return False, "DuckDB database is empty"
    except Exception as e:
        return False, f"DuckDB error: {str(e)}"

# Load enhanced agent
@st.cache_resource
def load_enhanced_agent():
    try:
        # Check DuckDB first
        duckdb_ready, duckdb_status = check_duckdb_setup()
        if not duckdb_ready:
            st.error(f"❌ DuckDB Setup Required: {duckdb_status}")
            st.info("📋 Run setup script: `python setup_duckdb_database.py`")
            return None
        
        st.success(duckdb_status)
        return create_enhanced_business_agent()
    except Exception as e:
        st.error(f"Error loading enhanced agent: {str(e)}")
        return None

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🔥 Enhanced Business Agent")
        st.markdown("**Powered by DuckDB Analytics**")
        st.markdown("---")
        
        # DuckDB status check
        duckdb_ready, duckdb_status = check_duckdb_setup()
        
        if duckdb_ready:
            st.success("🔥 DuckDB Analytics Ready")
            st.caption(duckdb_status)
        else:
            st.error("❌ DuckDB Setup Required")
            st.caption(duckdb_status)
            with st.expander("🔧 Setup Instructions"):
                st.code("python setup_duckdb_database.py", language="bash")
                st.markdown("Or see `DUCKDB_SETUP.md` for detailed instructions")
        
        st.markdown("---")
        
        # Agent status
        if st.session_state.agent_executor is None:
            if st.button("Load Enhanced Agent", type="primary", disabled=not duckdb_ready):
                with st.spinner("Loading enhanced agent with DuckDB..."):
                    st.session_state.agent_executor = load_enhanced_agent()
                    if st.session_state.agent_executor:
                        st.session_state.agent_loaded = True
                        st.success("Enhanced agent loaded successfully!")
                        st.rerun()
        else:
            st.success("🚀 Enhanced Agent Ready")
            st.caption("ChromaDB + DuckDB Analytics")
        
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
        
        # Performance metrics
        st.subheader("📊 Analytics")
        st.metric("Messages", len(st.session_state.messages))
        
        if duckdb_ready:
            with st.expander("🔥 DuckDB Performance"):
                st.markdown("""
                **Enhanced Capabilities:**
                - ⚡ 10x+ faster analytics
                - 📊 Complex aggregations  
                - 📈 Real-time trends
                - 🔍 Hybrid search
                """)
        
        # Debug info (collapsible)
        with st.expander("🔧 Debug Info"):
            st.text(f"Enhanced agent: {st.session_state.agent_loaded}")
            st.text(f"DuckDB ready: {duckdb_ready}")
            st.text(f"Chat history: {len(st.session_state.chat_history)} chars")
    
    # Main chat interface
    st.title("🔥 Enhanced Business Agent Chat")
    
    if duckdb_ready:
        st.markdown("🚀 **Powered by DuckDB Analytics** - Ask complex questions about business performance, trends, and insights!")
        
        # Show example queries
        with st.expander("💡 Try These Enhanced Queries"):
            st.markdown("""
            **🔥 DuckDB-Powered Analytics:**
            - "Analyze performance trends for the top restaurants"
            - "What are the sentiment trends for delivery services?"
            - "Find businesses with declining ratings and analyze why"
            - "Compare restaurant performance across different cities"
            
            **🔍 Hybrid Search + Analytics:**
            - "Find reviews mentioning 'slow service' and analyze the businesses"
            - "What do customers say about food quality in 5-star restaurants?"
            - "Analyze sentiment for businesses with parking complaints"
            """)
    else:
        st.warning("⚠️ Basic mode - Load DuckDB for enhanced analytics capabilities")
        st.markdown("Ask me about business reviews and I'll help with basic insights.")
    
    # Check if agent is loaded
    if not st.session_state.agent_executor:
        if duckdb_ready:
            st.info("🚀 Ready to load enhanced agent! Click 'Load Enhanced Agent' in the sidebar.")
        else:
            st.error("❌ DuckDB setup required. Run `python setup_duckdb_database.py` first.")
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
                with st.spinner("🔥 Enhanced agent analyzing with DuckDB..."):
                    # Call the enhanced agent
                    response = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    agent_reply = response.get("output", "I apologize, but I couldn't generate a response.")
                    
                    # Display the response with enhanced formatting
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
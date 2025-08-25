# enhanced_langchain_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage
from langchain.tools import Tool as LangChainTool
from langchain.prompts import PromptTemplate
from langchain_agent import LangChainLocalLLM, create_langchain_tools
from local_llm import LocalLLM

def create_enhanced_business_agent():
    """Enhanced version with memory and streaming"""
    
    # Initialize LocalLLM
    local_llm = LocalLLM(use_4bit=False)
    llm = LangChainLocalLLM(local_llm)
    
    # Get tools
    tools = create_langchain_tools()
    
    # Add memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Enhanced prompt with memory
    enhanced_prompt = PromptTemplate.from_template("""
You are a business review analysis agent with expertise in sentiment analysis and business intelligence.

TOOLS:
------
{tools}

CONVERSATION HISTORY:
{chat_history}

Use this format:
Thought: Do I need to use a tool? Yes/No
Action: [tool name]
Action Input: [input]
Observation: [result]
... (repeat as needed)
Final Answer: [your response]

Current request: {input}
{agent_scratchpad}
""")
    
    # Create agent with memory
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=enhanced_prompt
    )
    
    # Create executor with callbacks
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    return agent_executor

# Example with Chainlit integration
def create_chainlit_app():
    """Create a Chainlit app with the LangChain agent"""
    import chainlit as cl
    
    @cl.on_chat_start
    async def start():
        agent = create_enhanced_business_agent()
        cl.user_session.set("agent", agent)
        await cl.Message("Business Review Analysis Agent ready! How can I help you?").send()
    
    @cl.on_message
    async def main(message: cl.Message):
        agent = cl.user_session.get("agent")
        
        # Run agent asynchronously
        response = await cl.make_async(agent.invoke)({"input": message.content})
        
        await cl.Message(content=response["output"]).send()

if __name__ == "__main__":
    # For testing without Chainlit
    agent = create_enhanced_business_agent()
    
    result = agent.invoke({
        "input": "Analyze the sentiment of recent reviews and provide insights"
    })
    
    print(f"Final Response: {result['output']}")

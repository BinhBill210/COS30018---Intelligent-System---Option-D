# agent.py (updated with debugging)
from typing import List, Dict, Any, Callable
import json
import re

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Agent:
    def __init__(self, tools: List[Tool], max_iterations: int = 5, verbose: bool = True):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.conversation_history = []
        self.verbose = verbose
    
    def log(self, message):
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def run(self, user_input: str, llm_generate_func: Callable):
        self.conversation_history.append(f"User: {user_input}")
        
        # System prompt that defines the agent's behavior
        system_prompt = """You are a business review analysis agent. You have access to various tools to help analyze business reviews.

Available tools:
{tools_descriptions}

When you need to use a tool, respond in the following format:
Thought: [Your reasoning about what to do next]
Action: [Tool Name]
Action Input: [JSON input for the tool]

When you have a final answer, respond with:
Final Answer: [Your final response to the user]

IMPORTANT: Always respond in this exact format. Do not add any extra text outside of these sections."""

        tools_descriptions = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        
        system_prompt = system_prompt.format(tools_descriptions=tools_descriptions)
        
        # Initial prompt to the LLM
        prompt = f"{system_prompt}\n\nCurrent conversation:\n" + "\n".join(self.conversation_history[-5:]) + "\nAgent:"
        
        self.log(f"Initial prompt:\n{prompt}")
        
        for i in range(self.max_iterations):
            self.log(f"Iteration {i+1}")
            response = llm_generate_func(prompt)
            self.log(f"LLM response: {response}")
            self.conversation_history.append(f"Agent: {response}")
            
            # Parse the response for actions
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                self.log(f"Found final answer: {final_answer}")
                return final_answer
            
            # Use regex to better parse the response
            action_match = re.search(r"Action:\s*(\w+)", response)
            action_input_match = re.search(r"Action Input:\s*(\{.*\})", response, re.DOTALL)
            
            if action_match and action_input_match:
                action_line = action_match.group(1).strip()
                action_input_str = action_input_match.group(1).strip()
                
                self.log(f"Parsed action: {action_line}")
                self.log(f"Parsed action input: {action_input_str}")
                
                try:
                    action_input = json.loads(action_input_str)
                except json.JSONDecodeError:
                    self.log(f"Failed to parse JSON: {action_input_str}")
                    action_input = {"input": action_input_str}
                
                # Execute the tool
                if action_line in self.tools:
                    self.log(f"Executing tool: {action_line}")
                    try:
                        tool_result = self.tools[action_line](**action_input)
                        self.log(f"Tool result: {tool_result}")
                        self.conversation_history.append(f"Observation: {tool_result}")
                        
                        # Update prompt with the observation
                        prompt += f"\nObservation: {tool_result}\nAgent:"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        self.log(error_msg)
                        self.conversation_history.append(f"Observation: {error_msg}")
                        prompt += f"\nObservation: {error_msg}\nAgent:"
                else:
                    error_msg = f"Error: Tool '{action_line}' not found."
                    self.log(error_msg)
                    self.conversation_history.append(f"Observation: {error_msg}")
                    prompt += f"\nObservation: {error_msg}\nAgent:"
            else:
                self.log("Response format not recognized")
                error_msg = "Please use the specified response format with 'Action:' and 'Action Input:' or 'Final Answer:'."
                self.conversation_history.append(f"Observation: {error_msg}")
                prompt += f"\nObservation: {error_msg}\nAgent:"
        
        return "I'm sorry, I wasn't able to complete your request within the allowed iterations."
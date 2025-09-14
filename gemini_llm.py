# gemini_llm.py
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import os
import json
import logging
import time
from dataclasses import dataclass


@dataclass
class GeminiConfig:
    """Configuration for Gemini model"""
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40


class GeminiLLM(LLM):
    """Custom LangChain LLM wrapper for Google's Gemini API"""
    
    # Declare fields for Pydantic (similar to LangChainLocalLLM pattern)
    config: Any
    api_key: Any
    model: Any
    _generation_config: Any
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[GeminiConfig] = None, **kwargs):
        """Initialize Gemini LLM
        
        Args:
            api_key: Google API key for Gemini. If None, will look for GEMINI_API_KEY env var
            config: GeminiConfig object for model configuration
            **kwargs: Additional LangChain LLM parameters
        """
        # Prepare configuration
        config_obj = config or GeminiConfig()
        api_key_val = api_key or os.getenv('GEMINI_API_KEY')
        
        if not api_key_val:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize the Gemini components first
        try:
            # Configure the API
            genai.configure(api_key=api_key_val)
            
            # Initialize the model
            model_obj = genai.GenerativeModel(config_obj.model_name)
            
            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=config_obj.temperature,
                max_output_tokens=config_obj.max_output_tokens,
                top_p=config_obj.top_p,
                top_k=config_obj.top_k,
            )
            
            logging.info(f"✓ Gemini LLM initialized with model: {config_obj.model_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Gemini LLM: {e}")
            raise
        
        # Initialize parent LangChain LLM with the fields
        super().__init__(
            config=config_obj,
            api_key=api_key_val,
            model=model_obj,
            _generation_config=generation_config,
            **kwargs
        )
        
        # Ensure the attributes are properly set as instance variables
        # (In case the parent constructor doesn't handle them)
        self.config = config_obj
        self.api_key = api_key_val
        self.model = model_obj
        self._generation_config = generation_config

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini API
        
        Args:
            prompt: The input prompt
            stop: Stop sequences (not directly supported by Gemini, will be handled post-generation)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Track generation time for monitoring
            start_time = time.time()
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self._generation_config
            )
            
            # Get the generated text
            generated_text = response.text
            
            # Handle stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break
            
            # Log generation metrics
            generation_time = time.time() - start_time
            logging.info(f"Gemini generation completed in {generation_time:.2f}s")
            
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"Gemini generation error: {str(e)}"
            logging.error(error_msg)
            
            # Return a graceful error message instead of raising
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters for the LLM"""
        return {
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics (placeholder for future implementation)"""
        return {
            "model": self.config.model_name,
            "total_requests": "N/A",  # Would need to implement request tracking
            "total_tokens": "N/A",    # Would need to implement token tracking
        }
    
    @classmethod
    def test_connection(cls, api_key: str) -> bool:
        """Test if the API key is valid and connection works
        
        Args:
            api_key: Gemini API key to test
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Test with a simple prompt
            response = model.generate_content("Hello, this is a connection test.")
            return bool(response.text)
            
        except Exception as e:
            logging.error(f"Gemini connection test failed: {e}")
            return False


# Utility functions for API key management
class GeminiAPIKeyManager:
    """Manage Gemini API keys securely"""
    
    KEY_FILE = ".gemini_api_key"
    
    @classmethod
    def save_key(cls, api_key: str) -> bool:
        """Save API key to file (encrypted in real implementation)
        
        Args:
            api_key: The API key to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real implementation, this should be encrypted
            with open(cls.KEY_FILE, 'w') as f:
                f.write(api_key)
            
            # Set restrictive permissions
            os.chmod(cls.KEY_FILE, 0o600)
            return True
            
        except Exception as e:
            logging.error(f"Failed to save API key: {e}")
            return False
    
    @classmethod
    def load_key(cls) -> Optional[str]:
        """Load API key from file
        
        Returns:
            API key if found, None otherwise
        """
        try:
            if os.path.exists(cls.KEY_FILE):
                with open(cls.KEY_FILE, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logging.error(f"Failed to load API key: {e}")
        
        return None
    
    @classmethod
    def delete_key(cls) -> bool:
        """Delete saved API key
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(cls.KEY_FILE):
                os.remove(cls.KEY_FILE)
            return True
        except Exception as e:
            logging.error(f"Failed to delete API key: {e}")
            return False


# Example usage and testing
def main():
    """Test the Gemini LLM wrapper"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test API key (you'll need to set this)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("Please set GEMINI_API_KEY environment variable")
            return
        
        # Test connection
        print("Testing Gemini connection...")
        if GeminiLLM.test_connection(api_key):
            print("✓ Connection successful")
        else:
            print("✗ Connection failed")
            return
        
        # Create Gemini LLM instance
        config = GeminiConfig(temperature=0.1, max_output_tokens=512)
        gemini_llm = GeminiLLM(api_key=api_key, config=config)
        
        # Test generation
        test_prompt = "Explain the importance of customer reviews for businesses in 2-3 sentences."
        print(f"\nTest prompt: {test_prompt}")
        print("\nGenerating response...")
        
        response = gemini_llm._call(test_prompt)
        print(f"\nResponse: {response}")
        
        # Display model info
        print(f"\nModel info: {gemini_llm._identifying_params}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

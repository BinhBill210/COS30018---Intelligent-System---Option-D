# local_llm.py (updated with a more reliable model)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LocalLLM:
    def __init__(self, model_name="distilgpt2"):  # Try a smaller, more reliable model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a pipeline approach
            self.fallback_pipeline = pipeline("text-generation", model=model_name)
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs):
        try:
            # Try the original approach first
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # If response is empty, try with different parameters
            if not response:
                print("Empty response, trying with different parameters...")
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.9,  # Higher temperature
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error in generation: {e}")
            # Fallback to pipeline
            try:
                result = self.fallback_pipeline(prompt, max_length=max_length, **kwargs)
                return result[0]['generated_text'][len(prompt):].strip()
            except:
                return "I'm sorry, I couldn't generate a response. Please try again."
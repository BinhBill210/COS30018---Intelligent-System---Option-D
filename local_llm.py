# local_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from base_llm import BaseLLM


class LocalLLM(BaseLLM):
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_length: int = 500, **kwargs):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = response[len(prompt):].strip()
        
        return response
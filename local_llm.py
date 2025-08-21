# local_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import login, whoami
import torch
import os
from typing import Optional

class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_4bit: bool = True, hf_token: Optional[str] = None):
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        # Handle authentication
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        if self.hf_token:
            try:
                login(token=self.hf_token)
                print("✓ Logged in to Hugging Face Hub")
            except Exception as e:
                print(f"Warning: Could not login to Hugging Face Hub: {e}")
        else:
            print("Warning: No Hugging Face token provided. Some models may require authentication.")
        
        print(f"Loading model: {model_name}")
        print(f"Using 4-bit quantization: {self.use_4bit}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        try:
            # Load tokenizer with authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                token=self.hf_token
            )
            
            # Configure model loading with quantization if requested
            if self.use_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        token=self.hf_token
                    )
                    print("✓ Loaded with 4-bit quantization")
                except Exception as e:
                    print(f"4-bit loading failed: {e}, falling back to standard loading")
                    self.use_4bit = False
            
            # Standard loading without quantization
            if not self.use_4bit or not torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    token=self.hf_token
                )
                print("✓ Loaded with standard precision")
            
            # Set generation config
            self.generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print(f"✓ Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            # Suggest authentication if it's a 401 error
            if "401" in str(e) or "authentication" in str(e).lower():
                print("\nThis model may require authentication.")
                print("1. Create a Hugging Face account at https://huggingface.co/join")
                print("2. Get an access token from https://huggingface.co/settings/tokens")
                print("3. Set it as an environment variable: export HF_TOKEN=your_token_here")
                print("4. Or pass it to the constructor: LocalLLM(hf_token='your_token_here')")
            import traceback
            traceback.print_exc()
            raise
    
    def generate(self, prompt: str, **kwargs):
        try:
            # Format the prompt for Qwen2.5
            messages = [
                {"role": "system", "content": "You are a business review analysis agent that follows instructions precisely and uses tools when needed."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt")
            if torch.cuda.is_available():
                model_inputs = model_inputs.to(self.model.device)
            
            # Generate
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                **kwargs
            )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return "I couldn't generate a response due to an error."
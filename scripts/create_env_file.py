# Create or update a .env file with your Gemini API key
# This script helps you create a .env file for storing your API keys

import os
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.api_keys import APIKeyManager

def create_env_file():
    """Create a .env file with API keys"""
    print("📝 Creating .env file for API keys")
    print("=" * 50)
    
    env_path = Path(".env")
    
    # Check if .env already exists
    if env_path.exists():
        print(f"⚠️ A .env file already exists at {env_path.resolve()}")
        overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
    
    # Get Gemini API key
    print("\n🔑 Enter your Google Gemini API Key")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    gemini_key = input("Enter your Gemini API key: ").strip()
    
    if not gemini_key:
        print("❌ No API key provided. Operation cancelled.")
        return
    
    # Write to .env file
    try:
        with open(env_path, 'w') as f:
            f.write("# API Keys for LLM System\n\n")
            f.write("# Google Gemini API Key\n")
            f.write(f"GEMINI_API_KEY={gemini_key}\n\n")
        
        print(f"✅ .env file created successfully at {env_path.resolve()}")
        print("You can now run the application and it will automatically use this API key.")
        
        # Verify the key works
        api_manager = APIKeyManager(load_env=True)
        loaded_key = api_manager.get_api_key('gemini')
        
        if loaded_key == gemini_key:
            print("✅ Verified: API key loads correctly from .env file")
        else:
            print("⚠️ Warning: The API key doesn't seem to load correctly from the .env file.")
            
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    create_env_file()

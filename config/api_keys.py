# config/api_keys.py
import os
import json
import keyring
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
import base64


class APIKeyManager:
    """Secure API key management system with multiple storage options"""
    
    def __init__(self, use_keyring: bool = True, encryption_key: Optional[str] = None):
        """Initialize API key manager
        
        Args:
            use_keyring: Whether to use system keyring for storage
            encryption_key: Optional encryption key for file-based storage
        """
        self.use_keyring = use_keyring
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.key_file = self.config_dir / ".api_keys.enc"
        
        # Initialize encryption
        if encryption_key:
            self.encryption_key = encryption_key.encode()
        else:
            self.encryption_key = self._get_or_create_encryption_key()
        
        self.cipher = Fernet(base64.urlsafe_b64encode(self.encryption_key.ljust(32)[:32]))
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for file storage"""
        key_path = self.config_dir / ".enc_key"
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = os.urandom(32)
            with open(key_path, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            return key
    
    def save_api_key(self, service: str, key: str) -> bool:
        """Save API key securely
        
        Args:
            service: Service name (e.g., 'gemini', 'openai')
            key: API key to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_keyring:
                keyring.set_password("llm_hybrid_system", service, key)
                logging.info(f"API key for {service} saved to system keyring")
            else:
                # Encrypted file storage
                keys = self._load_keys_from_file()
                keys[service] = key
                self._save_keys_to_file(keys)
                logging.info(f"API key for {service} saved to encrypted file")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save API key for {service}: {e}")
            return False
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for service
        
        Args:
            service: Service name (e.g., 'gemini', 'openai')
            
        Returns:
            API key if found, None otherwise
        """
        try:
            # First try environment variable
            env_key = f"{service.upper()}_API_KEY"
            if env_key in os.environ:
                return os.environ[env_key]
            
            # Then try keyring or file storage
            if self.use_keyring:
                key = keyring.get_password("llm_hybrid_system", service)
                if key:
                    logging.info(f"API key for {service} loaded from system keyring")
                    return key
            else:
                keys = self._load_keys_from_file()
                if service in keys:
                    logging.info(f"API key for {service} loaded from encrypted file")
                    return keys[service]
            
            logging.warning(f"No API key found for {service}")
            return None
            
        except Exception as e:
            logging.error(f"Failed to get API key for {service}: {e}")
            return None
    
    def delete_api_key(self, service: str) -> bool:
        """Delete API key for service
        
        Args:
            service: Service name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_keyring:
                keyring.delete_password("llm_hybrid_system", service)
            else:
                keys = self._load_keys_from_file()
                if service in keys:
                    del keys[service]
                    self._save_keys_to_file(keys)
            
            logging.info(f"API key for {service} deleted")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete API key for {service}: {e}")
            return False
    
    def list_services(self) -> list:
        """List services with stored API keys
        
        Returns:
            List of service names
        """
        try:
            if self.use_keyring:
                # Keyring doesn't have a direct way to list keys
                # We'll maintain a list in a separate file
                return self._get_keyring_services()
            else:
                keys = self._load_keys_from_file()
                return list(keys.keys())
                
        except Exception as e:
            logging.error(f"Failed to list services: {e}")
            return []
    
    def _load_keys_from_file(self) -> Dict[str, str]:
        """Load encrypted keys from file"""
        if not self.key_file.exists():
            return {}
        
        try:
            with open(self.key_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            logging.error(f"Failed to load keys from file: {e}")
            return {}
    
    def _save_keys_to_file(self, keys: Dict[str, str]) -> None:
        """Save encrypted keys to file"""
        try:
            data = json.dumps(keys).encode()
            encrypted_data = self.cipher.encrypt(data)
            
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            
        except Exception as e:
            logging.error(f"Failed to save keys to file: {e}")
            raise
    
    def _get_keyring_services(self) -> list:
        """Get list of services from keyring metadata file"""
        metadata_file = self.config_dir / ".keyring_services"
        
        if not metadata_file.exists():
            return []
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _update_keyring_services(self, service: str, operation: str) -> None:
        """Update keyring services metadata"""
        metadata_file = self.config_dir / ".keyring_services"
        services = self._get_keyring_services()
        
        if operation == "add" and service not in services:
            services.append(service)
        elif operation == "remove" and service in services:
            services.remove(service)
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(services, f)
        except Exception as e:
            logging.error(f"Failed to update keyring services: {e}")
    
    def test_api_key(self, service: str, key: str) -> bool:
        """Test if an API key is valid
        
        Args:
            service: Service name
            key: API key to test
            
        Returns:
            True if valid, False otherwise
        """
        if service.lower() == 'gemini':
            from gemini_llm import GeminiLLM
            return GeminiLLM.test_connection(key)
        else:
            logging.warning(f"Testing not implemented for service: {service}")
            return True  # Assume valid if we can't test
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all services
        
        Returns:
            Dictionary with usage stats
        """
        stats = {}
        services = self.list_services()
        
        for service in services:
            stats[service] = {
                "has_key": bool(self.get_api_key(service)),
                "last_used": "N/A",  # Would need to implement usage tracking
                "total_requests": "N/A",
                "total_cost": "N/A"
            }
        
        return stats


# Convenience functions
def get_gemini_key() -> Optional[str]:
    """Get Gemini API key"""
    manager = APIKeyManager()
    return manager.get_api_key('gemini')


def save_gemini_key(key: str) -> bool:
    """Save Gemini API key"""
    manager = APIKeyManager()
    return manager.save_api_key('gemini', key)


def setup_api_keys_interactive():
    """Interactive setup for API keys"""
    manager = APIKeyManager()
    
    print("🔐 API Key Setup")
    print("=" * 50)
    
    # Gemini API Key
    print("\n📡 Google Gemini API Key")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    
    gemini_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    if gemini_key:
        if manager.test_api_key('gemini', gemini_key):
            if manager.save_api_key('gemini', gemini_key):
                print("✓ Gemini API key saved successfully")
            else:
                print("✗ Failed to save Gemini API key")
        else:
            print("✗ Invalid Gemini API key")
    
    print("\n✓ API key setup complete!")
    print(f"Configured services: {manager.list_services()}")


if __name__ == "__main__":
    # Interactive setup
    setup_api_keys_interactive()

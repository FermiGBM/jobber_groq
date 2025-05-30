"""
Model configuration for jobber.
Contains settings for different LLM providers and models.

USER GUIDE:
-----------
1. To change the default model, modify the DEFAULT_MODEL variable below.
2. To add or modify available models, edit the AVAILABLE_MODELS dictionary.
3. Each model configuration should include:
   - provider: The model provider (e.g., "groq", "openai", "anthropic")
   - api_base: Base URL for the provider's API
   - max_tokens: Maximum number of tokens for generation
   - temperature: Sampling temperature (0.0 to 1.0)
   - top_p: Nucleus sampling parameter

Example usage in code:
    from jobber.config.model_config import get_default_model
    # Use default model
    agent = BaseAgent()  # Uses DEFAULT_MODEL
    
    # Or specify a model
    agent = BaseAgent(model="groq/llama-3.3-70b-versatile")
"""

from typing import Dict, Any, Optional

# Provider API base URLs
API_BASES = {
    "groq": "https://api.groq.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1"
}

# Available models and their configurations
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "groq/llama-3.3-70b-versatile": {
        "provider": "groq",
        "api_base": API_BASES["groq"],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "groq/llama2-70b-4096": {
        "provider": "groq",
        "api_base": API_BASES["groq"],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "openai/gpt-4-turbo-preview": {
        "provider": "openai",
        "api_base": API_BASES["openai"],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "anthropic/claude-3-opus-20240229": {
        "provider": "anthropic",
        "api_base": API_BASES["anthropic"],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    # Add more models as needed
}

# Default model to use
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

def get_model_config(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model: Model identifier. If None, returns default model config.
        
    Returns:
        Dictionary containing model configuration.
    """
    if model is None:
        model = DEFAULT_MODEL
        
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} not found in available models. Available models: {list(AVAILABLE_MODELS.keys())}")
        
    return AVAILABLE_MODELS[model]

def get_default_model() -> str:
    """
    Get the default model identifier.
    
    Returns:
        String containing the default model identifier.
    """
    return DEFAULT_MODEL

def list_available_models() -> list:
    """
    List all available models.
    
    Returns:
        List of available model identifiers.
    """
    return list(AVAILABLE_MODELS.keys()) 
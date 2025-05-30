"""
Model configuration settings for the Jobber project.
This file centralizes all model-related configurations.

USER GUIDE:
-----------
1. To change the default model, modify the DEFAULT_MODEL variable below.
2. To add or modify available models, edit the AVAILABLE_MODELS dictionary.
3. Each model configuration should include:
   - provider: The model provider (e.g., "groq", "openai", "anthropic")
   - max_tokens: Maximum number of tokens for generation
   - temperature: Sampling temperature (0.0 to 1.0)
   - top_p: Nucleus sampling parameter

Example usage in code:
    from jobber.config.model_config import get_default_model
    # Use default model
    agent = BaseAgent()  # Uses DEFAULT_MODEL
    
    # Or specify a model
    agent = BaseAgent(model="gpt-4-turbo-preview")
"""

from typing import Dict, Optional

# Available models and their configurations
AVAILABLE_MODELS = {
    "groq/llama-3.3-70b-versatile": {
        "provider": "groq",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "groq/llama2-70b-4096": {
        "provider": "groq",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "gpt-4-turbo-preview": {
        "provider": "openai",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
    },
    # Add more models as needed
}

# Default model to use
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

def get_model_config(model_name: Optional[str] = None) -> Dict:
    """
    Get the configuration for a specific model.
    If no model is specified, returns the default model configuration.
    
    Args:
        model_name (str, optional): Name of the model to get configuration for.
        
    Returns:
        Dict: Model configuration including provider and parameters.
        
    Raises:
        ValueError: If the specified model is not available.
    """
    model = model_name or DEFAULT_MODEL
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model}' not found in available models. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model]

def get_default_model() -> str:
    """
    Get the default model name.
    
    Returns:
        str: Name of the default model.
    """
    return DEFAULT_MODEL

def list_available_models() -> list:
    """
    Get a list of all available models.
    
    Returns:
        list: List of available model names.
    """
    return list(AVAILABLE_MODELS.keys()) 
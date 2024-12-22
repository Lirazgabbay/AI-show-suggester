
import os
from dotenv import load_dotenv
from openai import OpenAI


def connect_to_openai():
    """
    Connects to the OpenAI API and returns the client instance. Get API key from environment variable.

    Returns:
        OpenAI: An instance of the OpenAI client for interacting with the OpenAI API.
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_1')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_1 environment variable is not set. Please set it with your OpenAI API key.")
    client = OpenAI(api_key=api_key)
    return client


def connect_to_LightX():
    """
    Connects to the LightX API by retrieving the API key from an environment variable.
    
    This function loads environment variables using `dotenv` and retrieves the `LightX_API_KEY`. 
    If the key is not found, it raises an exception with a clear error message.

    Returns:
        str: The LightX API key retrieved from the environment variable.
    """
    load_dotenv()
    api_key = os.getenv('LightX_API_KEY')
    if not api_key:
        raise ValueError("LightX_API_KEY environment variable is not set. Please set it with your LightX key.")
    print(f"API Key loaded: {api_key[:5]}...") # Print first 5 chars to verify it's loaded
    return api_key

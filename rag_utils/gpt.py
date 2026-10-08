import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
# Load environment variables from .env file
load_dotenv()
from openai import AzureOpenAI
# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")  # This must be your Azure deployment name

# Initialize Azure OpenAI client
# client = openai.AzureOpenAI(
#     api_key=AZURE_OPENAI_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT
# )
client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
    )
def chatgpt_response(system_prompt, user_prompt, max_tokens=256, temperature=0.2):
    """
    Call Azure OpenAI GPT model and return the assistant's response.

    Args:
        system_prompt (str): System-level instructions that set the assistant's behavior.
        user_prompt (str): The user’s query, potentially including context from RAG.
        max_tokens (int): Maximum number of tokens in the response (default: 256).
        temperature (float): Controls randomness (0.0 = deterministic, 1.0 = creative; default: 0.2).

    Returns:
        str: The assistant's textual response.

    Behavior:
        - Uses Azure OpenAI's Chat Completion API (`gpt-4o` or `gpt-35-turbo` via deployment name).
        - Supports system+user message format for chat interaction.

    Other options:
        - You can add `"role": "assistant"` messages in between to continue prior conversations.
        - For OpenAI (non-Azure), use:
            ```python
            openai.ChatCompletion.create(model="gpt-4", ...)
            ```
        - For streaming responses, pass `stream=True` and iterate over `response`.
        - For multi-turn memory, store message history in a list and include all prior turns.
        - For larger responses (e.g., summarization), increase `max_tokens` to 1024–4096.
        - For reproducibility in tests, set `temperature=0` and `seed` if supported.
        - Add `stop` tokens to control where the model should stop generating output.
        - Customize function to return full response object if logging/usage tracking is needed.

    Example:
        >>> chatgpt_response("You are a helpful assistant.", "What is leave policy?")
        'Employees are entitled to...'

    Environment variables required:
        - AZURE_OPENAI_KEY
        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_API_VERSION
        - AZURE_OPENAI_GPT_DEPLOYMENT
    """
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    
        max_completion_tokens=13107,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
 
    )
    return response.choices[0].message.content



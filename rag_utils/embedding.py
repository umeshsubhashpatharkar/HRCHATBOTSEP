import numpy as np
import os
import openai
from dotenv import load_dotenv
from openai import AzureOpenAI
# Load environment variables from .env
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = "text-embedding-3-small"
endpoint = "https://instanceragchatbot.openai.azure.com/"
# EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")  # This should be the *deployment name* in Azure

# Initialize Azure OpenAI client
# client = openai.AzureOpenAI(
#     api_key=AZURE_OPENAI_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT
# )
client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_deployment = EMBED_MODEL,
        azure_endpoint=endpoint,
        api_key=AZURE_OPENAI_KEY
    )
def get_embedding(text):
    """
    Generate a dense vector embedding for a given input text using Azure OpenAI.

    Args:
        text (str): The input string to embed. Should be a single paragraph or chunk.

    Returns:
        np.ndarray: A 1D NumPy array representing the embedding (float32 type).

    Notes:
        - This function uses a deployed Azure OpenAI embedding model (e.g., `text-embedding-ada-002`).
        - The embedding model must be properly deployed and named in your Azure OpenAI resource.
        - The environment must include keys in the `.env` file:
            AZURE_OPENAI_KEY
            AZURE_OPENAI_ENDPOINT
            AZURE_OPENAI_API_VERSION
            AZURE_OPENAI_EMBEDDING_MODEL

    Other options:
        - If not using Azure, you can use OpenAI’s standard API:
            ```python
            openai.OpenAI(api_key="...").embeddings.create(model="text-embedding-ada-003", input=[text])
            ```
        - For open-source models, consider HuggingFace `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
        - Normalize vectors if using cosine similarity search:
            ```python
            emb = emb / np.linalg.norm(emb)
            ```
        - You can also cache embeddings to avoid repeated API calls during development or for common inputs.

    Example:
        >>> get_embedding("What are the leave policies?")
        array([0.0021, 0.3182, ..., 0.0043], dtype=float32)
    """
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return np.array(response.data[0].embedding, dtype='float32')
if __name__ == "__main__":
    test_text = "What are the leave policies?"
    embedding = get_embedding(test_text)
    print(f"Embedding for '{test_text}': {embedding}")
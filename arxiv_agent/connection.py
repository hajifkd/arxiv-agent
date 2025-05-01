from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled
import os
import dotenv
dotenv.load_dotenv()

set_tracing_disabled(disabled=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_FAST_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT_NAME")
AZURE_OPENAI_BALANCED_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_BALANCED_DEPLOYMENT_NAME")
AZURE_OPENAI_DEEP_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEEP_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")

def get_custom_client(deployment_name):
    return AsyncOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_API_BASE}openai/deployments/{deployment_name}",
        default_headers={"api-key": AZURE_OPENAI_API_KEY},
        default_query={"api-version": AZURE_OPENAI_API_VERSION},
    )


fast_client = get_custom_client(AZURE_OPENAI_FAST_DEPLOYMENT_NAME)
balanced_client = get_custom_client(AZURE_OPENAI_BALANCED_DEPLOYMENT_NAME)
deep_client = get_custom_client(AZURE_OPENAI_DEEP_DEPLOYMENT_NAME)
set_default_openai_client(fast_client, use_for_tracing=False)
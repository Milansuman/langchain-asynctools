from dotenv import load_dotenv
import os
import pytest
from langchain_litellm import ChatLiteLLM
from netra import Netra

@pytest.fixture
def env():
    load_dotenv()
    return os.environ

@pytest.fixture
def netra(env):
    Netra.init(
        app_name="langchain-asynctools",
        headers=f"x-api-key={env.get('NETRA_API_KEY')}"
    )

@pytest.fixture
def llm(env):
    llm = ChatLiteLLM(
        model="litellm_proxy/gpt-4-turbo",
        api_key=env.get("LITELLM_API_KEY"),
        api_base="https://llm.keyvalue.systems"
    )

    return llm
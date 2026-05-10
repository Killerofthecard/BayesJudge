"""
LLM client.
"""

import json
from typing import List, Dict, Union, Optional


class LLMClient:
    """
    LLM client wrapper.

    Supports OpenAI-compatible API endpoints.
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the client.

        Args:
            api_key: API key.
            base_url: API base URL.
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")

        self.api_key = api_key
        self.base_url = base_url

    def call(
        self,
        prompt: Union[str, List[Dict]],
        model: str,
        temperature: float = 0.0,
        request_format: str = None,
        max_retries: int = 10
    ) -> Union[str, Dict, None]:
        """
        Call the LLM.

        Args:
            prompt: Prompt text or list of message dicts.
            model: Model name.
            temperature: Sampling temperature.
            request_format: Request format (None or "json").
            max_retries: Maximum number of retries.

        Returns:
            LLM response content.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("prompt must be a string or a list of dicts")

        try_times = 0
        result = None

        while try_times < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                content = completion.choices[0].message.content

                if request_format == "json":
                    try:
                        result = json.loads(content)
                        break
                    except json.JSONDecodeError:
                        try_times += 1
                        continue
                else:
                    result = content
                    break

            except Exception as e:
                print(f"LLM call failed ({try_times + 1}/{max_retries}): {e}")
                try_times += 1
                continue

        if result is None:
            print(f"LLM call failed after reaching the maximum retry count ({max_retries})")

        return result


_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """
    Get the default LLM client.

    Note: Set the API key via environment variables before using this function,
    or modify the configuration below.
    """
    global _default_client

    if _default_client is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        _default_client = LLMClient(api_key=api_key, base_url=base_url)

    return _default_client


def call_llm(prompt, model, request_format=None):
    """
    Backward-compatible LLM call function.

    Prefer using the LLMClient class for more flexible configuration.
    """
    client = get_default_client()
    return client.call(prompt, model, request_format=request_format)

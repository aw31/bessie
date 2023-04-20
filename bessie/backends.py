import logging
import os
import pprint
import time
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

import anthropic
import openai
import tiktoken

import bessie.settings  # Loads environment variables
from bessie.utils import Timeout

_OPENAI_TOKEN_LIMIT = os.getenv("OPENAI_TOKEN_LIMIT", 8000)
_ANTHROPIC_TOKEN_LIMIT = os.getenv("OPENAI_TOKEN_LIMIT", 8000)


@dataclass
class Message:
    """Message class for text-based agents/backends."""

    sender: str
    content: str


PromptType = TypeVar("PromptType")


@dataclass
class Request(Generic[PromptType]):
    prompt: PromptType
    stop: Optional[Sequence[str]] = None


class Backend:
    def run(self, request: Request) -> str:
        raise NotImplementedError

    def batch_run(self, requests: list[Request]) -> list[str]:
        return [self.run(request) for request in requests]


class ChatBackend(Backend):
    def _condense_messages(self, messages: Sequence[Message]) -> Sequence[Message]:
        """Condense consecutive messages from the same sender into a single message and strip whitespace."""
        condensed_messages = []
        current_message = ""
        current_sender = ""
        for message in messages:
            assert message.sender != ""
            if message.sender != current_sender:
                if current_message:
                    condensed_message = Message(current_sender, current_message.strip())
                    condensed_messages.append(condensed_message)
                current_message = message.content
                current_sender = message.sender
            else:
                current_message += message.content
        if current_sender != "":
            condensed_message = Message(current_sender, current_message.strip())
            condensed_messages.append(condensed_message)
        return condensed_messages

    def run(self, request: Request[Sequence[Message]]) -> str:
        raise NotImplementedError


class DummyChat(ChatBackend):
    def run(self, request: Request[Sequence[Message]]) -> str:
        return "Moo!"


class OpenAIChat(ChatBackend):
    _ROLE_LOOKUP = {"system": "system", "environment": "user", "agent": "assistant"}

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        token_limit_str = _OPENAI_TOKEN_LIMIT
        self.token_limit = int(token_limit_str) if token_limit_str else None

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _count_tokens(self, messages: Sequence[dict]) -> int:
        """
        Counts the total number of tokens prompted/requested in a chat completion request.
        (See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.)
        """
        encoding = tiktoken.encoding_for_model(self.model)
        # There are 4 metadata tokens per message and 2 prompt tokens to prime the model response
        num_tokens = len(messages) * 4 + 2
        for m in messages:  # Count the message content tokens
            num_tokens += len(encoding.encode(m["content"]))
        return num_tokens

    def _truncate_messages(self, messages: list[dict]) -> Sequence[dict]:
        """Truncates messages to fit within the token limit."""
        while (
            self.token_limit is not None
            and self._count_tokens(messages) >= self.token_limit - self.max_tokens
        ):
            pop_index = 0
            while messages[pop_index]["role"] == "system":
                pop_index += 1
            messages.pop(pop_index)
        return messages

    def run(self, request: Request[Sequence[Message]]) -> str:
        messages = []
        for message in self._condense_messages(request.prompt):
            role, content = self._ROLE_LOOKUP[message.sender], message.content
            messages.append({"role": role, "content": content})
        messages = self._truncate_messages(messages)
        logging.info(
            f"Sending request to OpenAI chat completion API:\n"
            f"{pprint.pformat(messages, sort_dicts=False)}"
        )
        with Timeout(120, "OpenAI API timed out!"):
            response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stop=request.stop,
            )
        response_text = response["choices"][0].message.content  # type: ignore
        logging.info(f'Received response from OpenAI API: "{response_text}"')
        return response_text

    def _send_request(
        self, messages: list[dict], request: Request[Sequence[Message]]
    ) -> str:
        logging.info(
            f"Sending request to OpenAI chat completion API:\n"
            f"{pprint.pformat(messages, sort_dicts=False)}"
        )
        # Retry up to 3 times if the request errors
        for _ in range(3):
            try:
                with Timeout(60, "OpenAI API timed out!"):
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=messages,
                        stop=request.stop,
                    )
                response_text = response["choices"][0].message.content  # type: ignore
                logging.info(f'Received response from OpenAI API: "{response_text}"')
                return response_text
            except Exception as e:
                logging.warning(f"OpenAI API request failed: {e}")
                time.sleep(10)
                continue
        raise RuntimeError("OpenAI API request failed 3 times!")


class AnthropicChat(ChatBackend):
    _ROLE_LOOKUP = {"system": "system", "environment": "Human", "agent": "Assistant"}

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        token_limit_str = _ANTHROPIC_TOKEN_LIMIT
        self.token_limit = int(token_limit_str) if token_limit_str else None

        self.client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY", ""))

    def _format(self, messages: Sequence[dict]) -> str:
        """Formats a list of messages for Anthropic API. Ignores system messages."""
        return (
            "".join(
                f"\n\n{m['role']}: {m['content']}"
                for m in messages
                if m["role"] != "system"
            )
            + "\n\nAssistant:"
        )

    def _count_tokens(self, messages: Sequence[dict]) -> int:
        """
        Counts the total number of tokens prompted/requested in a chat completion request.
        (See https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/count_tokens.py.)
        """
        return anthropic.count_tokens(self._format(messages))

    def _truncate_messages(self, messages: list[dict]) -> Sequence[dict]:
        """Truncates messages to fit within the token limit."""
        while (
            self.token_limit is not None
            and self._count_tokens(messages) >= self.token_limit - self.max_tokens
        ):
            pop_index = 0
            messages.pop(pop_index)
        return messages

    def run(self, request: Request[Sequence[Message]]) -> str:
        messages = []
        for message in self._condense_messages(request.prompt):
            role, content = self._ROLE_LOOKUP[message.sender], message.content
            messages.append({"role": role, "content": content})
        prompt = self._format(self._truncate_messages(messages))
        logging.info(
            f"Sending request to Anthropic chat completion API:\n"
            f"{pprint.pformat(prompt, sort_dicts=False)}"
        )
        with Timeout(60, "Anthropic API timed out!"):
            stop_sequences = request.stop or []
            response = self.client.completion(
                model=self.model,
                temperature=self.temperature,
                max_tokens_to_sample=self.max_tokens,
                prompt=prompt,
                stop_sequences=stop_sequences,
            )
        response_text = response["completion"]
        logging.info(f'Received response from Anthropic API: "{response_text}"')
        return response_text.strip()

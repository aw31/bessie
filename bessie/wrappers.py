from typing import Any, List, Optional

from bessie.backends import Message, Request


class Wrapper:
    def prompt(self, observation: Any) -> Any:
        """Given an observation, return a prompt for the agent."""
        return self._prompt(observation)

    def _prompt(self, observation: Any) -> Any:
        raise NotImplementedError

    def parse(self, action: Any) -> Any:
        """Given a prompted action, return a legal action for the environment."""
        parsed_action = self._parse(action)
        return parsed_action

    def _parse(self, action: Any) -> Any:
        raise NotImplementedError

    def run(self, backend, observation, **kwargs):
        prompt = self.prompt(observation)
        response = backend.run(Request(prompt=prompt, **kwargs))
        return self.parse(response)

    def reset(self):
        pass


class ChatWrapper(Wrapper):
    def __init__(self, system_message: Optional[str] = None):
        self._system_message = system_message
        self.reset()

    def prompt(self, observation: Any) -> List[Message]:
        if self._last_action is not None:  # Finalize agent action
            self._messages += [self._last_action]
            self._last_action = None
        prompt = super().prompt(observation)
        self._messages += [Message("environment", p) for p in prompt]
        return self._messages

    def _prompt(self, observation: Any) -> List[str]:
        return [observation]

    def parse(self, action: str):
        # Must be idempotent! (We wait until the next prompt to finalize action.)
        self._last_action = Message("agent", action.strip() + "\n")
        return super().parse(action)

    def _parse(self, action: str) -> Any:
        return action

    def reset(self):
        self._messages: List[Message] = []
        self._last_action: Optional[Message] = None
        if self._system_message is not None:
            self._messages.append(Message("system", self._system_message.strip()))

"""A stand-in for the Anthropic client, so the pipeline can be tested without calling a model.

Each call returns the next canned reply. The replies are plain text blocks, the same shape
the real SDK gives back.
"""
from types import SimpleNamespace


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._replies.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def sql_reply(sql, note="Here is how I read it."):
    return f"{note}\n\n```sql\n{sql}\n```"

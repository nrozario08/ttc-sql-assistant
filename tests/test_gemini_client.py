import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from assistant import make_client  # noqa: E402
from gemini_client import GeminiClient, _to_text  # noqa: E402


def test_text_from_plain_string_and_from_reply_blocks():
    assert _to_text("hello") == "hello"
    blocks = [SimpleNamespace(type="text", text="a"), SimpleNamespace(type="other", text="x"),
              SimpleNamespace(type="text", text="b")]
    assert _to_text(blocks) == "ab"


def test_gemini_calls_are_reshaped_like_the_claude_ones(monkeypatch):
    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.update(model=model, roles=[c.role for c in contents], system=config.system_instruction)
            meta = SimpleNamespace(prompt_token_count=50, candidates_token_count=10, thoughts_token_count=5)
            return SimpleNamespace(text="```sql\nSELECT 1\n```", usage_metadata=meta)

    client = GeminiClient.__new__(GeminiClient)
    from google import genai
    client._genai = genai
    client._client = SimpleNamespace(models=FakeModels())
    client.messages = SimpleNamespace(create=client._create)

    reply = client.messages.create(
        model="gemini-3.8-flash", system="be careful", max_tokens=100,
        messages=[{"role": "user", "content": "q"},
                  {"role": "assistant", "content": [SimpleNamespace(type="text", text="first try")]},
                  {"role": "user", "content": "that failed"}],
        output_config={"effort": "medium"},        # Claude-only argument, ignored here
    )
    assert seen["roles"] == ["user", "model", "user"] and seen["system"] == "be careful"
    assert reply.content[0].text.startswith("```sql")
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (50, 15)


def test_make_client_picks_the_sdk_from_the_model_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    assert type(make_client("gemini-3.8-flash")).__name__ == "GeminiClient"
    assert type(make_client("claude-haiku-5-5")).__module__.startswith("anthropic")


def test_busy_server_is_retried_then_succeeds(monkeypatch):
    import gemini_client
    from google.genai import errors
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: None)
    calls = []

    class FlakyModels:
        def generate_content(self, model, contents, config):
            calls.append(1)
            if len(calls) < 3:
                raise errors.ServerError(503, {"error": {"message": "high demand"}})
            return "ok"

    client = GeminiClient.__new__(GeminiClient)
    client._client = SimpleNamespace(models=FlakyModels())
    assert client._generate_with_retries("m", [], None) == "ok" and len(calls) == 3

"""Lets the assistant talk to Gemini models through the same call it uses for Claude.

assistant.ask() only needs client.messages.create(...) to return text blocks and token counts,
so this wraps the google-genai SDK in that shape. The API key is read from the
GEMINI_API_KEY environment variable.
"""
from types import SimpleNamespace


def _to_text(content):
    """Message content is either a plain string or a list of reply blocks from an earlier call."""
    if isinstance(content, str):
        return content
    return "".join(b.text for b in content if getattr(b, "type", "") == "text")


class GeminiClient:
    def __init__(self):
        from google import genai
        self._genai = genai
        self._client = genai.Client()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, system, messages, max_tokens=16000, **_ignored):
        # Gemini calls the assistant's turns "model"
        contents = [
            self._genai.types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[self._genai.types.Part(text=_to_text(m["content"]))],
            )
            for m in messages
        ]
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=self._genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0,
            ),
        )
        meta = response.usage_metadata
        # thinking tokens are billed as output
        output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response.text or "")],
            usage=SimpleNamespace(input_tokens=meta.prompt_token_count or 0, output_tokens=output),
        )

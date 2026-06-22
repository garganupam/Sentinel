from google import genai
from google.genai import types

# Current GA Flash model, tuned for coding/agentic tasks (June 2026).
# Override via the `model` argument if a different model is ever needed.
DEFAULT_MODEL = "gemini-2.5-flash"


class LLMClient:
    """Thin wrapper around the Gemini API.

    Every M2 agent calls this client to get model output — it is the
    single place that talks to Gemini, mirroring how GitHubClient is the
    single place that talks to GitHub.

    Usage:
        client = LLMClient(api_key="...")
        text = client.generate("Say hello in one word.")
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        """Set up the Gemini client.

        Args:
            api_key: Gemini API key (from Google AI Studio).
            model:   Model ID used for every call made through this client.
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a single prompt to Gemini and return its text reply.

        Args:
            prompt:      The full prompt text to send.
            temperature: Lower = more consistent output. 0.2 default suits
                         review tasks, where repeatability matters more
                         than creative variation.

        Returns:
            The model's text response. Empty string if Gemini returns no
            text (e.g. response blocked by safety filters).

        Raises:
            google.genai.errors.APIError: On auth, quota, or network
                failures (carries .code and .message). The caller
                (agents.py) decides how to handle a failed agent.
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        response_schema,
        temperature: float = 0.2,
    ):
        """Send a prompt and force Gemini's reply to match a schema.

        Unlike `generate()`, this constrains the model's output at the API
        level — Gemini cannot return malformed JSON or an unexpected shape.
        Use this whenever the caller needs structured data, not free text.

        Args:
            prompt:          The full prompt text to send.
            response_schema: A Pydantic model, or `list[Model]` for an
                             array of objects. Enum-typed fields (e.g. a
                             Python `Enum`) are constrained to their valid
                             values automatically.
            temperature:     Lower = more consistent output.

        Returns:
            The parsed result already deserialised into `response_schema`'s
            type (e.g. a `list[Model]` instance). `None` if Gemini could
            not produce a value matching the schema.

        Raises:
            google.genai.errors.APIError: On auth, quota, or network
                failures. The caller decides how to handle a failed agent.
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        return response.parsed
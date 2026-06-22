import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    github_token: str
    webhook_secret: str
    gemini_api_key: str
    chroma_persist_path: str = "./chroma_db"
    rag_debug: bool = False

    @staticmethod
    def load() -> "Settings":
        load_dotenv()

        token = os.getenv("GITHUB_TOKEN", "").strip()
        secret = os.getenv("GITHUB_WEBHOOK_SECRET", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        chroma_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db").strip()
        rag_debug = os.getenv("RAG_DEBUG", "false").strip().lower() == "true"

        missing = [
            name
            for name, value in [
                ("GITHUB_TOKEN", token),
                ("GITHUB_WEBHOOK_SECRET", secret),
                ("GEMINI_API_KEY", gemini_key),
            ]
            if not value
        ]

        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}\n"
                "Fix: copy .env.example → .env and fill in the values, "
                "or export them in your shell."
            )

        return Settings(
            github_token=token,
            webhook_secret=secret,
            gemini_api_key=gemini_key,
            chroma_persist_path=chroma_path,
            rag_debug=rag_debug,
        )
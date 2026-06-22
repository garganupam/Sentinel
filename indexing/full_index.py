"""CLI entry point for a full RAG index of one repo.

Run once, manually, before relying on the incremental indexer — and again
later for the weekly safety-net re-index (same command, just on a schedule):

    python -m indexing.full_index owner/repo --ref main
"""
import argparse
import logging

from app.config import Settings
from app.github_client import GitHubClient

from .indexer import full_index
from .store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("pr-reviewer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full RAG index of one repo.")
    parser.add_argument("repo", help='Repository to index, e.g. "owner/repo".')
    parser.add_argument(
        "--ref",
        default="main",
        help='Branch, tag, or commit SHA to index at (default: "main").',
    )
    args = parser.parse_args()

    settings = Settings.load()
    github = GitHubClient(settings.github_token)
    store = VectorStore(settings.chroma_persist_path, repo_full_name=args.repo)

    logger.info("Starting full index of %s@%s", args.repo, args.ref)
    full_index(args.repo, args.ref, github=github, store=store)
    logger.info("Done.")


if __name__ == "__main__":
    main()
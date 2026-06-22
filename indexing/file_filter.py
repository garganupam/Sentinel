from pathlib import Path

# Extensions worth indexing — anything not in this list is skipped even if
# it isn't explicitly excluded below. Add new languages here.
INCLUDE_EXTENSIONS = {
    # Documentation
    ".md", ".mdx", ".rst", ".txt",
    # Python,CPP
    ".py",".cpp", ".c", ".h", ".hpp",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx",
    # Other languages
    ".go", ".java", ".rb", ".rs",
    # Config (small ones)
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    # Web
    ".html", ".css",
}

# Any path containing one of these directory names anywhere is skipped
# entirely, regardless of extension.
EXCLUDE_DIRS = {
    "node_modules", ".venv", "venv", "env",
    "dist", "build", ".git", "__pycache__",
    ".tox", "vendor", "coverage", ".next",
    "migrations", "generated",
}

# Checked before INCLUDE_EXTENSIONS — binaries, lock files, compiled
# artifacts, logs. Excluded even if the extension would otherwise pass.
EXCLUDE_EXTENSIONS = {
    # Binaries
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".exe",
    ".so", ".dylib", ".woff", ".ttf", ".eot",
    # Lock files
    ".lock",
    # Compiled
    ".pyc", ".pyo", ".class",
    # Logs / data
    ".log", ".csv",
}

# Exact filenames to skip regardless of extension.
EXCLUDE_FILENAMES = {
    "package-lock.json", "yarn.lock",
    ".env", ".env.local", ".DS_Store",
}

# Files larger than this are almost certainly not human-written
# documentation or source code.
MAX_FILE_SIZE_BYTES = 100_000  # 100 KB


def should_index(filepath: str, file_size: int) -> tuple[bool, str]:
    """Decide whether a file should be indexed for RAG retrieval.

    Args:
        filepath:  Repo-relative path, e.g. "app/models.py".
        file_size: File size in bytes.

    Returns:
        (should_index, reason) — reason is human-readable, useful for
        logging what was skipped and why during a full index run.
    """
    p = Path(filepath)

    for part in p.parts:
        if part in EXCLUDE_DIRS:
            return False, f"excluded directory: {part}"

    if p.name in EXCLUDE_FILENAMES:
        return False, f"excluded filename: {p.name}"

    if p.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False, f"excluded extension: {p.suffix}"

    if p.suffix.lower() not in INCLUDE_EXTENSIONS:
        return False, f"extension not in include list: {p.suffix}"

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"file too large: {file_size} bytes"

    return True, "included"
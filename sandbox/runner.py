"""sandbox/runner.py — Run a PR's code in an isolated Docker container.

Entry point: run_sandbox(repo_full_name, head_sha, github_token)
Returns:     SandboxResult — never raises. Any failure produces a safe
             "unverified" result rather than crashing the review pipeline.

Key design decisions:
  - Code is materialized on the HOST (GitHub zipball → tempdir) and mounted
    into the container read-only. The PAT never enters the container.
  - TWO-PHASE execution:
      Phase 1 (network ON):  pip install deps → /deps (host volume)
      Phase 2 (network OFF): run pytest using PYTHONPATH=/deps
    Both phases use the same python:3.11-slim image so compiled extensions
    (numpy, cryptography, etc.) are built for the right architecture.
  - network_disabled=True is only set on Phase 2 (test execution).
    Phase 1 has network but does nothing except pip install.
  - Source mounted read-only at /src in both phases.
    Phase 2 copies to writable /workspace before running tests.
  - Retry only when logs show infrastructure flakiness, not real failures.
  - duration_seconds + timed_out on every result.

Isolation guarantees (§5, non-negotiable):
  - No network during test execution (Phase 2).
  - Hard CPU / memory / wall-clock limits on both phases.
  - No secrets or host volumes beyond pre-materialized source + deps.
  - Ephemeral — fresh container per phase, destroyed after.
"""
import io
import logging
import os
import shutil
import tempfile
import textwrap
import time
import zipfile
from dataclasses import dataclass

import docker
import docker.errors
import requests

logger = logging.getLogger("pr-reviewer")

# --- Tunable limits -------------------------------------------------------
_MEMORY_LIMIT        = "512m"
_CPU_PERIOD          = 100_000      # microseconds
_CPU_QUOTA           = 50_000       # 50% of one CPU
_INSTALL_TIMEOUT_S   = 120          # Phase 1 (pip install) hard limit
_TEST_TIMEOUT_S      = 300          # Phase 2 (pytest) hard limit
_MAX_LOG_CHARS       = 8_000

_BASE_IMAGE = "python:3.11-slim"

# Log patterns that suggest infrastructure/timing flakiness — not real failures.
_FLAKY_LOG_PATTERNS = [
    "connection refused", "connection reset", "temporarily unavailable",
    "resource busy", "too many open files", "cannot connect",
    "network unreachable", "timed out", "socket.timeout", "readtimeout",
]

# Python project markers — any one present = definitely Python.
_PYTHON_MARKERS = {
    "requirements.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "Pipfile", "pytest.ini", "tox.ini",
}

# Dirs to skip during recursive test-file search.
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules",
    "venv", ".venv", "env", ".env",
    "dist", "build", ".tox",
}

# Phase 1 script — runs WITH network, installs into /deps (mounted host dir).
#
# Security principle: NEVER install the project itself (pip install /src or
# pip install -e /src) — that triggers build hooks which may execute arbitrary
# project code. Instead:
#   - requirements.txt → pip install -r (safest, no project build hooks)
#   - pyproject.toml   → parse [project.dependencies] via tomllib (stdlib),
#                         install just the listed deps, never the project itself
#   - setup.py only    → skip project install entirely (setup.py IS
#                         arbitrary Python code), just ensure pytest exists
# Phase 2 sets PYTHONPATH=/deps:/workspace so the project source is
# importable without ever being installed.
_INSTALL_SCRIPT = textwrap.dedent("""
    set -e

    if [ -f /src/requirements.txt ]; then
        # Safest path — no project code executes
        echo "SANDBOX_INSTALL: requirements.txt"
        pip install -r /src/requirements.txt --target /deps -q 2>&1 \
            || { echo "SANDBOX_INSTALL_FAILED"; exit 2; }

    elif [ -f /src/pyproject.toml ]; then
        # Parse [project.dependencies] from pyproject.toml without invoking
        # the project's build system. tomllib is stdlib since Python 3.11.
        echo "SANDBOX_INSTALL: pyproject.toml (deps only, no build hooks)"
        python3 -c "
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
try:
    with open('/src/pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    deps = data.get('project', {}).get('dependencies', [])
    if deps:
        with open('/tmp/pyproject_deps.txt', 'w') as out:
            out.write('\\n'.join(deps))
        sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f'toml parse error: {e}', file=sys.stderr)
    sys.exit(1)
"
        if [ $? -eq 0 ] && [ -s /tmp/pyproject_deps.txt ]; then
            pip install -r /tmp/pyproject_deps.txt --target /deps -q 2>&1 \
                || { echo "SANDBOX_INSTALL_FAILED"; exit 2; }
        else
            echo "SANDBOX_INSTALL: no [project.dependencies] found in pyproject.toml — skipping"
        fi

    elif [ -f /src/setup.py ]; then
        # setup.py is arbitrary Python — we do NOT run it.
        # Only pytest is installed; project source is on PYTHONPATH in Phase 2.
        echo "SANDBOX_INSTALL: setup.py found but not executed (security boundary)"
    fi

    # Always ensure pytest is available (separate from project deps)
    pip install pytest --target /deps -q 2>&1
    exit 0
""").strip()

# Phase 2 script — runs WITHOUT network. Uses pre-installed /deps.
_TEST_SCRIPT = textwrap.dedent("""
    set -e
    mkdir -p /workspace
    cp -r /src/. /workspace/
    cd /workspace

    if find . -name "test_*.py" -not -path "./.git/*" | grep -q .; then
        echo "SANDBOX_TEST_RUNNER: pytest (test_*.py)"
        PYTHONPATH=/deps:/workspace python -m pytest -q 2>&1; exit $?
    elif find . -name "*_test.py" -not -path "./.git/*" | grep -q .; then
        echo "SANDBOX_TEST_RUNNER: pytest (*_test.py)"
        PYTHONPATH=/deps:/workspace python -m pytest -q 2>&1; exit $?
    elif [ -f pytest.ini ] || [ -f setup.cfg ] || [ -f tox.ini ]; then
        echo "SANDBOX_TEST_RUNNER: pytest (config)"
        PYTHONPATH=/deps:/workspace python -m pytest -q 2>&1; exit $?
    else
        echo "SANDBOX_NO_TESTS_FOUND"
        exit 3
    fi
""").strip()


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandbox run.

    Attributes:
        passed:           True only if build succeeded AND tests ran AND all passed.
        build_failed:     True if pip install failed (distinct from test failure).
        had_tests:        False if no test files found or unsupported language.
        logs:             Trimmed combined stdout+stderr.
        tested_sha:       Exact commit SHA tested — always in the PR report.
        duration_seconds: Wall-clock time for the full run (both phases).
        timed_out:        True if killed by the hard timeout.
        error:            Reason when the sandbox infrastructure itself failed.
    """
    passed:           bool
    build_failed:     bool
    had_tests:        bool
    logs:             str
    tested_sha:       str
    duration_seconds: float = 0.0
    timed_out:        bool  = False
    error:            str   = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sandbox(
    repo_full_name: str,
    head_sha: str,
    github_token: str,
) -> SandboxResult:
    """Materialize PR code, install deps, run tests — never raises.

    Args:
        repo_full_name: "owner/repo".
        head_sha:       PR head commit SHA — the exact code to test.
        github_token:   PAT used only for the GitHub zipball download on the
                        host. Never passed into either container phase.
    """
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.DockerException as exc:
        logger.error("Docker unavailable: %s", exc)
        return _error_result(head_sha, f"Docker unavailable: {exc}")

    tmpdir = None
    try:
        tmpdir, src_dir = _materialize_repo(repo_full_name, head_sha, github_token)
        logger.info("Sandbox: materialized %s@%s", repo_full_name, head_sha[:8])
    except Exception as exc:
        logger.error("Sandbox: repo download failed: %s", exc)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return _error_result(head_sha, f"Repo download failed: {exc}")

    if not _is_python_project(src_dir):
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("Sandbox: unsupported project type")
        return SandboxResult(
            passed=False, build_failed=False, had_tests=False,
            logs="", tested_sha=head_sha,
            error="Unsupported project type. Only Python projects are supported in M4.",
        )

    try:
        result = _run_with_retry(client, src_dir, head_sha)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(
        "Sandbox: passed=%s build_failed=%s had_tests=%s "
        "timed_out=%s duration=%.1fs sha=%s",
        result.passed, result.build_failed, result.had_tests,
        result.timed_out, result.duration_seconds, head_sha[:8],
    )
    return result


# ---------------------------------------------------------------------------
# Host-side materialisation
# ---------------------------------------------------------------------------

def _materialize_repo(
    repo_full_name: str,
    head_sha: str,
    github_token: str,
) -> tuple:
    """Download repo zipball at head_sha to a host temp dir.

    Streams the response to a temp file first (avoids loading large repos
    entirely into RAM before extraction).

    Returns (tmpdir, src_dir) — caller owns cleanup of tmpdir.
    """
    url = f"https://api.github.com/repos/{repo_full_name}/zipball/{head_sha}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }

    tmpdir = tempfile.mkdtemp(prefix="pr-reviewer-sandbox-")
    zip_path = os.path.join(tmpdir, "repo.zip")

    # Stream to temp file — avoids holding large zip in RAM
    with requests.get(url, headers=headers, timeout=60, stream=True) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    os.remove(zip_path)

    # GitHub zipball extracts into one top-level dir: "owner-repo-<sha>/"
    entries = [
        e for e in os.listdir(tmpdir)
        if os.path.isdir(os.path.join(tmpdir, e))
    ]
    src_dir = os.path.join(tmpdir, entries[0]) if len(entries) == 1 else tmpdir
    return tmpdir, src_dir


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _is_python_project(src_dir: str) -> bool:
    """Three-tier Python project detection.

    Tier 1: explicit project markers (requirements.txt, pyproject.toml, etc.)
    Tier 2: tests/ directory containing at least one .py file
    Tier 3: recursive walk (max depth 3) for test_*.py / *_test.py files
    """
    # Tier 1
    for marker in _PYTHON_MARKERS:
        if os.path.exists(os.path.join(src_dir, marker)):
            return True

    # Tier 2
    tests_dir = os.path.join(src_dir, "tests")
    if os.path.isdir(tests_dir):
        if any(f.endswith(".py") for f in os.listdir(tests_dir)):
            return True

    # Tier 3
    for root, dirs, files in os.walk(src_dir):
        depth = root[len(src_dir):].count(os.sep)
        if depth >= 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if fname.endswith(".py") and (
                fname.startswith("test_") or fname.endswith("_test.py")
            ):
                return True

    return False


# ---------------------------------------------------------------------------
# Two-phase container execution
# ---------------------------------------------------------------------------

def _run_once(client, src_dir: str, head_sha: str) -> SandboxResult:
    """Phase 1 (install) + Phase 2 (test) — both ephemeral containers."""
    t_start = time.monotonic()
    deps_dir = tempfile.mkdtemp(prefix="pr-reviewer-deps-")

    try:
        # --- Phase 1: install deps (network ON, into host deps_dir) ---
        build_ok, build_logs = _run_phase(
            client=client,
            script=_INSTALL_SCRIPT,
            volumes={
                src_dir:  {"bind": "/src",  "mode": "ro"},
                deps_dir: {"bind": "/deps", "mode": "rw"},
            },
            network_disabled=False,
            timeout=_INSTALL_TIMEOUT_S,
        )

        if not build_ok:
            return SandboxResult(
                passed=False, build_failed=True, had_tests=False,
                logs=build_logs[-_MAX_LOG_CHARS:], tested_sha=head_sha,
                duration_seconds=round(time.monotonic() - t_start, 2),
            )

        # --- Phase 2: run tests (network OFF, deps pre-installed) ---
        timed_out, exit_code, test_logs = _run_phase_raw(
            client=client,
            script=_TEST_SCRIPT,
            volumes={
                src_dir:  {"bind": "/src",  "mode": "ro"},
                deps_dir: {"bind": "/deps", "mode": "ro"},  # read-only in test phase
            },
            network_disabled=True,
            timeout=_TEST_TIMEOUT_S,
        )
        duration = round(time.monotonic() - t_start, 2)
        logs = (build_logs + "\n" + test_logs)[-_MAX_LOG_CHARS:]

        if timed_out:
            return SandboxResult(
                passed=False, build_failed=False, had_tests=False,
                logs=logs, tested_sha=head_sha, duration_seconds=duration,
                timed_out=True, error=f"Timed out after {_TEST_TIMEOUT_S}s",
            )

        if exit_code == 0:
            return SandboxResult(passed=True,  build_failed=False, had_tests=True,  logs=logs, tested_sha=head_sha, duration_seconds=duration)
        elif exit_code == 1:
            return SandboxResult(passed=False, build_failed=False, had_tests=True,  logs=logs, tested_sha=head_sha, duration_seconds=duration)
        elif exit_code == 2:
            return SandboxResult(passed=False, build_failed=True,  had_tests=False, logs=logs, tested_sha=head_sha, duration_seconds=duration)
        elif exit_code == 3:
            return SandboxResult(passed=False, build_failed=False, had_tests=False, logs=logs, tested_sha=head_sha, duration_seconds=duration)
        else:
            return _error_result(head_sha, f"Unexpected exit code {exit_code}", duration, logs)

    except Exception as exc:
        logger.warning("Sandbox unexpected error: %s", exc)
        return _error_result(head_sha, f"Unexpected error: {exc}", round(time.monotonic() - t_start, 2))
    finally:
        shutil.rmtree(deps_dir, ignore_errors=True)


def _run_phase(client, script, volumes, network_disabled, timeout):
    """Run one container phase. Returns (success: bool, logs: str)."""
    timed_out, exit_code, logs = _run_phase_raw(client, script, volumes, network_disabled, timeout)
    success = (not timed_out) and (exit_code == 0)
    return success, logs


def _run_phase_raw(client, script, volumes, network_disabled, timeout):
    """Run one container phase. Returns (timed_out, exit_code, logs)."""
    timed_out = False
    exit_code = -1

    container = client.containers.run(
        image=_BASE_IMAGE,
        command=["bash", "-c", script],
        volumes=volumes,
        network_disabled=network_disabled,
        mem_limit=_MEMORY_LIMIT,
        cpu_period=_CPU_PERIOD,
        cpu_quota=_CPU_QUOTA,
        security_opt=["no-new-privileges"],
        detach=True,
        remove=False,
    )

    try:
        wait_result = container.wait(timeout=timeout)
        exit_code = wait_result.get("StatusCode", -1)
    except Exception:
        timed_out = True
        try:
            container.kill()
        except Exception:
            pass

    logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
    container.remove(force=True)
    return timed_out, exit_code, logs


# ---------------------------------------------------------------------------
# Retry (flaky-only)
# ---------------------------------------------------------------------------

def _run_with_retry(client, src_dir: str, head_sha: str) -> SandboxResult:
    """Run once; retry exactly once ONLY if logs show infrastructure flakiness."""
    result = _run_once(client, src_dir, head_sha)

    if (
        not result.passed
        and not result.build_failed
        and result.had_tests
        and not result.timed_out
        and _looks_flaky(result.logs)
    ):
        logger.info("Sandbox: flaky pattern in logs — retrying once")
        retry = _run_once(client, src_dir, head_sha)
        logger.info("Sandbox: retry result: passed=%s", retry.passed)
        return retry

    return result


def _looks_flaky(logs: str) -> bool:
    lower = logs.lower()
    return any(p in lower for p in _FLAKY_LOG_PATTERNS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(tested_sha, error, duration=0.0, logs="") -> SandboxResult:
    return SandboxResult(
        passed=False, build_failed=False, had_tests=False,
        logs=logs, tested_sha=tested_sha,
        duration_seconds=round(duration, 2), error=error,
    )
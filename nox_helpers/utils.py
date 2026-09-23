from __future__ import annotations

import json
import os
from pathlib import Path

import nox

from .matrix import MatrixEntry

__all__ = ("list_dependencies", "write_github_output")


def list_dependencies(session: nox.Session) -> None:
    """List the packages installed in a session's environment."""
    if session.venv_backend == "uv":
        session.run(
            "uv",
            "pip",
            "list",
            "--python",
            session.virtualenv.location,
            external=True,
        )
    else:
        session.run("python", "-m", "pip", "list")


def write_github_output(session: nox.Session, matrix: tuple[MatrixEntry, ...]) -> None:
    """Write the GitHub Actions matrix to the environment."""

    outputs = [json.dumps(entry.github_matrix_entry) for entry in matrix]
    if (github_output := os.getenv("GITHUB_OUTPUT")) is None:
        session.log("GITHUB_OUTPUT environment variable is not set, listing matrix:")
        for output in outputs:
            session.log(f"    {output}")

        session.error("GITHUB_OUTPUT environment variable is not set")
        return  # For mypy type checking the error should stop nox

    with Path(github_output).open("a", encoding="utf-8") as out:
        out.write(f"matrix={outputs}\n")

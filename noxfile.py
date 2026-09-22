from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import nox
import requests

# Make nox default to uv if its available, if not fallback on virtualenv
nox.options.default_venv_backend = "uv|virtualenv"


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    """Represents an entry in a github workflow job matrix."""

    DEFAULT_RUNS_ON: ClassVar[str] = "ubuntu-latest"
    MACOS_RUNS_ON: ClassVar[str] = "macos-latest"

    session: str
    """The session name for this matrix entry."""
    python: str
    """The Python version for this matrix entry."""
    args: tuple[str, ...] = field(default_factory=tuple)
    """The positional arguments for this matrix entry."""
    options: tuple[str, ...] = field(default_factory=tuple)
    """The options for this matrix entry."""
    runs_on: str = field(default=DEFAULT_RUNS_ON)
    """The runner environment for this matrix entry."""

    @property
    def nox_id(self) -> str:
        """Return the factor string for this matrix entry."""
        nox_id = f"py{self.python}"

        if self.args:
            nox_id = f"{'-'.join(self.args)}--{nox_id}"

        if self.options:
            nox_id = f"{nox_id}-{'-'.join(self.options)}"

        return nox_id

    @property
    def nox_param(self) -> nox.param:
        """Return the nox parameter for this matrix entry."""
        return nox.param(self, id=self.nox_id)

    @property
    def session_name(self) -> str:
        """Return the session name for this matrix entry."""
        return f"{self.session}({self.nox_id})"

    @property
    def job_name(self) -> str:
        """Return the job name for this matrix entry for github"""
        if self.runs_on == self.DEFAULT_RUNS_ON:
            return f"{self.nox_id}"

        return f"{self.nox_id} ({self.runs_on})"

    @property
    def session_flags(self) -> tuple[str, ...]:
        """Return posargs (flags) for this"""
        return tuple(f"--{option}" for option in self.options if option)

    @property
    def github_matrix_entry(self) -> dict[str, str]:
        """Return the github matrix entry for this matrix entry."""
        return {
            "name": self.job_name,
            "session": self.session_name,
            "python": self.python,
            "runs-on": self.runs_on,
        }

    def run_session(self, session: nox.Session) -> None:
        """Run the session with the appropriate Python version and flags."""
        session.notify(f"{self.session}-{self.python}", posargs=self.session_flags)


@dataclass(frozen=True, slots=True)
class PythonVersions:
    """
    Class to manage and retrieve the Python versions for this project.

    This class is a singleton; every call to ``PythonVersions()`` returns the
    same instance, so the underlying network request and version resolution
    only happen once per process.
    """

    PYTHON_RELEASES_URL: ClassVar[str] = (
        "https://www.python.org/api/v2/downloads/release/"
    )
    PYTHON_RELEASE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^Python (?P<version>\d+\.\d+\.\d+)$"
    )
    _instance: ClassVar[PythonVersions | None] = None

    versions: tuple[str, ...] = field(init=False)
    """The versions supported by this project"""
    oldest: str = field(init=False)
    """The oldest supported Python version for this project"""
    newest: str = field(init=False)
    """The newest supported Python version for this project"""
    github_ci_factors: tuple[MatrixEntry, ...] = field(init=False)
    """The nox session parameter matrix entries for the github ci workflow"""

    def __new__(cls) -> PythonVersions:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __post_init__(self) -> None:
        if hasattr(self, "versions"):
            return
        object.__setattr__(self, "versions", self._get_versions())
        object.__setattr__(self, "oldest", self.versions[0])
        object.__setattr__(self, "newest", self.versions[-1])
        object.__setattr__(
            self, "github_ci_factors", tuple(self._get_github_ci_factors())
        )

    def _latest_stable_version(self) -> str:
        """Return the latest stable Python version published on python.org."""
        response = requests.get(
            self.PYTHON_RELEASES_URL,
            params={"is_published": "true"},
            timeout=30,
        )
        response.raise_for_status()

        stable_versions: list[tuple[tuple[int, ...], str]] = []
        for release in response.json():
            match = self.PYTHON_RELEASE_PATTERN.fullmatch(release["name"])
            if match is not None and not release["pre_release"]:
                version = match.group("version")
                stable_versions.append((tuple(map(int, version.split("."))), version))

        if not stable_versions:
            message = "python.org returned no stable Python releases"
            raise RuntimeError(message)

        return max(stable_versions)[1]

    def _get_versions(self) -> tuple[str, ...]:
        """
        Return the Python versions to test, read from this project's pyproject.toml.
        """
        pyproject = nox.project.load_toml("pyproject.toml")
        # mypy is not looking into nox for the types
        return tuple(
            nox.project.python_versions(  # type: ignore[no-any-return]
                pyproject,
                max_version=self._latest_stable_version(),
            )
        )

    def _get_github_ci_factors(self) -> list[MatrixEntry]:
        """
        Return the fixed set of CI factors run by ``github_ci``.

        This covers: the oldest supported Python with ``--oldest``, the newest
        supported Python with ``--dev``/``--editable``/``--coverage``, and every
        other supported Python with no extra flags.
        """
        return [
            MatrixEntry(session="test", python=self.oldest, options=("oldest",)),
            MatrixEntry(session="test", python=self.newest, options=("dev",)),
            MatrixEntry(session="test", python=self.newest, options=("editable",)),
            MatrixEntry(session="test", python=self.newest, options=("coverage",)),
            *[
                MatrixEntry(session="test", python=version)
                for version in self.versions
                if version != self.newest
            ],
        ]

    @property
    def github_ci_matrix(self) -> tuple[MatrixEntry, ...]:
        return (
            MatrixEntry(
                session="test", python=self.newest, runs_on=MatrixEntry.MACOS_RUNS_ON
            ),
            *self.github_ci_factors,
        )


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


@nox.session(python=PythonVersions().versions)
def test(session: nox.Session) -> None:
    """Run the unit tests with the given command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="nox -s test --",
        allow_abbrev=False,
        description="Run the gwcs test suite.",
    )

    # Build the parser, adding the options as we go
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )
    parser.add_argument(
        "--xdist",
        action="store_true",
        help="Enable pytest-xdist for parallel test execution",
    )
    # Ensure that only one of --dev or --oldest can be specified
    dependency_group = parser.add_mutually_exclusive_group()
    dependency_group.add_argument(
        "--dev", action="store_true", help="Install development dependencies"
    )
    dependency_group.add_argument(
        "--oldest",
        action="store_true",
        help="Install the oldest compatible dependencies",
    )
    # Ensure that only one of --editable or --wheel can be specified
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--editable",
        action="store_true",
        help="Install gwcs in editable mode",
    )
    install_group.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Install gwcs from a built wheel file instead of the source tree",
    )
    args, pytest_args = parser.parse_known_args(session.posargs)

    # Determine the dependencies that nox needs to install in the virtual environment
    if args.wheel is not None:
        dependencies = [f"{args.wheel}[test]"]
    elif args.editable:
        dependencies = ["-e", ".[test]"]
    else:
        dependencies = [".[test]"]
    if args.dev:
        dependencies[:0] = ["-r", "requirements-dev.txt"]
    if args.oldest:
        if session.venv_backend != "uv":
            session.error("--oldest requires the uv backend")
        dependencies[:0] = ["--resolution", "lowest-direct"]
    if args.coverage:
        dependencies.append("pytest-cov")
    if args.xdist:
        dependencies.append("pytest-xdist")
    session.install(*dependencies)

    # Configure the additional pytest arguments based on the command-line options
    if args.coverage:
        pytest_args[:0] = [
            "--cov=.",
            "--cov-config=pyproject.toml",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
    if args.xdist:
        pytest_args[:0] = ["-n", "auto"]

    list_dependencies(session)
    session.run("pytest", *pytest_args)


@nox.session(venv_backend="none")
@nox.parametrize(
    "factor",
    PythonVersions().github_ci_factors,
    ids=[entry.nox_id for entry in PythonVersions().github_ci_factors],
)
def github_ci(session: nox.Session, factor: MatrixEntry) -> None:
    """
    Dispatch a short CI factor (e.g. ``py3.13-dev``) to the ``test`` session.

    This exists so CI configuration only needs to name a small, fixed subset
    of the ``test`` session's many possible configurations. For example,
    ``nox -e github_ci-py3.13-dev`` runs ``nox -e test-3.13 -- --dev``.
    """
    factor.run_session(session)


@nox.session(venv_backend="none")
def github_ci_matrix(session: nox.Session) -> None:
    """
    Session that writes out the matrix to the environment for the ci.yml workflow
    """
    write_github_output(session, PythonVersions().github_ci_matrix)

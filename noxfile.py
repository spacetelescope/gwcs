from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
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
        return self.args + tuple(f"--{option}" for option in self.options if option)

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
        session.log(
            f"Running session {self.session} with on {self.python} with flags "
            f"{self.session_flags}"
        )
        session.notify(f"{self.session}-{self.python}", posargs=self.session_flags)


@dataclass(frozen=True, slots=True)
class Downstream:
    """Description of a downstream package to test gwcs against."""

    JWST_CRDS: ClassVar[dict[str, str]] = {
        "CRDS_SERVER_URL": "https://jwst-crds.stsci.edu"
    }
    ROMAN_CRDS: ClassVar[dict[str, str]] = {
        "CRDS_SERVER_URL": "https://roman-crds.stsci.edu"
    }

    name: str
    repo: str
    branch: str
    extra: str | None
    env: dict[str, str] = field(default_factory=dict)
    # Extra arguments passed to pytest when running this package's own tests.
    extra_pytest_args: tuple[str, ...] = ()
    # Packages that are only tested when CI explicitly opts in, e.g. when a pull
    # request carries the "Downstream CI" label.
    label_only: bool = True
    options: tuple[str, ...] = ()

    @property
    def matrix_entry(self) -> MatrixEntry:
        """Return the matrix entry for this downstream package."""
        return MatrixEntry(
            session="downstream",
            python=PythonVersions().default,
            args=(self.name,),
            options=self.options,
        )


DOWNSTREAM = {
    "jwst": Downstream(
        "jwst",
        "https://github.com/spacetelescope/jwst.git",
        "main",
        "test",
        Downstream.JWST_CRDS,
        label_only=False,
        options=("xdist",),
    ),
    "romancal": Downstream(
        "romancal",
        "https://github.com/spacetelescope/romancal.git",
        "main",
        "test",
        Downstream.ROMAN_CRDS,
        label_only=False,
        options=("xdist",),
    ),
    "romanisim": Downstream(
        "romanisim",
        "https://github.com/spacetelescope/romanisim.git",
        "main",
        "test",
        Downstream.ROMAN_CRDS,
        label_only=False,
        options=("xdist",),
    ),
    "specutils": Downstream(
        "specutils", "https://github.com/astropy/specutils.git", "main", "test"
    ),
    "dkist": Downstream(
        "dkist",
        "https://github.com/DKISTDC/dkist.git",
        "main",
        "tests",
        extra_pytest_args=("--benchmark-skip",),
        options=("xdist",),
    ),
    "ndcube": Downstream(
        "ndcube",
        "https://github.com/sunpy/ndcube.git",
        "main",
        "dev",
        options=("xdist",),
    ),
    "stcal": Downstream(
        "stcal",
        "https://github.com/spacetelescope/stcal.git",
        "main",
        "test",
        Downstream.JWST_CRDS,
        label_only=False,
        options=("xdist",),
    ),
}


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
    default: str = field(init=False)
    """The default Python version for this project"""
    github_test_factors: tuple[MatrixEntry, ...] = field(init=False)
    """The nox session parameter matrix entries for the github test workflow"""

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
        object.__setattr__(self, "default", self.versions[-2])
        object.__setattr__(
            self, "github_test_factors", tuple(self._get_github_test_factors())
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

    def _get_github_test_factors(self) -> list[MatrixEntry]:
        """
        Return the fixed set of CI factors run by ``github_test``.

        This covers: the oldest supported Python with ``--oldest``, the newest
        supported Python with ``--dev``/``--editable``/``--coverage``, and every
        other supported Python with no extra flags.
        """
        return [
            MatrixEntry(session="test", python=self.oldest, options=("oldest",)),
            MatrixEntry(session="test", python=self.newest, options=("dev",)),
            MatrixEntry(session="test", python=self.default, options=("editable",)),
            MatrixEntry(session="test", python=self.default, options=("coverage",)),
            *[
                MatrixEntry(session="test", python=version)
                for version in self.versions
                if version != self.default
            ],
        ]

    @property
    def github_test_matrix(self) -> tuple[MatrixEntry, ...]:
        return (
            MatrixEntry(
                session="test", python=self.default, runs_on=MatrixEntry.MACOS_RUNS_ON
            ),
            *self.github_test_factors,
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


def _add_standard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard command-line arguments to the parser."""
    parser.add_argument(
        "--xdist",
        action="store_true",
        help="Enable pytest-xdist for parallel test execution",
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


def _install_gwcs(
    session: nox.Session,
    args: argparse.Namespace,
    install_args: list[str] | None = None,
) -> None:
    """Install gwcs in the virtual environment based on the command-line arguments."""
    install_args = install_args or []

    # Setup the install arguments for gwcs itself
    if args.wheel is not None:
        install_args += [f"{args.wheel}[test]"]
    elif args.editable:
        install_args += ["-e", ".[test]"]
    else:
        install_args += [".[test]"]

    session.log("Installing the gwcs package for testing:")

    # Add pytest-xdist if requested
    if args.xdist:
        session.log("  Including pytest-xdist for parallel test execution")
        install_args += ["pytest-xdist"]

    session.install(*install_args)


def _init_pytest_arguments(session: nox.Session, args: argparse.Namespace) -> list[str]:
    """Initialize the pytest arguments based on the command-line options."""
    session.log("Running tests:")

    arguments: list[str] = []
    if args.xdist:
        session.log("  Including pytest-xdist for parallel test execution")
        arguments += ["-n", "auto"]

    return arguments


@nox.session(python=PythonVersions().versions)
def test(session: nox.Session) -> None:
    """Run the unit tests with the given command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="nox -s test --",
        allow_abbrev=False,
        description="Run the gwcs test suite.",
    )
    _add_standard_arguments(parser)

    # Build the parser, adding the options as we go
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
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
    args, pytest_args = parser.parse_known_args(session.posargs)
    install_args: list[str] = []

    # Prepare the install arguments for dev/oldest if required
    if args.dev:
        install_args += ["-r", "requirements-dev.txt"]
    if args.oldest:
        if session.venv_backend != "uv":
            session.error("--oldest requires the uv backend")

        install_args += ["--resolution", "lowest-direct"]

    _install_gwcs(session, args, install_args)

    # Install coverage if requested
    if args.coverage:
        session.log("Installing coverage dependencies:")
        session.install("pytest-cov")

    list_dependencies(session)

    # Configure the pytest arguments
    arguments = _init_pytest_arguments(session, args)
    if args.coverage:
        session.log("  Enabling coverage reporting")
        arguments += [
            "--cov=.",
            "--cov-config=pyproject.toml",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
    arguments += pytest_args

    session.run("pytest", *arguments)


@nox.session(python=PythonVersions().versions)
def downstream(session: nox.Session) -> None:
    """
    Session to test downstream compatibility.
    """

    parser = argparse.ArgumentParser(
        prog="nox -s downstream --",
        allow_abbrev=False,
        description="Run a downstream package's tests against this version of gwcs.",
    )
    parser.add_argument(
        "package",
        choices=sorted(DOWNSTREAM),
        help="Downstream package to test against gwcs",
    )
    _add_standard_arguments(parser)

    args, pytest_args = parser.parse_known_args(session.posargs)

    downstream_package = DOWNSTREAM[args.package]

    # Clone into a temporary directory so stale state cannot leak between runs
    # and the repo working tree is never touched.
    with tempfile.TemporaryDirectory(prefix="gwcs-downstream-") as tmp_dir:
        downstream_dir = Path(tmp_dir) / args.package
        session.run(
            "git",
            "clone",
            "--branch",
            downstream_package.branch,
            # A blobless clone keeps the tags that setuptools-scm needs to
            # determine a version, without paying for the full file history.
            "--filter=blob:none",
            downstream_package.repo,
            str(downstream_dir),
            external=True,
        )

        # Install the downstream package before gwcs so gwcs's pin wins if
        # they conflict.
        with session.chdir(downstream_dir):
            session.log(f"Installing the downstream package: {args.package}")
            # -e fixes issues with C extensions not being available for some reason
            session.install(
                "-e",
                f".[{downstream_package.extra}]"
                if downstream_package.extra is not None
                else ".",
            )
        _install_gwcs(session, args)

        list_dependencies(session)

        # Configure the pytest arguments
        arguments = _init_pytest_arguments(session, args)
        arguments += list(downstream_package.extra_pytest_args)
        arguments += pytest_args

        # Run the tests as if it were the downstream package running them
        with session.chdir(downstream_dir):
            session.run("pytest", *arguments, env=downstream_package.env)


@nox.session(venv_backend="none")
@nox.parametrize(
    "factor",
    [factor.nox_param for factor in PythonVersions().github_test_factors],
)
def github_test(session: nox.Session, factor: MatrixEntry) -> None:
    """
    Dispatch a short CI factor (e.g. ``py3.13-dev``) to the ``test`` session.

    This exists so CI configuration only needs to name a small, fixed subset
    of the ``test`` session's many possible configurations. For example,
    ``nox -e github_test(py3.13-dev)`` runs ``nox -e test-3.13 -- --dev``.
    """
    factor.run_session(session)


@nox.session(venv_backend="none")
def github_test_matrix(session: nox.Session) -> None:
    """
    Session that writes out the matrix to the environment for the test.yml workflow
    """
    write_github_output(session, PythonVersions().github_test_matrix)


@nox.session(venv_backend="none")
@nox.parametrize(
    "factor",
    [package.matrix_entry.nox_param for package in DOWNSTREAM.values()],
)
def github_downstream(session: nox.Session, factor: MatrixEntry) -> None:
    """
    Dispatch a short CI factor (e.g. ``jwst--py3.13-xdist``) to ``downstream`` session.

    This exists so CI configuration only needs to name a small, fixed subset
    of the ``test`` session's many possible configurations. For example,
    ``nox -e github_downstream(jwst--py3.13-xdist)`` runs
    ``nox -e downstream-3.13 -- jwst --xdist``.
    """
    factor.run_session(session)


@nox.session(venv_backend="none")
def github_downstream_matrix(session: nox.Session) -> None:
    """
    Session that writes out the matrix for the downstream.yml workflow
    """
    write_github_output(
        session, tuple(package.matrix_entry for package in DOWNSTREAM.values())
    )

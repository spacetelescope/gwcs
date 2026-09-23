from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nox

# nox loads this file via importlib.util.spec_from_file_location, which does not
# add its directory to sys.path, so the sibling nox_helpers package needs a hand.
# TODO: If/when nox_helpers is moved into its own package this can be a direct import
sys.path.insert(0, str(Path(__file__).parent))

from nox_helpers import (
    DOWNSTREAM,
    MatrixEntry,
    PythonVersions,
    list_dependencies,
    write_github_output,
)

# Make nox default to uv if its available, if not fallback on virtualenv
nox.options.default_venv_backend = "uv|virtualenv"


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
    # This uses the tempfile module instead of the session.create_tmp() method
    # so that the clone is performed freshly each time the session is run.
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


@nox.session(python=PythonVersions().versions)
def build(session: nox.Session) -> None:
    """Build the sdist and wheel into dist/."""
    parser = argparse.ArgumentParser(
        prog="nox -s build --",
        allow_abbrev=False,
        description="Build the sdist and wheel, optionally testing the wheel.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test session installing gwcs from the built wheel",
    )
    args, test_posargs = parser.parse_known_args(session.posargs)

    dist = Path("dist")
    shutil.rmtree(dist, ignore_errors=True)

    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", *(str(p) for p in dist.glob("*")))

    if args.test:
        wheels = sorted(dist.glob("*.whl"))
        if not wheels:
            session.error("No wheel found in dist/ to test")
        session.notify("test", posargs=["--wheel", str(wheels[-1]), *test_posargs])


@nox.session(name="check-style")
def check_style(session: nox.Session) -> None:
    """Run all style and file checks with prek."""
    default_args = ("--color", "always", "--all-files", "--show-diff-on-failure")

    session.install("prek")
    session.run("prek", "prepare-hooks")
    session.run("prek", "run", *(session.posargs or default_args))


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

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

import nox
import requests

from .matrix import MatrixEntry

__all__ = ("PythonVersions",)


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
            cls._instance = object.__new__(cls)
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
                session="test", python=self.newest, runs_on=MatrixEntry.MACOS_RUNS_ON
            ),
            *self.github_test_factors,
        )

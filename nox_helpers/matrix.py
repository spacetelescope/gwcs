from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import nox

__all__ = ("MatrixEntry",)


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

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from .matrix import MatrixEntry
from .python import PythonVersions

__all__ = ("DOWNSTREAM",)


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


DOWNSTREAM = MappingProxyType(
    {
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
)

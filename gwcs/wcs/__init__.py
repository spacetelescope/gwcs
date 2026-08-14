"""
This subpackage contains the Main WCS class for GWCS along with its building blocks.
"""

from ._exception import GwcsBoundingBoxWarning, NoConvergence
from ._pipeline import DirectionalWCS, Pipeline, _BasePipeline
from ._step import Step
from ._wcs import WCS

__all__ = [
    "WCS",
    "DirectionalWCS",
    "GwcsBoundingBoxWarning",
    "NoConvergence",
    "Pipeline",
    "Step",
    # See note in _pipeline.py about why _BasePipeline is included here.
    "_BasePipeline",
]

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import ManifestExtension
from .converters.wcs import (
    CelestialFrameConverter, CompositeFrameConverter, FrameConverter,
    Frame2DConverter, SpectralFrameConverter, StepConverter,
    StokesFrameConverter, TemporalFrameConverter, WCSConverter,
)
from .converters.selector import (
    LabelMapperConverter, RegionsSelectorConverter
)
from .converters.spectroscopy import (
    GratingEquationConverter, SellmeierGlassConverter, SellmeierZemaxConverter,
    Snell3DConverter
)
from .converters.geometry import (
    DirectionCosinesConverter, SphericalCartesianConverter
)


WCS_MODEL_CONVERTERS = [
    CelestialFrameConverter(),
    CompositeFrameConverter(),
    FrameConverter(),
    Frame2DConverter(),
    SpectralFrameConverter(),
    StepConverter(),
    StokesFrameConverter(),
    TemporalFrameConverter(),
    WCSConverter(),
    LabelMapperConverter(),
    RegionsSelectorConverter(),
    GratingEquationConverter(),
    SellmeierGlassConverter(),
    SellmeierZemaxConverter(),
    Snell3DConverter(),
    DirectionCosinesConverter(),
    SphericalCartesianConverter(),
]

# The order here is important; asdf will prefer to use extensions
# that occur earlier in the list.
WCS_MANIFEST_URIS = [
    "asdf://asdf-format.org/astronomy/gwcs/manifests/gwcs-1.0.0",
]


TRANSFORM_EXTENSIONS = [
    ManifestExtension.from_uri(
        uri,
        legacy_class_names=["gwcs.extension.GWCSExtension"],
        converters=WCS_MODEL_CONVERTERS,
    )
    for uri in WCS_MANIFEST_URIS
]

def get_extensions():
    """
    Get the gwcs.converters extension.
    This method is registered with the asdf.extensions entry point.
    Returns
    -------
    list of asdf.extension.Extension
    """
    return TRANSFORM_EXTENSIONS

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import importlib.resources

from asdf.extension import Extension, ManifestExtension
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
    f"asdf://asdf-format.org/astronomy/gwcs/manifests/{path.stem}"
    for path in sorted((importlib.resources.files("asdf_wcs_schemas.resources") / "manifests").iterdir(), reverse=True)
]

# 1.0.0 contains multiple versions of the same tag, a bug fixed in
# 1.0.1 so only register 1.0.0 if it's the only available manifest
TRANSFORM_EXTENSIONS = [
    ManifestExtension.from_uri(
        uri,
        legacy_class_names=["gwcs.extension.GWCSExtension"],
        converters=WCS_MODEL_CONVERTERS,
    )
    for uri in WCS_MANIFEST_URIS
    if len(WCS_MANIFEST_URIS) == 1 or '1.0.0' not in uri
]

# if we don't register something for the 1.0.0 extension/manifest
# opening old files will issue AsdfWarning messages stating that
# the file was produced with an extension that is not installed
# As the 1.0.1 and 1.1.0 extensions support all the required tags
# it's not a helpful warning so here we register an 'empty'
# extension for 1.0.0 which doesn't support any tags or types
# but will be installed into asdf preventing the warning
if len(TRANSFORM_EXTENSIONS) > 1:
    class _EmptyExtension(Extension):
        extension_uri = 'asdf://asdf-format.org/astronomy/gwcs/extensions/gwcs-1.0.0'
        legacy_class_names=["gwcs.extension.GWCSExtension"]

    TRANSFORM_EXTENSIONS.append(_EmptyExtension())


def get_extensions():
    """
    Get the gwcs.converters extension.
    This method is registered with the asdf.extensions entry point.
    Returns
    -------
    list of asdf.extension.Extension
    """
    return TRANSFORM_EXTENSIONS

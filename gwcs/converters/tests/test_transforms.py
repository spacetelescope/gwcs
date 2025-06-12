# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy import units as u
from astropy.modeling.models import Identity, Pix2Sky_Gnomonic

try:
    from asdf_astropy.testing.helpers import assert_model_roundtrip
except ImportError:
    from asdf_astropy.converters.transform.tests.test_transform import (
        assert_model_roundtrip,
    )

from gwcs import fitswcs, geometry
from gwcs import spectroscopy as sp

sell_glass = sp.SellmeierGlass(
    B_coef=[0.58339748, 0.46085267, 3.8915394],
    C_coef=[0.00252643, 0.010078333, 1200.556],
)
sell_zemax = sp.SellmeierZemax(
    65,
    35,
    0,
    0,
    [0.58339748, 0.46085267, 3.8915394],
    [0.00252643, 0.010078333, 1200.556],
    [-2.66e-05, 0.0, 0.0],
)
snell = sp.Snell3D()
todircos = geometry.ToDirectionCosines()
fromdircos = geometry.FromDirectionCosines()
tocart = geometry.SphericalToCartesian()
tospher = geometry.CartesianToSpherical()
tan = Pix2Sky_Gnomonic()
fwcs = fitswcs.FITSImagingWCSTransform(tan)


transforms = [
    todircos,
    fromdircos,
    tospher,
    tocart,
    snell,
    sell_glass,
    sell_zemax,
    sell_zemax & todircos | snell & Identity(1) | fromdircos,
    sell_glass & todircos | snell & Identity(1) | fromdircos,
    sp.WavelengthFromGratingEquation(50000, -1),
    sp.AnglesFromGratingEquation3D(20000, 1),
    sp.WavelengthFromGratingEquation(15000 * 1 / u.m, -1),
    fwcs,
]


@pytest.mark.parametrize(("model"), transforms)
def test_transforms(tmp_path, model):
    assert_model_roundtrip(model, tmp_path, version="1.6.0")

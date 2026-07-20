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
from gwcs.converters.spectroscopy import GratingEquationConverter

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
    sp.WavelengthFromGrismEquation(
        groove_density=23000 * 1 / u.m,
        spectral_order=90,
        reference_wavelength=854.1738582455826 * u.nm,
        refractive_index=1.25 * u.one,
        refractive_index_derivative=1000 * 1 / u.m,
        out_of_plane_angle=1.5 * u.deg,
    ),
    fwcs,
]


@pytest.mark.parametrize(("model"), transforms)
def test_transforms(tmp_path, model):
    assert_model_roundtrip(model, tmp_path, version="1.6.0")


def test_wavelength_grating_equation_converter_omits_default_parameters():
    converter = GratingEquationConverter()
    model = sp.WavelengthFromGratingEquation(50000 * 1 / u.m, -1)

    node = converter.to_yaml_tree_transform(model, tag=None, ctx=None)

    assert node == {
        "output": "wavelength",
        "order": -1.0,
        "groove_density": 50000.0 / u.m,
    }


def test_wavelength_grating_equation_converter_serializes_non_default_parameters():
    converter = GratingEquationConverter()
    model = sp.WavelengthFromGrismEquation(
        groove_density=23000 * 1 / u.m,
        spectral_order=90,
        reference_wavelength=854.1738582455826 * u.nm,
        refractive_index=1.25 * u.one,
        refractive_index_derivative=1000 * 1 / u.m,
        out_of_plane_angle=1.5 * u.deg,
    )

    node = converter.to_yaml_tree_transform(model, tag=None, ctx=None)

    assert node == {
        "output": "wavelength",
        "order": 90.0,
        "groove_density": 23000.0 / u.m,
        "reference_wavelength": 854.1738582455826 * u.nm,
        "refractive_index": 1.25 * u.one,
        "refractive_index_derivative": 1000.0 / u.m,
        "out_of_plane_angle": 1.5 * u.deg,
    }


def test_wavelength_grating_equation_converter_deserializes_grism_node():
    converter = GratingEquationConverter()
    node = {
        "output": "wavelength",
        "order": 90,
        "groove_density": 23000.0 / u.m,
        "reference_wavelength": 854.1738582455826 * u.nm,
        "refractive_index": 1.25 * u.one,
        "refractive_index_derivative": 1000.0 / u.m,
        "out_of_plane_angle": 1.5 * u.deg,
    }

    model = converter.from_yaml_tree_transform(node, tag=None, ctx=None)

    assert isinstance(model, sp.WavelengthFromGrismEquation)

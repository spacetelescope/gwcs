import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Identity
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from gwcs import geometry
from gwcs import spectroscopy as sp


def test_angles_grating_equation():
    """Test agaibst the Nispec implementation."""
    lam = np.array([2e-6] * 4)
    alpha_in = np.linspace(0.01, 0.05, 4)

    model = sp.AnglesFromGratingEquation3D(20000, -1)

    # Eq. from Nirspec model.
    xout = -alpha_in - (model.groove_density * model.spectral_order * lam)

    alpha_out, beta_out, gamma_out = model(lam, -alpha_in, alpha_in)
    assert_allclose(alpha_out, xout)
    assert_allclose(beta_out, -alpha_in)
    assert_allclose(gamma_out, np.sqrt(1 - alpha_out**2 - beta_out**2))

    # Now with units
    model = sp.AnglesFromGratingEquation3D(20000 * 1 / u.m, -1)

    # Eq. from Nirspec model.
    xout = -alpha_in - (model.groove_density * model.spectral_order * lam * u.m)

    alpha_out, beta_out, gamma_out = model(
        lam * u.m, -u.Quantity(alpha_in), u.Quantity(alpha_in)
    )
    assert_allclose(alpha_out, xout)
    assert_allclose(beta_out, -alpha_in)
    assert_allclose(gamma_out, np.sqrt(1 - alpha_out**2 - beta_out**2))


def test_wavelength_grating_equation_units() -> None:
    alpha_in = np.linspace(0.01, 0.05, 4)

    # Bare numbers are coerced: groove_density → 1/m, spectral_order → dimensionless.
    # Result therefore has units of meters.
    model = sp.WavelengthFromGratingEquation(20000, -1)
    wave = -(alpha_in + alpha_in) / (20000 / u.m * -1)
    result = model(-alpha_in, -alpha_in)
    assert u.allclose(result, wave)

    # Explicit Quantity inputs.
    model = sp.WavelengthFromGratingEquation(20000 * 1 / u.m, -1)
    wave = -(u.Quantity(alpha_in) + u.Quantity(alpha_in)) / (20000 * 1 / u.m * -1)
    result = model(-u.Quantity(alpha_in), -u.Quantity(alpha_in))
    assert u.allclose(result, wave)


def test_wavelength_grating_equation_incompatible_units_raises() -> None:
    """Passing units that are incompatible with 1/length for groove_density should
    raise UnitConversionError rather than silently producing a wrong result."""
    model = sp.WavelengthFromGratingEquation(
        groove_density=20000 / u.m,
        spectral_order=-1,
        refractive_index_derivative=1000 * u.m,  # wrong: should be 1/length
    )
    alpha_in = np.sin(65.0 * u.deg)
    with pytest.raises(u.UnitConversionError):
        model(alpha_in, alpha_in)


def test_refracted_angle_sine_model_basic() -> None:
    """Output should be the sine of the refracted angle at the reference pixel."""
    groove_density = 23000 / u.m
    spectral_order = 90 * u.one
    incident_angle = 65.696 * u.deg
    refractive_index = 1.25 * u.one
    refractive_index_derivative = 1000 / u.m
    out_of_plane_angle = 1.5 * u.deg
    model = sp.RefractedAngleSineModel(
        reference_pixel=217,
        reference_wavelength=854.1738582455826 * u.nm,
        dispersion=0.0022975580183395555 * u.nm / u.pix,
        groove_density=groove_density,
        spectral_order=spectral_order,
        incident_angle=incident_angle,
        refractive_index=refractive_index,
        refractive_index_derivative=refractive_index_derivative,
        out_of_plane_angle=out_of_plane_angle,
        camera_angle=0 * u.deg,
    )

    grism_constant = (groove_density * spectral_order) / np.cos(out_of_plane_angle)
    reference_refracted_angle = np.arcsin(
        (grism_constant * (854.1738582455826 * u.nm))
        - refractive_index * np.sin(incident_angle)
    )
    result = model(217)
    assert u.allclose(result, np.sin(reference_refracted_angle), atol=1e-12)


def test_refracted_angle_sine_model_bare_number_coercion() -> None:
    """Bare-number arguments should be coerced to the assumed units."""
    model = sp.RefractedAngleSineModel(
        reference_pixel=0,
        reference_wavelength=0,  # assumed m
        dispersion=0,  # assumed m/pix
        groove_density=1,  # assumed 1/m
        spectral_order=1,  # assumed dimensionless
        incident_angle=0,  # assumed deg
        refractive_index=1,  # assumed dimensionless
        refractive_index_derivative=0,  # assumed 1/m
        out_of_plane_angle=0,  # assumed deg
        camera_angle=0,  # assumed deg
    )
    assert model.reference_wavelength.unit == u.m
    assert model.dispersion.unit == u.m / u.pix
    assert model.groove_density.unit == 1 / u.m
    assert model.spectral_order.unit == u.one
    assert model.incident_angle.unit == u.deg
    assert model.refractive_index.unit == u.one
    assert model.refractive_index_derivative.unit == 1 / u.m
    assert model.out_of_plane_angle.unit == u.deg
    assert model.camera_angle.unit == u.deg


def test_refracted_angle_sine_model_matches_manual() -> None:
    """Result should match a manually computed refracted-angle sine."""
    reference_pixel = 217.0
    reference_wavelength = 854.1738582455826 * u.nm
    dispersion = 0.0022975580183395555 * u.nm / u.pix
    groove_density = 23000.0 / u.m
    spectral_order = 90 * u.one
    incident_angle = 65.696 * u.deg
    refractive_index = 1.25 * u.one
    refractive_index_derivative = 1000.0 / u.m
    out_of_plane_angle = 1.5 * u.deg
    camera_angle = 0.8 * u.deg

    model = sp.RefractedAngleSineModel(
        reference_pixel=reference_pixel,
        reference_wavelength=reference_wavelength,
        dispersion=dispersion,
        groove_density=groove_density,
        spectral_order=spectral_order,
        incident_angle=incident_angle,
        refractive_index=refractive_index,
        refractive_index_derivative=refractive_index_derivative,
        out_of_plane_angle=out_of_plane_angle,
        camera_angle=camera_angle,
    )

    grism_constant = (groove_density * spectral_order) / np.cos(out_of_plane_angle)
    reference_refracted_angle = np.arcsin(
        (grism_constant * reference_wavelength)
        - refractive_index * np.sin(incident_angle)
    )
    grism_parameter_per_wavelength = (
        grism_constant - refractive_index_derivative * np.sin(incident_angle)
    ) / (np.cos(reference_refracted_angle) * np.cos(camera_angle) ** 2)

    pixels = np.array([0.0, 100.0, 217.0, 300.0, 511.0])
    wavelength_offset = ((pixels - reference_pixel) * u.pix) * dispersion
    expected_angle = (
        np.arctan(
            -np.tan(camera_angle) + wavelength_offset * grism_parameter_per_wavelength
        )
        + reference_refracted_angle
        + camera_angle
    )
    expected = np.sin(expected_angle)
    result = model(pixels)
    assert u.allclose(result, expected, atol=1e-12)


def test_wavelength_grating_equation_defaults():
    model = sp.WavelengthFromGratingEquation(groove_density=20000, spectral_order=-1)
    assert model.reference_wavelength.value == 0
    assert model.refractive_index.value == 1
    assert model.refractive_index_derivative.value == 0
    assert model.out_of_plane_angle.value == 0


def test_wavelength_grating_equation_grating_mode_reference_pixel():
    params = {
        "reference_pixel": 217.0,
        "reference_wavelength": 854.1738582455826 * u.nm,
        "dispersion": 0.0022975580183395555 * u.nm / u.pix,
        "grating_density": 23000.0 / u.m,
        "spectral_order": 90,
        "incident_angle": 65.696 * u.deg,
        "refractive_index": 1.25,
        "refractive_index_derivative": 1000.0 / u.m,
        "out_of_plane_angle": 1.5 * u.deg,
        "camera_angle": 0.8 * u.deg,
    }
    model = sp.WavelengthFromGratingEquation(
        groove_density=params["grating_density"],
        spectral_order=params["spectral_order"],
        reference_wavelength=params["reference_wavelength"],
        refractive_index=params["refractive_index"],
        refractive_index_derivative=params["refractive_index_derivative"],
        out_of_plane_angle=params["out_of_plane_angle"],
    )

    grism_constant = (params["grating_density"] * params["spectral_order"]) / np.cos(
        params["out_of_plane_angle"]
    )
    reference_refracted_angle = np.arcsin(
        (grism_constant * params["reference_wavelength"])
        - params["refractive_index"] * np.sin(params["incident_angle"])
    )
    incident_angle_sine = np.sin(params["incident_angle"])
    adjusted_groove_density = (
        (params["grating_density"] * params["spectral_order"])
        / np.cos(params["out_of_plane_angle"])
        - params["refractive_index_derivative"] * incident_angle_sine
    ) / params["spectral_order"]

    alpha_in = incident_angle_sine
    alpha_out = np.sin(reference_refracted_angle)
    result = model(alpha_in, alpha_out)
    expected = (
        (
            params["refractive_index"]
            - params["refractive_index_derivative"] * params["reference_wavelength"]
        )
        * incident_angle_sine
        + np.sin(reference_refracted_angle)
    ) / (adjusted_groove_density * params["spectral_order"])

    assert u.allclose(result, expected)


def test_wavelength_grating_equation_grating_mode_matches_closed_form_for_pixel_array():
    params = {
        "reference_pixel": 217.0,
        "reference_wavelength": 854.1738582455826 * u.nm,
        "dispersion": 0.0022975580183395555 * u.nm / u.pix,
        "grating_density": 23000.0 / u.m,
        "spectral_order": 90,
        "incident_angle": 65.696 * u.deg,
        "refractive_index": 1.25,
        "refractive_index_derivative": 1000.0 / u.m,
        "out_of_plane_angle": 1.5 * u.deg,
        "camera_angle": 0.8 * u.deg,
    }
    model = sp.WavelengthFromGratingEquation(
        groove_density=params["grating_density"],
        spectral_order=params["spectral_order"],
        reference_wavelength=params["reference_wavelength"],
        refractive_index=params["refractive_index"],
        refractive_index_derivative=params["refractive_index_derivative"],
        out_of_plane_angle=params["out_of_plane_angle"],
    )

    grism_constant = (params["grating_density"] * params["spectral_order"]) / np.cos(
        params["out_of_plane_angle"]
    )
    reference_refracted_angle = np.arcsin(
        (grism_constant * params["reference_wavelength"])
        - params["refractive_index"] * np.sin(params["incident_angle"])
    )
    grism_parameter_per_wavelength = (
        grism_constant
        - params["refractive_index_derivative"] * np.sin(params["incident_angle"])
    ) / (np.cos(reference_refracted_angle) * np.cos(params["camera_angle"]) ** 2)
    pixels = np.array([0.0, 100.0, 217.0, 300.0, 511.0])
    wavelength_offset = ((pixels - params["reference_pixel"]) * u.pix) * params[
        "dispersion"
    ]
    refracted_angle_sine = np.sin(
        np.arctan(
            -np.tan(params["camera_angle"])
            + wavelength_offset * grism_parameter_per_wavelength
        )
        + reference_refracted_angle
        + params["camera_angle"]
    )
    incident_angle_sine = np.sin(params["incident_angle"])
    adjusted_groove_density = (
        (params["grating_density"] * params["spectral_order"])
        / np.cos(params["out_of_plane_angle"])
        - params["refractive_index_derivative"] * incident_angle_sine
    ) / params["spectral_order"]
    expected = (
        (
            params["refractive_index"]
            - params["refractive_index_derivative"] * params["reference_wavelength"]
        )
        * incident_angle_sine
        + refracted_angle_sine
    ) / (adjusted_groove_density * params["spectral_order"])

    alpha_in = incident_angle_sine
    alpha_out = refracted_angle_sine
    result = model(alpha_in, alpha_out)

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_wavelength_grating_equation_grating_mode_matches_astropy():
    header = {
        "CTYPE1": "AWAV-GRA",
        "CUNIT1": "nm",
        "CRPIX1": 218,
        "CRVAL1": 854.1738582455826,
        "CDELT1": 0.0022975580183395555,
        "PV1_0": 23000.0,
        "PV1_1": 90,
        "PV1_2": 65.696,
        "PV1_3": 1.25,
        "PV1_4": 1000.0,
        "PV1_5": 1.5,
        "PV1_6": 0.8,
    }
    model = sp.WavelengthFromGratingEquation(
        groove_density=header["PV1_0"] / u.m,
        spectral_order=header["PV1_1"],
        reference_wavelength=header["CRVAL1"] * u.nm,
        refractive_index=header["PV1_3"] * u.one,
        refractive_index_derivative=header["PV1_4"] / u.m,
        out_of_plane_angle=header["PV1_5"] * u.deg,
    )

    pixels = np.array([0, 100, 217, 300, 511], dtype=float)
    reference_pixel = header["CRPIX1"] - 1
    reference_wavelength = header["CRVAL1"] * u.nm
    dispersion = header["CDELT1"] * u.nm / u.pix
    refractive_index = header["PV1_3"]
    camera_angle = header["PV1_6"] * u.deg

    incident_angle_sine = np.sin(header["PV1_2"] * u.deg)
    grism_constant = ((header["PV1_0"] / u.m) * (header["PV1_1"])) / np.cos(
        header["PV1_5"] * u.deg
    )
    reference_refracted_angle = np.arcsin(
        (grism_constant * reference_wavelength) - refractive_index * incident_angle_sine
    )
    grism_parameter_per_wavelength = (
        grism_constant - (header["PV1_4"] / u.m) * incident_angle_sine
    ) / (np.cos(reference_refracted_angle) * np.cos(camera_angle) ** 2)
    wavelength_offset = ((pixels - reference_pixel) * u.pix) * dispersion
    alpha_out = np.sin(
        np.arctan(
            -np.tan(camera_angle) + wavelength_offset * grism_parameter_per_wavelength
        )
        + reference_refracted_angle
        + camera_angle
    )
    alpha_in = incident_angle_sine

    expected = WCS(header).spectral.pixel_to_world(pixels)
    result = model(alpha_in, alpha_out)

    assert_allclose(
        result.to_value(u.nm), expected.to_value(u.nm), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize(
    ("wavelength", "n"), [(1, 1.43079543), (2, 1.42575377), (5, 1.40061966)]
)
def test_SellmeierGlass(wavelength, n, sellmeier_glass):
    """Test from Nirspec team.

    Wavelength is in microns.
    """
    n_result = sellmeier_glass(wavelength)
    assert_allclose(n_result, n)


def test_SellmeierZemax(sellmeier_zemax):
    """The data for this test come from Nirspec."""
    n = 1.4254647475849418
    assert_allclose(sellmeier_zemax(2), n)


def test_SellmeierZemax_array(sellmeier_zemax):
    """Covers a bug where multiple inputs would result in identical outputs"""
    wl = np.linspace(0.5, 8.0, 50)
    n = sellmeier_zemax(wl)
    expected_n0 = 1.43802003
    expected_nf = 1.35076012
    assert np.isclose(n[0], expected_n0)
    assert np.isclose(n[-1], expected_nf)
    assert np.unique(n).size == wl.size


def test_Snell3D(sellmeier_glass):
    """Test from Nirspec."""
    expected = (0.07015255913513296, 0.07015255913513296, 0.9950664484814988)
    model = sp.Snell3D()
    n = 1.4254647475849418
    assert_allclose(model(n, 0.1, 0.1, 0.9), expected)


def test_snell_sellmeier_combined(sellmeier_glass):
    fromdircos = geometry.FromDirectionCosines()
    todircos = geometry.ToDirectionCosines()
    model = sellmeier_glass & todircos | sp.Snell3D() & Identity(1) | fromdircos

    expected = (0.07013833805527926, 0.07013833805527926, 1.0050677723764139)
    assert_allclose(model(2, 0.1, 0.1, 0.9), expected)

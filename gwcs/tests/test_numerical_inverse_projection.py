# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regression tests for numerical inverse projection handling."""

from astropy import coordinates as coord
from astropy import units as u
from astropy.modeling import models

from gwcs import coordinate_frames as cf
from gwcs import wcs


def test_calc_approx_inv_handles_projection_model_instance():
    """A Pix2Sky model in the forward transform should build an approximate inverse."""
    crpix = (2044, 2044)
    shift_by_crpix = models.Shift(-crpix[0]) & models.Shift(-crpix[1])

    distortion_x = models.Polynomial2D(
        4,
        c0_0=0.0,
        c1_0=0.00002993,
        c2_0=-0.0,
        c3_0=-0.0,
        c4_0=-0.0,
        c0_1=-0.00000451,
        c0_2=0.0,
        c0_3=-0.0,
        c0_4=-0.0,
        c1_1=-0.0,
        c1_2=-0.0,
        c1_3=-0.0,
        c2_1=-0.0,
        c2_2=-0.0,
        c3_1=-0.0,
    )
    distortion_y = models.Polynomial2D(
        4,
        c0_0=0.0,
        c1_0=0.00000397,
        c2_0=-0.0,
        c3_0=0.0,
        c4_0=0.0,
        c0_1=0.00002904,
        c0_2=-0.0,
        c0_3=-0.0,
        c0_4=-0.0,
        c1_1=-0.0,
        c1_2=0.0,
        c1_3=0.0,
        c2_1=0.0,
        c2_2=0.0,
        c3_1=0.0,
    )
    distortion = models.Mapping([0, 1, 0, 1]) | (distortion_x & distortion_y)
    transform = (
        shift_by_crpix
        | distortion
        | models.Pix2Sky_TAN()
        | models.RotateNative2Celestial(5.63056810618, -72.05457184279, 180)
    )

    detector_frame = cf.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky_frame = cf.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    gwcs = wcs.WCS([(detector_frame, transform), (sky_frame, None)])
    gwcs.bounding_box = ((0, crpix[0] * 2), (0, crpix[1] * 2))

    gwcs._calc_approx_inv()

    assert gwcs._approx_inverse is not None

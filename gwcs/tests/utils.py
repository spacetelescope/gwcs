import numpy as np

from astropy.modeling.models import (
    Shift, Polynomial2D, Pix2Sky_TAN, RotateNative2Celestial, Mapping
)

from astropy import coordinates as coord
from astropy import units
from astropy import wcs as fits_wcs

from .. wcs import WCS
from .. import coordinate_frames as cf


def _gwcs_from_hst_fits_wcs(header, hdu=None):
    # NOTE: this function ignores table distortions
    def coeffs_to_poly(mat, degree):
        pol = Polynomial2D(degree=degree)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if 0 < i + j <= degree:
                    setattr(pol, f'c{i}_{j}', mat[i, j])
        return pol

    w = fits_wcs.WCS(header, hdu)
    ny, nx = w.pixel_shape
    x0, y0 = w.wcs.crpix - 1

    cd = w.wcs.piximg_matrix

    cfx, cfy = np.dot(cd, [w.sip.a.ravel(), w.sip.b.ravel()])
    a = np.reshape(cfx, w.sip.a.shape)
    b = np.reshape(cfy, w.sip.b.shape)
    a[1, 0] = cd[0, 0]
    a[0, 1] = cd[0, 1]
    b[1, 0] = cd[1, 0]
    b[0, 1] = cd[1, 1]

    polx = coeffs_to_poly(a, w.sip.a_order)
    poly = coeffs_to_poly(b, w.sip.b_order)

    # construct GWCS:
    det2sky = (
        (Shift(-x0) & Shift(-y0)) | Mapping((0, 1, 0, 1)) | (polx & poly) |
        Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, 180)
    )

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(units.pix, units.pix))
    sky_frame = cf.CelestialFrame(
        reference_frame=getattr(coord, w.wcs.radesys).__call__(),
        name=w.wcs.radesys,
        unit=(units.deg, units.deg)
    )
    pipeline = [(detector_frame, det2sky), (sky_frame, None)]
    gw = WCS(pipeline)
    gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

    return gw

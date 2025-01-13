.. _fits_equivalent_example:

FITS Equivalent WCS Example
===========================

The following example shows how to construct a GWCS object equivalent to
a FITS imaging WCS without distortion, defined in this FITS imaging header::

  WCSAXES =                    2 / Number of coordinate axes
  WCSNAME = '47 Tuc     '        / Coordinate system title
  CRPIX1  =               2048.0 / Pixel coordinate of reference point
  CRPIX2  =               1024.0 / Pixel coordinate of reference point
  PC1_1   =   1.290551569736E-05 / Coordinate transformation matrix element
  PC1_2   =  5.9525007864732E-06 / Coordinate transformation matrix element
  PC2_1   =  5.0226382102765E-06 / Coordinate transformation matrix element
  PC2_2   = -1.2644844123757E-05 / Coordinate transformation matrix element
  CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
  CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
  CUNIT1  = 'deg'                / Units of coordinate increment and value
  CUNIT2  = 'deg'                / Units of coordinate increment and value
  CTYPE1  = 'RA---TAN'           / TAN (gnomonic) projection + SIP distortions
  CTYPE2  = 'DEC--TAN'           / TAN (gnomonic) projection + SIP distortions
  CRVAL1  =        5.63056810618 / [deg] Coordinate value at reference point
  CRVAL2  =      -72.05457184279 / [deg] Coordinate value at reference point
  LONPOLE =                180.0 / [deg] Native longitude of celestial pole
  LATPOLE =      -72.05457184279 / [deg] Native latitude of celestial pole
  RADESYS = 'ICRS'                / Equatorial coordinate system


For this example the following imports are needed:

  >>> import numpy as np
  >>> from astropy.modeling import models
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

The ``forward_transform`` is constructed as a combined model using `astropy.modeling`.
The ``frames`` are subclasses of `~gwcs.coordinate_frames.CoordinateFrame`. Although strings are
acceptable as ``coordinate_frames`` it is recommended this is used only in testing/debugging.

Using the `~astropy.modeling` package create a combined model to transform
detector coordinates to ICRS following the FITS WCS standard convention.

First, create a transform which shifts the input  ``x`` and ``y`` coordinates by ``CRPIX``.  We subtract 1 from the CRPIX values because the first pixel is considered pixel ``1`` in FITS WCS:

  >>> shift_by_crpix = models.Shift(-(2048 - 1)*u.pix) & models.Shift(-(1024 - 1)*u.pix)

Create a transform which rotates the inputs using the ``PC matrix``.

  >>> matrix = np.array([[1.290551569736E-05, 5.9525007864732E-06],
  ...                    [5.0226382102765E-06 , -1.2644844123757E-05]])
  >>> rotation = models.AffineTransformation2D(matrix * u.deg,
  ...                                          translation=[0, 0] * u.deg)
  >>> rotation.input_units_equivalencies = {"x": u.pixel_scale(1*u.deg/u.pix),
  ...                                       "y": u.pixel_scale(1*u.deg/u.pix)}
  >>> rotation.inverse = models.AffineTransformation2D(np.linalg.inv(matrix) * u.pix,
  ...                                                  translation=[0, 0] * u.pix)
  >>> rotation.inverse.input_units_equivalencies = {"x": u.pixel_scale(1*u.pix/u.deg),
  ...                                               "y": u.pixel_scale(1*u.pix/u.deg)}

Create a tangent projection and a rotation on the sky using ``CRVAL``.

  >>> tan = models.Pix2Sky_TAN()
  >>> celestial_rotation =  models.RotateNative2Celestial(5.63056810618*u.deg, -72.05457184279*u.deg, 180*u.deg)

  >>> det2sky = shift_by_crpix | rotation | tan | celestial_rotation
  >>> det2sky.name = "linear_transform"

Create a ``detector`` coordinate frame and a ``celestial`` ICRS frame.

  >>> detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
  ...                             unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
  ...                               unit=(u.deg, u.deg))

This WCS pipeline has only one step - from ``detector`` to ``sky``:

  >>> pipeline = [(detector_frame, det2sky),
  ...             (sky_frame, None)
  ...            ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
    From      Transform
  -------- ----------------
  detector linear_transform
      icrs             None

Now we have a complete WCS object. The next example will use it to convert pixel
coordinates(1, 2) to sky coordinates:

  >>> sky = wcsobj(1*u.pix, 2*u.pix, with_units=True)
  >>> print(sky)
  <SkyCoord (ICRS): (ra, dec) in deg
    (5.52515954, -72.05190935)>

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform` to provide a mapping from sky coordinates to pixel coordinates 
if available, otherwise it applies an iterative method to calculate the pixel coordinates.

  >>> wcsobj.invert(sky)
  (<Quantity 1. pix>, <Quantity 2. pix>)

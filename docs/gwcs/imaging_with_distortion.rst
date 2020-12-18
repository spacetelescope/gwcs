.. _imaging_example:

Adding distortion to the imaging example
========================================

Let's expand the WCS created in :ref:`getting-started` by adding a polynomial
distortion correction.

Because the polynomial models in `~astropy.modeling` do not support units yet,
this example will use transforms without units. At the end the units
associated with the output frame are used to create a `~astropy.coordinates.SkyCoord` object.

The imaging example without units:

  >>> import numpy as np
  >>> from astropy.modeling import models
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

  >>> crpix = (2048, 1024)
  >>> shift_by_crpix = models.Shift(-crpix[0]) & models.Shift(-crpix[1])
  >>> matrix = np.array([[1.290551569736E-05, 5.9525007864732E-06],
  ...                    [5.0226382102765E-06 , -1.2644844123757E-05]])
  >>> rotation = models.AffineTransformation2D(matrix)
  >>> rotation.inverse = models.AffineTransformation2D(np.linalg.inv(matrix))
  >>> tan = models.Pix2Sky_TAN()
  >>> celestial_rotation =  models.RotateNative2Celestial(5.63056810618, -72.05457184279, 180)
  >>> det2sky = shift_by_crpix | rotation | tan | celestial_rotation
  >>> det2sky.name = "linear_transform"
  >>> detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
  ...                             unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
  ...                               unit=(u.deg, u.deg))
  >>> pipeline = [(detector_frame, det2sky),
  ...             (sky_frame, None)
  ...            ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
    From      Transform
  -------- ----------------
  detector linear_transform
      icrs             None

First create distortion corrections represented by a polynomial
model of fourth degree. The example uses the astropy `~astropy.modeling.polynomial.Polynomial2D`
and `~astropy.modeling.mappings.Mapping` models.

  >>> poly_x = models.Polynomial2D(4)
  >>> poly_x.parameters = [0, 1, 8.55e-06, -4.73e-10, 2.37e-14, 0, -5.20e-06,
  ...                      -3.98e-11, 1.97e-15, 2.17e-06, -5.23e-10, 3.47e-14,
  ...                      1.08e-11, -2.46e-14, 1.49e-14]
  >>> poly_y = models.Polynomial2D(4)
  >>> poly_y.parameters = [0, 0, -1.75e-06, 8.57e-11, -1.77e-14, 1, 6.18e-06,
  ...                      -5.09e-10, -3.78e-15, -7.22e-06, -6.17e-11,
  ...                      -3.66e-14, -4.18e-10, 1.22e-14, -9.96e-15]
  >>> distortion = ((models.Shift(-crpix[0]) & models.Shift(-crpix[1])) |
  ...               models.Mapping((0, 1, 0, 1)) | (poly_x & poly_y) |
  ...               (models.Shift(crpix[0]) & models.Shift(crpix[1])))
  >>> distortion.name = "distortion"

Create an intermediate frame for distortion free coordinates.

  >>> undistorted_frame = cf.Frame2D(name="undistorted_frame", unit=(u.pix, u.pix),
  ...                                axes_names=("undist_x", "undist_y"))

Using the example in :ref:`getting-started`, add the distortion correction to
the WCS pipeline and initialize the WCS.

  >>> pipeline = [(detector_frame, distortion),
  ...             (undistorted_frame, det2sky),
  ...             (sky_frame, None)
  ...             ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
         From          Transform
  ----------------- ----------------
           detector       distortion
  undistorted_frame linear_transform
               icrs             None

Finally, save this WCS to an ``ASDF`` file:

.. doctest-skip::

  >>> from asdf import AsdfFile
  >>> tree = {"wcs": wcsobj}
  >>> wcs_file = AsdfFile(tree)
  >>> wcs_file.write_to("imaging_wcs_wdist.asdf")

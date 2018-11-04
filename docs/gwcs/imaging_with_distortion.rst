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

  >>> shift_by_crpix = models.Shift(-2048) & models.Shift(-1024)
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
  >>> poly_x.parameters = np.arange(15) * .1
  >>> poly_y = models.Polynomial2D(4)
  >>> poly_y.parameters = np.arange(15) * .2
  >>> distortion = models.Mapping((0, 1, 0, 1)) | poly_x & poly_y
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
  

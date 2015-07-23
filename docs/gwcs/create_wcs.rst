An imaging example
==================


This is a step by step example of creating a WCS for an imaging observation.
The forward transformation is from detector pixels to sky coordinates in ICRS.
There's an intermediate coordinate frame, called 'focal', after applying the distortion.

The following packages must be imported.


  >>> import numpy as np
  >>> from astropy.modeling.models import (Shift, Scale, Rotation2D,
      Pix2Sky_TAN, RotateNative2Celestial, Mapping, Polynomial2D, AffineTransformation2D)
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf


Create the transform
~~~~~~~~~~~~~~~~~~~~

First create a model for the distortion transform. Let's assume the distortion
in each direction is represented by a 2nd degree Polynomial model of `x` and `y`.

  >>> dist_x = Polynomial2D(2, c0_0=0.0013, c1_0=0.5, c2_0=0, c0_1=0.823, c0_2=1.4, c1_1=1.7, name='x_distortion')
  >>> dist_y = Polynomial2D(2, c0_0=0.03, c1_0=0.25, c2_0=1.2, c0_1=0.3, c0_2=0.4, c1_1=0.7, name='y_distortion')
  >>> distortion = dist_x & dist_y

The last line above joins the two distortion models in one model which now takes
4 inputs (x, y, x, y), two for each model. In order for this to work a
:class:`~astropy.modeling.mappings.Mapping` model must be prepended to the ``distortion`` transform.

  >>> distortion_mapping = Mapping((0, 1, 0, 1), name='distortion_mapping')
  >>> distortion = distortion_mapping | dist_x & dist_y

Next we create the transform from focal plane to sky. For this example, suppose the WCS is in a FITS
header represented through the usual FITS WCS keywords - a point on the detector (CRPIX1/2) corresponds
to a point on a projection plane tangent to the celestial sphere (CRVAL1/2). The FITS keywords are: ..

::

  CRPIX1 = 2048.0
  CRPIX2 = 1024.0
  CRVAL1 = 5.63056810618
  CRVAL2 = -72.0545718428
  LONPOLE = 180
  PC1_1 =  1.29058668e-05
  PC1_2 = 5.95320246e-06
  PC2_1 = 5.02215196e-06
  PC2_2 = -1.26450104e-05
  CTYPE1 = 'RA---TAN'
  CTYPE2 = 'DEC---TAN'

.. note:: FITS WCS keywords are given simply as an example and because it's the most often
  used way to represent this information. However, this code is not limited to FITS WCS.

The WCS information above represents a serial compound model consisting of a shift in the focal plane
by CRPIX, a rotation by the PC matrix, a tangent deprojection and a sky rotation. Using
`astropy.modeling <http://docs.astropy.org/en/stable/modeling>`__ this can be written as

  >>> shift = Shift(CRPIX1, name="x_shift") & Shift(CRPIX2, name="y_shift")
  >>> plane_rotation = AffineTransformation2D(matrix=np.array([[PC1_1, PC1_2], [PC2_1, PC2_2]]))
  >>> tangent = Pix2Sky_TAN()
  >>> sky_rotation = RotateNative2Celestial(CRVAL1, CRVAL2, LONPOLE)

Chaining these models into one compound models creates the total transformation from focal plane to sky.

  >>> focal2sky = shift | plane_rotation | tangent | sky_rotation


Create the coordinate frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  >>> outframe = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
  >>> outframe.unit
  [Unit("deg"), Unit("deg")]
  >>> inframe = cf.Frame2D(name='detector')
  >>> print(inframe.unit)
  [Unit("pix"), Unit("pix")]
  >>> focal = cf.Frame2D(name='focal')


Create the WCS object
~~~~~~~~~~~~~~~~~~~~~

In this case it is convenient to initialize the WCS object with a list of tuples,
where each tuple (step_frame, step_transform) represents a transform "step_transform"
from frame "step_frame" to the next frame in the WCS pipeline.
The transform in the last step is always None to indicate end of the pipeline.

  >>> pipeline = [(inframe, distortion), (focal, focal2sky), (outframe, None)]
  >>> w = wcs.WCS(pipeline)
  >>> w(1, 2)
      (5.736718396223817, -72.057214400243)

Frame objects allow to extend the functionality by using `astropy.coordinates` and `astropy.units`.
Frames are available as attributes of the WCS object.

  >>> w.available_frames
      ['detector', 'focal', 'icrs']
  >>> w.icrs
      <CelestialFrame(reference_frame=<ICRS Frame>, unit=[Unit("deg"), Unit("deg")], name=icrs)>
  >>> w.icrs.coordinates(1, 2)
      <SkyCoord (ICRS): (ra, dec) in deg
          (5.7367184, -72.0572144)>
  >>> w.icrs.transform_to('galactic', 1, 2)
      <SkyCoord (Galactic): (l, b) in deg
          (306.02322236, -44.89963512)>




Creating a WCS
==============

An imaging example
------------------

This is a step by step example of creating a WCS for an imagng observation.
The forward transformation is from detector pixels to sky coordinates in ICRS.

The following packages must be imported.


  >>> import numpy as np
  >>> from astropy.modeling import models
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> import gwcs


Create the transform
~~~~~~~~~~~~~~~~~~~~

First create a model for the distortion transform. Let's assume the distortion
in each direction is represented by a 2nd degree Polynomial model of `x` and `y`.

  >>> dist_x = models.Polynomial2D(2, c0_0=0.0013, c1_0=0.5, c2_0=0, c0_1=0.823, c0_2=1.4, c1_1=1.7, name='x_distortion')
  >>> dist_y = models.Polynomial2D(2, c0_0=0.03, c1_0=0.25, c2_0=1.2, c0_1=0.3, c0_2=0.4, c1_1=0.7, name='y_distortion')
  >>> distortion = dist_x & dist_y

The last line above joins the two distortion models in a one model which now takes
4 inputs (x, y, x, y), two for each model. In order for this to work a
:class:`~astropy.modeling.mappings.Mapping` model must be prepended to the ``distortion`` transform.

  >>> distortion_mapping = models.Mapping((0, 1, 0, 1), name='distortion_mapping')
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
  used way to represent this information. However, this code is not limited to FITS WCS. In fact the
  distortion cannot be represented using FITS WCS. An alternative way is shown in `WCS serialization`.

The WCS information above represents a serial compound model consisting of a shift in the focal plane
by CRPIX, a rotation by the PC matrix, a tangent deprojection and a sky rotation. Using
`astropy.modeling <http://docs.astropy.org/en/stable/modeling>`__ this can be written as

  >>> shift = models.Shift(CRPIX1, name="x_shift") & models.Shift(CRPIX2, name="y_shift")
  >>> plane_rotation = models.AffineTransformation2D(matrix=np.array([[PC1_1, PC1_2], [PC2_1], PC2_2]])
  >>> tangent = models.Pix2Sky_TAN()
  >>> sky_roations = models.RotateNative2Celestial(CRVAL1, CRVAL2, LONPOLE)

Chaining these models into one compound models creates the total transformation from focal plane to sky

  >>> focal2sky = shift | plane_rotation | tangent | sky_rotation


Create the coordinate systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  >>> icrs = coordinates.ICRS()
  >>> final_system = gwcs.CelestialFrame(reference_frame=icrs)
  >>> final_system.unit
  <CelestialFrame(reference_frame=<ICRS Frame>, axes_order=(0, 1), reference_position=Barycenter,
  unit=[Unit("deg"), Unit("deg")], name="ICRS")>
  >>> start_system = gwcs.DetectorFrame()
  >>> print start_system.unit
  [Unit("pix"), Unit("pix")]
  >>> focal_plane = gwcs.FocalPlaneFrame()


Create the WCS object
~~~~~~~~~~~~~~~~~~~~~

The easiest way to create the WCS object is to pass the total transform and the output coordinate sytem as paraneters.
The default value for th input coordinate system is `~gwcs.cooridnate_frames.DetectorFrame`.

  >>> total_transform = distortion | focal2sky
  >>> image_wcs = gwcs.WCS(forward_transform=total_transform, output_coordinate_system=final_system)

A slightly more detailed approach gives some more control over the transformations:

  >>> image_wcs = gwcs.WCS(output_coordinate_system=final_system)
  >>> image_wcs.add_transform(w.input_coordinate_system, focal_plane, distortion)
  >>> image_wcs.add_transform(focal_plane, w.output_coordinate_system, focal2sky)


A spectral example
-------------------






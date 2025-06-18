.. _constructing_gwcs_models:

Introduction to Constructing Generalized World Coordinate System Models
=======================================================================

Pixel Conventions and Definitions
---------------------------------

See :ref:`pixel-conventions-and-definitions` for an overview.


.. _getting-started:

Basic Structure of a GWCS Object
--------------------------------

The key concept to be aware of is that a GWCS Object consists of a sequence
of steps; each step contains a transform (i.e., an Astropy model) that
converts the input coordinates of the step to the output coordinates of
the step. Furthermore, each step has a required coordinate frame associated
with the step. The coordinate frame represents the input coordinate frame, not
the output coordinates. Most typically, the first step coordinate frame is
the detector pixel coordinates (the default). Since no step has a coordinate
frame for the output coordinates, it is necessary to append a step with no
transform to the end of the pipeline to represent the output coordinate frame.
For imaging, this frame typically references one of the Astropy standard
Sky Coordinate Frames of Reference. The GWCS frames also serves as an
information container, holding the units on the axes, the names of the axes,
and the physical type of the axis (e.g., wavelength), as well as keeping
track of the axis order.

Since it is often useful to obtain coordinates in an intermediate frame of
reference, GWCS allows to consist of multiple steps each with its own transformo,
which itself may be a compound transform consisting of multiple elemental
transforms.
For example, for spectrographs, it is useful to have access to coordinates
in the slit plane, and in such a case, the first step would transform from
the detector to the slit plane, and the second step from the slit plane to
sky coordinates and a wavelength. Constructed this way, it is possible to
extract from the GWCS the needed transforms between identified frames of
reference.

The GWCS object can be saved to the ASDF format using the
`asdf <https://asdf.readthedocs.io/en/latest/>`__ package and validated
using `ASDF Standard <https://asdf-standard.readthedocs.io/en/latest/>`__

The way to save the GWCS object to a file:
`Save a WCS object as a pure ASDF file`_


A step-by-step example of constructing an imaging GWCS object.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to construct a GWCS object that maps
input pixel coordinates to sky coordinates. This example
involves 4 sequential transformations:

- Adjusting pixel coordinates such that the center of the array has
  (0, 0) value (typical of most WCS definitions, but any pixel may
  be the reference that is tied to the sky reference, even the (0, 0)
  pixel, or even pixels outside of the detector).
- Scaling pixels such that the center pixel of the array has the expected
  angular scale. (I.e., applying the plate scale)
- Projecting the resultant coordinates onto the sky using the tangent
  projection. If the field of view is small, the inaccuracies resulting
  leaving this out will be small; however, this is generally applied.
- Transforming the center pixel to the appropriate celestial coordinate
  with the appropriate orientation on the sky. For simplicity's sake,
  we assume the detector array is already oriented with north up, and
  that the array has the appropriate parity as the sky coordinates.


The detector has a 1000 pixel by 1000 pixel array.

For simplicity, no units will be used, but instead will be implicit.

The following imports are generally useful:

.. doctest-skip::

  >>> import numpy as np
  >>> from astropy.modeling import models
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

In the following transformation definitions, angular units are in degrees by
default.

.. doctest-skip::

  >>> pixelshift = models.Shift(-500) & models.Shift(-500)
  >>> pixelscale = models.Scale(0.1 / 3600.) & models.Scale(0.1 / 3600.) # 0.1 arcsec/pixel
  >>> tangent_projection = models.Pix2Sky_TAN()
  >>> celestial_rotation = models.RotateNative2Celestial(30., 45., 180.)

For the last transformation, the three arguments are, respectively:

- Celestial longitude (i.e., RA) of the fiducial point (e.g., (0, 0) in the input
  spherical coordinates).
  In this case we put the detector center at 30 degrees (RA = 2 hours)
- Celestial latitude (i.e., Dec) of the fiducial point. Here Dec = 45 degrees.
- Longitude of celestial pole in input coordinate system. With north up, and
  tangent projection, this always corresponds to a value of 180.

The more general case where the detector is not aligned with north, would have
a rotation transform after the pixelshift and pixelscale transformations to
align the detector coordinates with north up.

The net transformation from pixel coordinates to celestial coordinates then
becomes:

.. doctest-skip::

  >>> det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation

The remaining elements to defining the WCS are he input and output
frames of reference. While the GWCS scheme allows intermediate frames
of reference, this example doesn't have any. The output frame is
expressed with no associated transform

.. doctest-skip::

  >>> detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
  ...                             unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
  ...                               unit=(u.deg, u.deg))
  >>> wcsobj = wcs.WCS([(detector_frame, det2sky),
  ...                   (sky_frame, None)
  ...                  ])
  >>> print(wcsobj)
    From      Transform
  -------- ----------------
  detector detector_to_sky
      icrs             None

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

.. doctest-skip::

  >>> sky = wcsobj(1, 2)
  >>> print(sky)
  (29.980402161089177, 44.98616499109102)

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
if available, otherwise applies an iterative method to calculate the reverse coordinates.

GWCS supports the :ref:`wcsapi` which defines several methods to work with high level Astropy objects:

.. doctest-skip::

  >>> sky_obj = wcsobj.pixel_to_world(1, 2)
  >>> print(sky)
  <SkyCoord (ICRS): (ra, dec) in deg
    (29.98040216, 44.98616499)>
  >>> wcsobj.world_to_pixel(sky_obj)
  (0.9999999996185807, 1.999999999186798)

.. _save_as_asdf:

Save a WCS object as a pure ASDF file
+++++++++++++++++++++++++++++++++++++

.. doctest-skip::

  >>> from asdf import AsdfFile
  >>> tree = {"wcs": wcsobj}
  >>> wcs_file = AsdfFile(tree)
  >>> wcs_file.write_to("imaging_wcs.asdf")


:ref:`pure_asdf`


Reading a WCS object from a file
++++++++++++++++++++++++++++++++


`ASDF <https://asdf.readthedocs.io/>`__ is used to read a WCS object
from a pure ASDF file or from an ASDF extension in a FITS file.


.. doctest-skip::

  >>> import asdf
  >>> asdf_file = asdf.open("imaging_wcs.asdf")
  >>> wcsobj = asdf_file.tree['wcs']

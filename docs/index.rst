.. _gwcs:

GWCS Documentation
==================

`GWCS <https://github.com/spacetelescope/gwcs>`__ is a package for managing
the World Coordinate System (WCS) of astronomical data.

Introduction & Motivation for GWCS
----------------------------------

The mapping from ‘pixel’ coordinates to corresponding ‘real-world’ coordinates (e.g. celestial coordinates,
spectral wavelength) is crucial to relating astronomical data to the phenomena they describe. Images and
other types of data often come encoded with information that describes this mapping – this is referred
to as the ‘World Coordinate System’ or WCS. The term WCS is often used to refer specifically
to the most widely used 'FITS implementation of WCS', but here unless specified WCS refers to
the broader concept of relating pixel ⟷ world. (See the discussion in `APE14 <https://github.com/astropy/astropy-APEs/blob/master/APE14.rst#backgroundterminology>`__
for more on this topic).

The FITS WCS standard, currently the most widely used method of encoding WCS in data, describes a
set of required FITS header keywords and allowed values that describe how pixel ⟷ world transformations
should be done. This current paradigm of encoding data with only instructions on how to relate pixel to world, separate
from the transformation machinery itself, has several limitations:

* Limited flexibility. WCS keywords and their values are rigidly defined so that the instructions are unambiguous.
  This places limitations on, for example, describing geometric distortion in images since only a handful of distortion models are defined
  in the FITS standard (and therefore can be encoded in FITS headers as WCS information).
* Separation of data from transformation pipelines. The machinery that transforms pixel ⟷ world
  does not exist along side the data – there is merely a roadmap for how one *would* do the transformation.
  External packages and libraries (e.g wcslib, or its Python interface astropy.wcs) must be
  written to interpret the instructions and execute the transformation. These libraries
  don’t allow easy access to coordinate frames along the course of the full pixel to world
  transformation pipeline. Additionally, since these libraries can only interpret FITS WCS
  information, any custom ‘WCS’ definitions outside of FITS require the user to write their own transformation pipelines.
* Incompatibility with varying file formats. New file formats that are becoming more widely
  used in place of FITS to store astronomical data, like the ASDF format, also require a
  method of encoding WCS information. FITS WCS and the accompanying libraries are adapted for
  FITS only. A more flexible interface would be agnostic to file type, as long as the necessary
  information is present.

The `GWCS <https://github.com/spacetelescope/gwcs>`__ package and GWCS object is a generalized WCS
implementation that mitigates these limitations. The goal of the GWCS package is to provide a
flexible toolkit for expressing and evaluating transformations between pixel and world coordinates,
as well as intermediate frames along the course of this transformation.The GWCS object supports a
data model which includes the entire transformation pipeline from input pixel coordinates to
world coordinates (and vice versa). The basis of the GWCS object is astropy `modeling <https://docs.astropy.org/en/stable/modeling/>`__.
Models that describe the pixel ⟷ world transformations can be chained, joined or combined with arithmetic operators
using the flexible framework of compound models in modeling. This approach allows for easy
access to intermediate frames. In the case of a celestial output frame `coordinates <http://docs.astropy.org/en/stable/coordinates/>`__ provides further transformations between
standard celestial coordinate frames. Spectral output coordinates are instances of `~astropy.units.Quantity`
and can be transformed to other units with the tools in that package. `~astropy.time.Time` coordinates are instances of `~astropy.time.Time`.
GWCS supports transforms initialized with `~astropy.units.Quantity`
objects ensuring automatic unit conversion.

Pixel Conventions and Definitions
---------------------------------

This API assumes that integer pixel values fall at the center of pixels (as
assumed in the FITS-WCS standard, see Section 2.1.4 of `Greisen et al., 2002,
A&A 446, 747 <https://doi.org/10.1051/0004-6361:20053818>`_), while at the same
time matching the Python 0-index philosophy.  That is, the first pixel is
considered pixel ``0``, but pixel coordinates ``(0, 0)`` are the *center* of
that pixel.  Hence the first pixel spans pixel values ``-0.5`` to ``0.5``.

There are two main conventions for ordering pixel coordinates. In the context of
2-dimensional imaging data/arrays, one can either think of the pixel coordinates
as traditional Cartesian coordinates (which we call ``x`` and ``y`` here), which
are usually given with the horizontal coordinate (``x``) first, and the vertical
coordinate (``y``) second, meaning that pixel coordinates would be given as
``(x, y)``. Alternatively, one can give the coordinates by first giving the row
in the data, then the column, i.e. ``(row, column)``. While the former is a more
common convention when e.g. plotting (think for example of the Matplotlib
``scatter(x, y)`` method), the latter is the convention used when accessing
values from e.g. Numpy arrays that represent images (``image[row, column]``).

The GWCS object assumes Cartesian order ``(x, y)``, however the :ref:`ape14` accepts both conventions.
The order of the pixel coordinates (``(x, y)`` vs ``(row, column)``) in the ``Common API`` depends on the method or property used, and this can normally be
determined from the property or method name. Properties and methods containing
``pixel`` assume ``(x, y)`` ordering, while properties and methods containing
``array`` assume ``(row, column)`` ordering.

Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__

- `astropy <http://www.astropy.org/>`__

- `asdf <https://asdf.readthedocs.io/en/latest/>`__

To install from source::

    git clone https://github.com/spacetelescope/gwcs.git
    cd gwcs
    python setup.py install

To install the latest release::

    pip install gwcs

The latest release of GWCS is also available as part of `astroconda <https://github.com/astroconda/astroconda>`__.


.. _getting-started:

Basic Structure of a GWCS Object
--------------------------------

The key concept to be aware of is that a GWCS Object consists of a pipeline
of steps; each step contains a transform (i.e., an Astropy model) that
converts the input coordinates of the step to the output coordinates of
the step. Furthermore, each step has an optional coordinate frame associated
with the step. The coordinate frame represents the input coordinate frame, not
the output coordinates. Most typically, the first step coordinate frame is
the detector pixel coordinates (the default). Since no step has a coordinate
frame for the output coordinates, it is necessary to append a step with no
transform to the end of the pipeline to represent the output coordinate frame.
For imaging, this frame typically references one of the Astropy standard
Sky Coordinate Frames of Reference. The GWCS frames also serve to hold the
units on the axes, the names of the axes and the physical type of the axis
(e.g., wavelength).

Since it is often useful to obtain coordinates in an intermediate frame of
reference, GWCS allows the pipeline to consist of more than one transform.
For example, for spectrographs, it is useful to have access to coordinates
in the slit plane, and in such a case, the first step would transform from
the detector to the slit plane, and the second step from the slit plane to
sky coordinates and a wavelength. Constructed this way, it is possible to
extract from the GWCS the needed transforms between identified frames of
reference.

The GWCS object can be saved to the ASDF format using the
`asdf <https://asdf.readthedocs.io/en/latest/>`__ package and validated
using `ASDF Standard <https://asdf-standard.readthedocs.io/en/latest/>`__

There are two ways to save the GWCS object to a files:

- `Save a WCS object as a pure ASDF file`_

- `Save a WCS object as a pure ASDF file`_


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
  with the approprate orientation on the sky. For simplicity's sake,
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
- Longitude of celestial pole in input coordinate system. With north up, this
  always corresponds to a value of 180.

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

.. doctest-skip::

  >>> wcsobj.invert(*sky)
  (0.9999999996185807, 1.999999999186798)

GWCS supports the common WCS interface which defines several methods
to work with high level Astropy objects:

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


Other Examples
--------------

.. toctree::
  :maxdepth: 2

  gwcs/imaging_with_distortion.rst
  gwcs/ifu.rst



Using ``gwcs``
--------------

.. toctree::
  :maxdepth: 2

  gwcs/wcs_ape.rst
  gwcs/using_wcs.rst
  gwcs/wcstools.rst
  gwcs/pure_asdf.rst
  gwcs/wcs_validation.rst
  gwcs/points_to_wcs.rst
  gwcs/fits_analog.rst


See also
--------

- `The modeling  package in astropy
  <http://docs.astropy.org/en/stable/modeling/>`__

- `The coordinates package in astropy
  <http://docs.astropy.org/en/stable/coordinates/>`__

- `The Advanced Scientific Data Format (ASDF) standard
  <https://asdf-standard.readthedocs.io/>`__
  and its `Python implementation
  <https://asdf.readthedocs.io/>`__


Reference/API
-------------

.. automodapi:: gwcs.wcs
.. automodapi:: gwcs.coordinate_frames
.. automodapi:: gwcs.wcstools
.. automodapi:: gwcs.selector
.. automodapi:: gwcs.spectroscopy
.. automodapi:: gwcs.geometry

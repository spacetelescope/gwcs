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

Getting Started
---------------

The WCS data model represents a pipeline of transformations between two
coordinate frames, the final one usually a physical coordinate system.
It is represented as a list of steps executed in order. Each step defines a
starting coordinate frame and the transform to the next frame in the pipeline.
The last step has no transform, only a frame which is the output frame of
the total transform. As a minimum a WCS object has an ``input_frame`` (defaults to "detector"),
an ``output_frame`` and the transform between them.

The WCS is validated using the `ASDF Standard <https://asdf-standard.readthedocs.io/en/latest/>`__
and serialized to file using the  `asdf <https://asdf.readthedocs.io/en/latest/>`__ package.
There are two ways to save the WCS to a file:

- `Save a WCS object as a pure ASDF file`_

- `Save a WCS object as an ASDF extension in a FITS file`_


A step by step example of constructing an imaging GWCS object.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


The following imports are generally useful:

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

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

  >>> sky = wcsobj(1*u.pix, 2*u.pix, with_units=True)
  >>> print(sky)
  <SkyCoord (ICRS): (ra, dec) in deg
    (5.52515954, -72.05190935)>

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
if available, otherwise applies an iterative method to calculate the reverse coordinates.

  >>> wcsobj.invert(sky)
  (<Quantity 1. pix>, <Quantity 2. pix>)

.. _save_as_asdf:

Save a WCS object as a pure ASDF file
+++++++++++++++++++++++++++++++++++++

.. doctest-skip::

  >>> from asdf import AsdfFile
  >>> tree = {"wcs": wcsobj}
  >>> wcs_file = AsdfFile(tree)
  >>> wcs_file.write_to("imaging_wcs.asdf")


:ref:`pure_asdf`


Save a WCS object as an ASDF extension in a FITS file
+++++++++++++++++++++++++++++++++++++++++++++++++++++


.. doctest-skip::

  >>> from astropy.io import fits
  >>> from asdf import fits_embed
  >>> hdul = fits.open("example_imaging.fits")
  >>> hdul.info()
  Filename: example_imaging.fits
  No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU     775   ()
  1  SCI           1 ImageHDU        71   (600, 550)   float32
  >>> tree = {"sci": hdul[1].data,
  ...         "wcs": wcsobj}
  >>> fa = fits.embed.AsdfInFits(hdul, tree)
  >>> fa.write_to("imaging_with_wcs_in_asdf.fits")
  >>> fits.info("imaging_with_wcs_in_asdf.fits")
  Filename: example_with_wcs.asdf
  No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU     775   ()
  1  SCI           1 ImageHDU        71   (600, 550)   float32
  2  ASDF          1 BinTableHDU     11   1R x 1C   [5086B]

Reading a WCS object from a file
++++++++++++++++++++++++++++++++


`ASDF <https://asdf.readthedocs.io/>`__ is used to read a WCS object
from a pure ASDF file or from an ASDF extension in a FITS file.


.. doctest-skip::

  >>> import asdf
  >>> asdf_file = asdf.open("imaging_wcs.asdf")
  >>> wcsobj = asdf_file.tree['wcs']


.. doctest-skip::

  >>> import asdf
  >>> fa = asdf.open("imaging_with_wcs_in_asdf.fits")
  >>> wcsobj = fa.tree["wcs"]

Other Examples
--------------

.. toctree::
  :maxdepth: 2

  gwcs/imaging_with_distortion.rst
  gwcs/ifu.rst



Using `gwcs`
------------

.. toctree::
  :maxdepth: 2

  gwcs/wcs_ape.rst
  gwcs/using_wcs.rst
  gwcs/wcstools.rst
  gwcs/pure_asdf.rst
  gwcs/wcs_validation.rst
  gwcs/schemas/index.rst
  gwcs/points_to_wcs.rst


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

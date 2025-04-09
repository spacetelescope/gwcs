.. _gwcs_overview:

Overview
========

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
using `ASDF Standard <https://asdf-standard.readthedocs.io/en/latest/>`__.
See: :ref:`save_as_asdf`, for more information on saving a WCS object as an ASDF
file.

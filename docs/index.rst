GWCS Documentation
==================

`GWCS <https://github.com/spacetelescope/gwcs>`__ is a package for managing
the World Coordinate System (WCS) of astronomical data.

Introduction
------------


`GWCS <https://github.com/spacetelescope/gwcs>`__ takes a general approach to WCS.
It supports a data model which includes the entire transformation pipeline from
input coordinates (detector by default)  to world coordinates.
Transformations can be chained, joined or combined with arithmetic operators
using the flexible framework of compound models in `~astropy.modeling`.
In the case of a celestial output frame `~astropy.coordinates` provides
further transformations between standard coordinate frames.
Spectral output coordinates are instances of `~astropy.units.Quantity`  and are
transformed to other units with the tools in that package. The goal is to provide
a flexible toolkit which is easily extendable by adding new transforms and frames.


Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__ 1.7 or later

- `astropy <http://www.astropy.org/>`__ 1.2 or later

- `asdf <https://asdf.readthedocs.io/en/latest/>`__

To install from source::

    git clone https://github.com/spacetelescope/gwcs.git
    cd gwcs
    python setup.py install

To install the latest release::

    pip install gwcs

GWCS is also available as part of `astroconda <https://github.com/astroconda/astroconda>`__.

.. _getting-started:

Getting Started
===============

The WCS data model represents a pipeline of transformations from some
initial coordinate frame to a standard coordinate frame.
It is implemented as a list of steps executed in order. Each step defines a
starting coordinate frame and the transform to the next frame in the pipeline.
The last step has no transform, only a frame which is the output frame of
the total transform. As a minimum a WCS object has an ``input_frame`` (defaults to "detector"),
an ``output_frame`` and the transform between them.

As an example, consider a typical WCS of an image without distortion.

The following imports are generally useful:

  >>> import numpy as np
  >>> from astropy.modeling.models import (Shift, Scale, Rotation2D,
      Pix2Sky_TAN, RotateNative2Celestial, Mapping, Polynomial2D)
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

The ``forward_transform`` is constructed as a combined model using `astropy.modeling`.
The frames are subclasses of `~gwcs.coordinate_frames.CoordinateFrame` (although strings are
acceptable too).

Create a transform to convert detector coordinates to ICRS.

  >>> det2sky = (Shift(-10.5) & Shift(-13.2) | Rotation2D(0.0023) | \
      Scale(.01) & Scale(.04) | Pix2Sky_TAN() | \
      RotateNative2Celestial(5.6, -72.05, 180)).rename("detector2sky")

Create a coordinate frame associated with the detector and a celestial frame.

  >>> detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')

Initialize the WCS:

  >>> wcsobj = wcs.WCS(forward_transform=det2sky, input_frame=detector_frame, output_frame=sky_frame)
  >>> print(wcsobj)
      From        Transform
      ----------- ----------
      detector     detector2sky
      icrs         None

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

  >>> sky = wcsobj(1, 2)
  >>> print(sky)
      (5.284139265842838, -72.49775640633504)

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
if available, otherwise applies an iterative method to calculate the reverse coordinates.

  >>> wcsobj.invert(*sky)
      (1.000, 2.000)


Using `gwcs`
------------


.. toctree::
  :maxdepth: 2

  gwcs/using_wcs.rst
  gwcs/selector_model.rst
  gwcs/wcstools.rst



See also
--------

- The `modeling  package in astropy
  <http://docs.astropy.org/en/stable/modeling/>`__

- The `Coordinates package in astropy
  <http://docs.astropy.org/en/stable/coordinates/>`__

- The `Advanced Scientific Data Format (ASDF) standard
  <https://asdf-standard.readthedocs.io/>`__
  and its `Python implementation
  <https://asdf.readthedocs.io/>`__


Reference/API
-------------

.. automodapi:: gwcs.wcs
.. automodapi:: gwcs.coordinate_frames
.. automodapi:: gwcs.selector
.. automodapi:: gwcs.wcstools

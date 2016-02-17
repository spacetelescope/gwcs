GWCS Documentation
====================

`GWCS <https://github.com/spacetelescope/gwcs>`__ is a package for constructing and managing
the World Coordinate System (WCS) of astronomical data.

Introduction
------------


`GWCS <https://github.com/spacetelescope/gwcs>`__ takes a general approach to WCS.
It supports a data model which includes the entire transformation pipeline from
input coordinates (detector by default)  to world cooridnates.
Transformations can be chained, joined or combined with arithmetic operators
using the flexible framework of compound models in `~astropy.modeling`.
In the case of a celestial output frame `~astropy.coordinates` provides
further transformations between standard coordinate frames.
Spectral output coordinates are instances of `~astropy.units.Quantity`  and are
transformed to other units with the tools in that package.
The goal is to provide a flexible toolkit which is easily extendable by adding new
transforms and frames.


Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__ 1.6 or later

- `astropy <http://www.astropy.org/>`__ 1.1 or later

- `pyasdf <http://pyasdf.readthedocs.org/en/latest/>`__


Getting Started
---------------

The simplest way to initialize a WCS object is to pass a `forward_transform` and an `output_frame`
to `~gwcs.wcs.WCS`. As an example, consider a typical basic FITS WCS of an image.
The following imports are generally useful:

  >>> from astropy.modeling.models import (Shift, Scale, Rotation2D,
      Pix2Sky_TAN, RotateNative2Celestial, Mapping, Polynomial2D)
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames

The `forward_transform` is constructed as a combined model using `astropy.modeling`. The frames
can be strings or subclasses of `~gwcs.coordinate_frames.CoordinateFrame`.

  >>> transform = Shift(-10.5) & Shift(-13.2) | Rotation2D(0.0023) | \
      Scale(.01) & Scale(.04) | Pix2Sky_TAN() | RotateNative2Celestial(5.6, -72.05, 180)
  >>> sky_frame = coordinate_frames.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
  >>> w = wcs.WCS(forward_transform=transform, output_frame=sky_frame)

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

  >>> sky = w(1, 2)
  >>> print(sky)
      (5.284139265842845, -72.49775640633503)

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
if available, otherwise applies an iterative method to calculate the reverse coordinates.

  >>> w.invert(*sky)
      (1.000000000009388, 2.0000000000112728)

Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

  >>> dist = w.get_transform('detector', 'focal')
  >>> dist(1, 2)
      (0.16, 0.8)

Transforms in the pipeline can be replaced by new transforms.

  >>> new_transform = Shift(1) & Shift(1.5) | distortion
  >>> w.set_transform('detector', 'focal', new_transform)
  >>> w(1, 2)
      (5.257230028926096, -72.53171157138964)

A transform can be inserted before or after a frame in the pipeline.

  >>> scale = Scale(2) & Scale(1)
  >>> w.insert_transform('icrs', scale, after=False)
  >>> w(1, 2)
      (10.514460057852192, -72.53171157138964)


Using `gwcs`
------------


.. toctree::
  :maxdepth: 2

  gwcs/create_wcs
  gwcs/selector_model.rst
  gwcs/wcstools.rst


See also
--------

- The `modeling  package in astropy
  <http://docs.astropy.org/en/stable/modeling/>`__

- The `Coordinates package in astropy
  <http://docs.astropy.org/en/stable/coordinates/>`__

- The `Advanced Scientific Data Format (ASDF) standard
  <http://asdf-standard.readthedocs.org/>`__
  and its `Python implementation
  <http://pyasdf.readthedocs.org/>`__


Reference/API
-------------

.. automodapi:: gwcs.wcs
.. automodapi:: gwcs.coordinate_frames
.. automodapi:: gwcs.selector
.. automodapi:: gwcs.wcstools
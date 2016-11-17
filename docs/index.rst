GWCS Documentation
==================

`GWCS <https://github.com/spacetelescope/gwcs>`__ is a package for constructing and managing
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
transformed to other units with the tools in that package.
The goal is to provide a flexible toolkit which is easily extendable by adding new
transforms and frames.


Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__ 1.7 or later

- `astropy <http://www.astropy.org/>`__ 1.1 or later

- `asdf <http://pyasdf.readthedocs.org/en/latest/>`__ 1.2.1


Getting Started
---------------

The simplest way to initialize a WCS object is to pass a ``forward_transform`` and an ``output_frame``
to `~gwcs.wcs.WCS`. As an example, consider a typical basic FITS WCS of an image.
The following imports are generally useful:

  >>> from astropy.modeling.models import (Shift, Scale, Rotation2D,
      Pix2Sky_TAN, RotateNative2Celestial, Mapping, Polynomial2D)
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

The ``forward_transform`` is constructed as a combined model using `~astropy.modeling`. The frames
can be strings or subclasses of `~gwcs.coordinate_frames.CoordinateFrame`. There are additional benefits
having them as objects.

  >>> transform = Shift(-10.5) & Shift(-13.2) | Rotation2D(0.0023) | \
      Scale(.01) & Scale(.04) | Pix2Sky_TAN() | RotateNative2Celestial(5.6, -72.05, 180)
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
  >>> wcsobj = wcs.WCS(forward_transform=transform, output_frame=sky_frame)

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

  >>> ra, dec = wcsobj(x, y)
  >>> print(ra, dec)
      (5.284139265842845, -72.49775640633503)

It is possible to get the result as a `~astropy.coordinates.SkyCoord` object`:

  >>> sky_coord = wcsobj(x, y, output="numericals_plus")
  >>> print(sky_coord)
      <SkyCoord (ICRS): (ra, dec) in deg
          (5.28413927, -72.49775641)>

This result can now be transformed to any other Celestial coordinate system supported by
`astropy.coordinates`.

  >>> sky_coord.transform_to("galactic")
      <SkyCoord (Galactic): (l, b) in deg
          (306.1152941, -44.44272451)>

The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
if available, otherwise applies an iterative method to calculate the reverse coordinates.

  >>> wcsobj.invert(ra, dec)
      (1.000000000009388, 2.0000000000112728)


Using `gwcs`
------------


.. toctree::
  :maxdepth: 2

  gwcs/using_wcs.rst
  gwcs/create_wcs.rst
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

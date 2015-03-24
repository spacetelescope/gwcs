GWCS Documentation
====================

`gwcs <https://github.com/spacetelescope/gwcs>`__ is a package for computing the World Coordinate System (WCS) of astronomical data.

Introduction
------------

This package provides tools to construct and work with WCS objects in a general way.
It uses the standard cordinate systems defined in
`astropy.coordinates <http://docs.astropy.org/en/stable/coordinates>`__, typically as an output
coordinate system which then can be transformed to another standard system supported
by `astropy.coordinates <http://docs.astropy.org/en/stable/coordinates>`__. The WCS provides
the transformation between input coordinates (usually on a detector) and the output coordinate system.
These can be compound transformations combining simple models in various ways - they can be chained,
joined or used in arithmetic operations. Although not limited to, this package was designed to use
the flexible framework of compound models in
`astropy.modeling <http://docs.astropy.org/en/stable/modeling>`__ .
Being written in `Python <http://www.python.org>`__ , it is easily extendable.

Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__ 1.6 or later

- `astropy <http://www.astropy.org/>`__ 1.0 or later



Getting Started
---------------

Usually a GWCS object is attached to data, often an astronomical observation.
Most users will simply work with a predefined WCS object, i.e.
the WCS for a particular instrument would have been designed already by specifying the
input and output coordinate systems and the transformation between them.
The simplest way to use it is to call the WCS object like a function. This will transform
positions in the `~gwcs.wcs.WCS.input_coordinate_system` to positions in the `~gwcs.wcs.WCS.output_coordinate_system`.

Let's consider as an example an `~astropy.nddata.NDData` object which represents imaging data.
Given positions on the detector, the WCS computes positions
on the sky. To convert a pixel (x, y) = (1,2) to sky coordinates:


  >>> sky = nddata_image.wcs(1, 2)
  >>> print sky
  (5.52509373, -72.05190053)


If available, the backward transform can be evaluated using the :meth:`~gwcs.wcs.WCS.invert` method.


  >>> nddata_image.wcs.invert(*sky)
  (1.0000000938994162, 2.000000047071694)

The result of the forward transform can be turned into a SkyCoord object which then can be used to
transform to other standard coordinate systems using the `astropy.coordinates` framework.


  >>> sky_coord = nddata_image.wcs.output_frame.world_coordinates(*sky)
  >>> print sky_coord
  <SkyCoord (ICRS): (ra, dec) in deg
  (5.52509373, -72.05190053)>
  >>> sky_coord.transform_to('galactic')
  <SkyCoord (Galactic): (l, b) in deg
      (306.11529787, -44.89457423)>

In this framework transformations can be chained and combined in arbitrary ways.
The entire transform represents a linear pipeline of transformations
between coordinate systems. Sometimes it is interesting to perform only part of the
transformations.

Let part of the transform in the above example include distortion correction which
converts positions on the detector to positions in a system associated with the focal
plane of the telescope (called ``FocalPlaneFrame``). There are a couple of ways to perform
the transformation from detector to focal plane coordinates - The :meth:`~gwcs.wcs.transform`
method

  >>> nddata_image.wcs.transform('detector', 'focal_plane', 1, 2)

or in two steps, getting the transform first using :meth:`~gwcs.wcs.get_transform`
and then evaluating it:

  >>> distoriton = nddata_image.wcs.get_transform('detector', 'focal_plane')
  >>> distortion(1, 2)


Using `gwcs`
------------


.. toctree::
  :maxdepth: 2

  gwcs/create_wcs
  gwcs/selector_model.rst



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

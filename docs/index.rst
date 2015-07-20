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
the transformation between input coordinates (default is detector) and the output coordinate system.
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

The simplest way to initialize a WCS object is to pass a `forward_transform` and an `output_frame`
to `~gwcs.wcs.WCS`. As an example, consider a typical basic FITS WCS of a image.
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

It is possible to have intermediate frames in the WCS pipeline. In this case it is convenient
to initialize the WCS object with a list of tuples, where each tuple (step_frame, step_transform)
represents a transform "step_transform" from frame "step_frame" to the next frame in the WCS pipeline.
The transform in the last step is always None to indicate end of the pipeline.
If a "focal" frame and polynomial distortion are added to the above example:

  >>> focal = coordinate_frames.Frame2D(name='focal', unit=('arcsec', 'arcsec'))
  >>> distortion = Mapping([0, 1, 0, 1]) | Polynomial2D(1, c0_0=0.1, c1_0=.02, c0_1=.02) & Polynomial2D(1, c0_0=.4, c1_0=.2, c0_1=.1)
  >>> pipeline = [('detector', distortion), (focal, transform), (sky_frame, None)]
  >>> w = wcs.WCS(pipeline)
  >>> w(1, 2)
      (5.249343926993615, -72.57769537481136)

Frame objects allow to extend the functionality by using `astropy.coordinates` and `astropy.units`.
Frames are available as attributes of the WCS object.

  >>> w.available_frames
      ['detector', 'focal', 'icrs']
  >>> w.icrs
      <CelestialFrame(reference_frame=<ICRS Frame>, unit=[Unit("deg"), Unit("deg")], name=icrs)>
  >>> w.icrs.coordinates(1,2)
      <SkyCoord (ICRS): (ra, dec) in deg
          (5.24934393, -72.57769537)>
  >>> w.icrs.transform_to('galactic', 1, 2)
      <SkyCoord (Galactic): (l, b) in deg
          (306.11129272, -44.36215723)>

Some methods allow managing the transforms in a more detaield manner.

  >>> dist = w.get_transform('detector', 'focal')
  >>> dist(1, 2)
      (0.16, 0.8)

  >>> new_transform = Shift(1) & Shift(1.5) | distortion
  >>> w.set_transform('detector', 'focal', new_transform)
  >>> w(1, 2)
      (5.257230028926096, -72.53171157138964)

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

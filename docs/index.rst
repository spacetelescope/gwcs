GWCS Documentation
====================

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
transformed to other units with the tools in that package.
The goal is to provide a flexible toolkit which is easily extendable by adding new
transforms and frames.


Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__ 1.67 or later

- `astropy <http://www.astropy.org/>`__ 1.2 or later

- `asdf <http://pyasdf.readthedocs.org/en/latest/>`__

To install from source::

    git clone https://github.com/spacetelescope/gwcs.git
    cd gwcs
    python setup.py install

To install the latest release::

    pip install gwcs

GWCS is also available as part of astroconda.

Getting Started
---------------

The WCS data model represents a pipeline of transformations from some
initial coordinate frame to a standard coordinate frame.
It is implemented as a list of steps executed in order. Each step defines a
starting coordinate frame and the transform to the next frame in the pipeline.
The last step has no transform, only a frame which is the output frame of
the total transform. As a minimum a WCS object has an input_frame (defaults to "detector"),
an output_frame and the transform between them.

As an example, consider a typical WCS of an image with some distortion.

The following imports are generally useful:

  >>> import numpy as np
  >>> from astropy.modeling.models import (Shift, Scale, Rotation2D,
      Pix2Sky_TAN, RotateNative2Celestial, Mapping, Polynomial2D)
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs
  >>> from gwcs import coordinate_frames as cf

The `forward_transform` is constructed as a combined model using `astropy.modeling`.
The frames are subclasses of `~gwcs.coordinate_frames.CoordinateFrame` (although strings are
acceptable too).

First create polynomial transform to represent distortion:

  >>> polyx = Polynomial2D(4)
  >>> polyx.parameters = np.random.randn(15)
  >>> polyy = Polynomial2D(4)
  >>> polyy.parameters = np.random.randn(15)
  >>> distortion = (Mapping((0, 1, 0, 1)) | polyx & polyy).rename("distortion")

Next create a transform from distortion fdree coordinates to ICRS.

  >>> dist2sky = (Shift(-10.5) & Shift(-13.2) | Rotation2D(0.0023) | \
      Scale(.01) & Scale(.04) | Pix2Sky_TAN() | RotateNative2Celestial(5.6, -72.05, 180)).rename("focal2sky")

Create three coordinate frames, associated with the detector, a celestial frame, and an intermediate one.

  >>> detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
  >>> focal_frame = cf.Frame2D(name="focal_frame", unit=(u.arcsec, u.arcsec))

Create the WCS pipeline and initialize the WCS:

  >>> pipeline = [(detector_frame, distortion),
                  (focal_frame, dist2sky),
                  (sky_frame, None)
                  ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
      From        Transform
      ----------- ----------
      detector     distortion
      focal_frame  focal2sky
      icrs         None

To convert a pixel (x, y) = (1, 2) to sky coordinates, call the WCS object as a function:

  >>> sky = wcsobj(1, 2)
  >>> print(sky)
      (5.759024831907874, -72.22626601919619)

#The :meth:`~gwcs.wcs.WCS.invert` method evaluates the :meth:`~gwcs.wcs.WCS.backward_transform`
#if available, otherwise applies an iterative method to calculate the reverse coordinates.
#
#  >>> w.invert(*sky)
#      (1.000000000009388, 2.0000000000112728)
#
#Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

  >>> dist = w.get_transform('detector_frame', 'focal_frame')
  >>> dist(1, 2)
      (15.354214305518118, 8.791536957201615)

Transforms in the pipeline can be replaced by new transforms.

  >>> new_transform = distortion | Shift(1) & Shift(1.5)
  >>> wcsobj.set_transform('detector', 'focal', new_transform)
  >>> wcsobj(1, 2)
      (5.791157736884894, -72.16623599444335)

A transform can be inserted before or after a frame in the pipeline.

  >>> scale = Scale(2) & Scale(1)
  >>> wcsobj.insert_transform('icrs', scale, after=False)
  >>> wcsobj(1, 2)
      (11.582315473769787, -72.16623599444335)


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

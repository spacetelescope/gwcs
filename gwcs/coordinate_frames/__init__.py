"""
This module defines coordinate frames for describing the inputs and/or outputs
of a transform.

In the block diagram, the WCS pipeline has a two stage transformation (two
astropy Model instances), with an input frame, an output frame and an
intermediate frame.

.. code-block::

    ┌───────────────┐
    │               │
    │     Input     │
    │     Frame     │
    │               │
    └───────┬───────┘
            │
      ┌─────▼─────┐
      │ Transform │
      └─────┬─────┘
            │
    ┌───────▼───────┐
    │               │
    │  Intermediate │
    │     Frame     │
    │               │
    └───────┬───────┘
            │
      ┌─────▼─────┐
      │ Transform │
      └─────┬─────┘
            │
    ┌───────▼───────┐
    │               │
    │    Output     │
    │     Frame     │
    │               │
    └───────────────┘


Each frame instance is both metadata for the inputs/outputs of a transform and
also a converter between those inputs/outputs and richer coordinate
representations of those inputs/outputs.

For example, an output frame of type `~gwcs.coordinate_frames.SpectralFrame`
provides metadata to the `.WCS` object such as the ``axes_type`` being
``"SPECTRAL"`` and the unit of the output etc.  The output frame also provides a
converter of the numeric output of the transform to a
`~astropy.coordinates.SpectralCoord` object, by combining this metadata with the
numerical values.

``axes_order`` and conversion between objects and arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key concepts regarding coordinate frames is the ``axes_order`` argument.
This argument is used to map from the components of the frame to the inputs/outputs
of the transform.  To illustrate this consider this situation where you have a
forward transform which outputs three coordinates ``[lat, lambda, lon]``. These
would be represented as a `.SpectralFrame` and a `.CelestialFrame`, however, the
axes of a `.CelestialFrame` are always ``[lon, lat]``, so by specifying two
frames as

.. code-block:: python

  [SpectralFrame(axes_order=(1,)), CelestialFrame(axes_order=(2, 0))]

we would map the outputs of this transform into the correct positions in the frames.
 As shown below, this is also used when constructing the inputs to the inverse
 transform.


When taking the output from the forward transform the following transformation
is performed by the coordinate frames:

.. code-block::

                   lat, lambda, lon
                   │      │     │
                   └──────┼─────┼────────┐
              ┌───────────┘     └──┐     │
              │                    │     │
    ┌─────────▼────────┐    ┌──────▼─────▼─────┐
    │                  │    │                  │
    │  SpectralFrame   │    │  CelestialFrame  │
    │                  │    │                  │
    │       (1,)       │    │      (2, 0)      │
    │                  │    │                  │
    └─────────┬────────┘    └──────────┬────┬──┘
              │                        │    │
              │                        │    │
              ▼                        ▼    ▼
   SpectralCoord(lambda)    SkyCoord((lon, lat))


When considering the backward transform the following transformations take place
in the coordinate frames before the transform is called:

.. code-block::

   SpectralCoord(lambda)    SkyCoord((lon, lat))
              │                        │    │
              └─────┐     ┌────────────┘    │
                    │     │    ┌────────────┘
                    ▼     ▼    ▼
                [lambda, lon, lat]
                    │     │    │
                    │     │    │
             ┌──────▼─────▼────▼────┐
             │                      │
             │  Sort by axes_order  │
             │                      │
             └────┬──────┬─────┬────┘
                  │      │     │
                  ▼      ▼     ▼
                 lat, lambda, lon

"""

from ._base import BaseCoordinateFrame
from ._coordinate_frames import (
    CelestialFrame,
    CompositeFrame,
    CoordinateFrame,
    EmptyFrame,
    Frame2D,
    SpectralFrame,
    StokesFrame,
    TemporalFrame,
    get_ctype_from_ucd,
)

__all__ = [
    "BaseCoordinateFrame",
    "CelestialFrame",
    "CompositeFrame",
    "CoordinateFrame",
    "EmptyFrame",
    "Frame2D",
    "SpectralFrame",
    "StokesFrame",
    "TemporalFrame",
    "get_ctype_from_ucd",
]

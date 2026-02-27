"""
This module defines coordinate frames for describing the inputs and/or outputs
of a transform.

In the block diagram, the WCS pipeline has a two stage transformation (two
astropy Model instances), with an input frame, an output frame and an
intermediate frame.

.. tab:: Diagram

    .. graphviz::

        digraph wcs_pipeline {
            rankdir=TB;
            node [shape=box, style=filled, fillcolor=lightblue, fontname="Helvetica", margin="0.3,0.3"];
            edge [fontname="Helvetica"];

            // Frame nodes
            input_frame [label="Input\\nFrame"];
            intermediate_frame [label="Intermediate\\nFrame", shape=diamond];
            output_frame [label="Output\\nFrame"];

            // Transform nodes
            transform1 [label="Transform", shape=ellipse, fillcolor=lightgrey];
            transform2 [label="Transform", shape=ellipse, fillcolor=lightgrey];

            // Connections
            input_frame -> transform1;
            transform1 -> intermediate_frame;
            intermediate_frame -> transform2;
            transform2 -> output_frame;
        }

.. tab:: ASCII

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

.. tab:: Diagram

    .. graphviz::
        :align: center

        digraph forward_transform {
            splines="ortho"
            rankdir=TB;
            node [style=filled, fontname="Helvetica", margin="0.3,0.3"];
            edge [fontname="Helvetica"];

            // Input coordinates
            input_coords [label="lat, lambda, lon", shape=plaintext];

            // Frame nodes with axes_order
            spectral_frame [label="SpectralFrame\\n(1,)", fillcolor=lightblue];
            celestial_frame [label="CelestialFrame\\n(2, 0)", fillcolor=lightblue];

            // Output coordinates
            spectral_output [label="SpectralCoord(lambda)", shape=record, fillcolor=lightyellow];
            celestial_output [label="SkyCoord((lon, lat))", shape=record, fillcolor=lightyellow];

            // Connections with routing
            input_coords -> spectral_frame [xlabel="lambda"];
            input_coords -> celestial_frame [xlabel="lon  "];
            input_coords -> celestial_frame [xlabel="lat  "];
            spectral_frame -> spectral_output;
            celestial_frame -> celestial_output [xlabel="lon  "];
            celestial_frame -> celestial_output [xlabel="lat  "]
        }

.. tab:: ASCII

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

.. tab:: Diagram

    .. graphviz::
        :align: center

        digraph backward_transform {
            splines="ortho"
            rankdir=TB;
            node [style=filled, fontname="Helvetica", margin="0.3,0.3"];
            edge [fontname="Helvetica"];

            // Input high-level objects
            spectral_input [label="SpectralCoord(lambda)", shape=record, fillcolor=lightyellow];
            celestial_input [label="SkyCoord((lon, lat))", shape=record, fillcolor=lightyellow];

            // Initial array from high-level objects
            array_unsorted [label="[lambda, lon, lat]", shape=plaintext];

            // Sorting operation
            sort_axes [label="Sort by axes_order", fillcolor=lightgrey];

            // Final sorted array for transform input
            array_sorted [label="lat, lambda, lon", shape=plaintext];

            // Connections
            spectral_input -> array_unsorted [xlabel="lambda "];
            celestial_input -> array_unsorted [xlabel="lon  "];
            celestial_input -> array_unsorted [xlabel="lat  "];
            array_unsorted -> sort_axes;
            sort_axes -> array_sorted;
        }

.. tab:: ASCII

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

"""  # noqa: E501

from ._axis import AxisType
from ._base import (
    BaseCoordinateFrame,
    CoordinateFrameProtocol,
    WorldAxisObjectClasses,
    WorldAxisObjectComponent,
)
from ._celestial import CelestialFrame
from ._composite import CompositeFrame
from ._core import CoordinateFrame
from ._empty import EmptyFrame, EmptyFrameDeprecationWarning
from ._frame import Frame2D
from ._spectral import SpectralFrame
from ._stokes import StokesFrame
from ._temporal import TemporalFrame
from ._utils import get_ctype_from_ucd

__all__ = [
    "AxisType",
    "BaseCoordinateFrame",
    "CelestialFrame",
    "CompositeFrame",
    "CoordinateFrame",
    "CoordinateFrameProtocol",
    "EmptyFrame",
    "EmptyFrameDeprecationWarning",
    "Frame2D",
    "SpectralFrame",
    "StokesFrame",
    "TemporalFrame",
    "WorldAxisObjectClasses",
    "WorldAxisObjectComponent",
    "get_ctype_from_ucd",
]

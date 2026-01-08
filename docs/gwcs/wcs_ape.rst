.. _ape14:

Shared Interface for World Coordinate System - APE 14
=====================================================

To improve interoperability between packages, the Astropy Project and other interested
parties have collaboratively defined a standardized application
programming interface (API) for world coordinate system objects to be used
in Python. This API is described in the Astropy Proposal for Enhancements (APE) 14:
`A shared Python interface for World Coordinate Systems
<https://doi.org/10.5281/zenodo.1188874>`_.

The base classes that define the low- (`~astropy.wcs.wcsapi.BaseLowLevelWCS`) and high-
(:class:`~astropy.wcs.wcsapi.BaseHighLevelWCS`) level APIs are in astropy.
GWCS implements both APIs. Once a gWCS object is created the API methods will be available.
It is recommended that applications use the ``Shared API`` to
ensure transparent use of ``GWCS`` and ``FITSWCS`` objects.

The example below illustrates the capabilities of the ``Shared API``.
The High Level interface provides methods for transforming coordinates between the input
and the output frames, generating High Level Objects (HLO). HLOs are rich astropy objects,
like ``SkyCoord``, ``SpectralCoord`` and ``StokesCoord``, which provide additional functionality.

.. doctest-skip::

  >>> from gwcs import examples
  >>> wcsobj = examples.gwcs_3d_spatial_wave()
  >>> result = wcsobj.pixel_to_world(1, 1, 1)
  >>> print(result)
   [<SkyCoord (ICRS): (ra, dec) in deg
    (2., 3.)>, <SpectralCoord 2. m>]
  >>> wcsobj.world_to_pixel(*result)
  (1.0, 1.0, 1.0)

Two other methods in the High Level Interface, ``pixel_to_world_values`` and ``world_pixel_values``
return the numerical results.

Note that there's an implicit assumption in the names of the methods (``pixel_to_world``
and ``world_to_pixel``) that the input is in pixels. However, this is only an unfortunate
naming choice. In reality the ``pixel_to_world`` method executes the ``forward_transform`` transform,
and ``world_to_pixel`` evaluates the ``backward_transform`` as defined in the WCS object.
It is possible to define the forward transform from sky to detector, in which case the output
of ``pixel_to_world`` will be in units of pixels.

The Low Level Interface, provides additional methods which may be useful in WCS introspections.
We list some of them using the example WCS above:

.. doctest-skip::

  >>> wcsobj.pixel_n_dim
  3
  >>> wcsobj.world_n_dim
  3
  >>> wcsobj.world_axis_names
  ('lon', 'lat', 'lambda')
  >>> wcsobj.world_axis_physical_types
  ('pos.eq.ra', 'pos.eq.dec', 'em.wl')
  >>> wcsobj.world_axis_units
  ('deg', 'deg', 'm')

A WCS object may not have a data array attached to it, as it represents a coordinate transformation.
However, it is usually read in from a data file and in this case there's a data array. The Low Level
Interface can be used to find out the shape if the array. As discussed
in :ref:`pixel-conventions-and-definitions` the WCS uses cartesian order of the coordinates, and hence
there are two methods for the shape of the data array. Let's assume a data array of shape (4, 5) is
in a file with the WCS object above. The two methods will return respectively:

.. doctest-skip::

  >>> wcsobj.pixel_shape
  (5, 4)
  >>> wcsobj.array_shape
  (4, 5)

If the WCS object has a ``bounding_box``, it can be accessed by

.. doctest-skip::

  >>> wcsobj.bounding_box = ((2, 10), (3, 7), (.1, .7))
  >>> wcsobj.pixel_bounds
  ((2, 10), (3, 7), (0.1, 0.7))

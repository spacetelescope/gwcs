0.16.1 (2020-12-20)
-------------------
Bug Fixes
^^^^^^^^^
-Fix a regression with ``pixel_to_world`` for output frames with one axis. [#342]

0.16.0 (2020-12-18)
-------------------
New Features
^^^^^^^^^^^^

- Added an option to `to_fits_sip()` to be able to specify the reference
  point (``crpix``) of the FITS WCS. [#337]

- Added support for providing custom range of degrees in ``to_fits_sip``. [#339]

Bug Fixes
^^^^^^^^^

- ``bounding_box`` now works with tuple of ``Quantities``. [#331]

- Fix a formula for estimating ``crpix`` in ``to_fits_sip()`` so that ``crpix``
  is near the center of the bounding box. [#337]

- Allow sub-pixel sampling of the WCS model when computing SIP approximation in
  ``to_fits_sip()``. [#338]

- Fixed a bug in ``to_fits_sip`` due to which ``inv_degree`` was ignored. [#339]


0.15.0 (2020-11-13)
-------------------
New Features
^^^^^^^^^^^^

- Added ``insert_frame`` method to modify the pipeline of a ``WCS`` object. [#299]

- Added ``to_fits_tab`` method to generate FITS header and binary table
  extension following FITS WCS ``-TAB`` convension. [#295]

- Added ``in_image`` function for testing whether a point in world coordinates
  maps back to the domain of definition of the forward transformation. [#322]

- Implemented iterative inverse for some imaging WCS. [#324]

0.14.0 (2020-08-19)
-------------------
New Features
^^^^^^^^^^^^

- Updated versions of schemas for gwcs objects based on latest versions of
  transform schemas in asdf-standard. [#307]

- Added a ``wcs.Step`` class to allow serialization to ASDF to use references. [#317]

- ``wcs.pipeline`` now is a list of ``Step`` instances instead of
  a (frame, transform) tuple. Use ``WCS.pipeline.transform`` and
  ``WCS.pipeline.frame`` to access them. [#319]

Bug Fixes
^^^^^^^^^

- Fix a bug in polygon fill for zero-width bounding boxes. [#293]

- Add an optional parameter ``input_frame`` to ``wcstools.wcs_from_fiducial`. [#312]

0.13.0 (2020-03-26)
-------------------
New Features
^^^^^^^^^^^^

- Added two new transforms - ``SphericalToCartesian`` and
  ``CartesianToSpherical``. [#275, #284, #285]

- Added ``to_fits_sip`` method to generate FITS header with SIP keywords [#286]

- Added ``get_ctype_from_ucd`` function. [#288]

Bug Fixes
^^^^^^^^^

- Fixed an off by one issue in ``utils.make_fitswcs_transform``. [#290]

0.12.0 (2019-12-24)
-------------------
New Features
^^^^^^^^^^^^

- ``gwcs.WCS`` now supports the ``world_axis_object_components`` and
  ``world_axis_object_classes`` methods of the low level WCS API as specified by
  APE 14.

- Removed astropy-helpers from package. [#249]

- Added a method ``fix_inputs`` which rturns an unique WCS from a compound
  WCS by fixing inputs. [#254]

- Added two new transforms - ``ToDirectionCosines`` and ``FromDirectionCosines``. [#256]

- Added new transforms ``WavelengthFromGratingEquation``, ``AnglesFromGratingEquation3D``. [#259]

- ``gwcs.WCS`` now supports the new ``world_axis_names`` and
  ``pixel_axis_names`` properties on ``LowLevelWCS`` objects. [#260]

- Update the ``StokesFrame`` to work for arrays of coordinates and integrate
  with APE 14. [#258]

- Added ``Snell3D``, ``SellmeierGlass`` and ``SellmeierZemax`` transforms. [#270]

API Changes
^^^^^^^^^^^

- Changed the initialization of ``TemporalFrame`` to be consistent with other
   coordinate frames. [#242]

Bug Fixes
^^^^^^^^^

- Ensure that ``world_to_pixel_values`` and ``pixel_to_world_values`` always
  accept and return floats, even if the underlying transform uses units. [#248]

0.11.0 (2019/07/26)
-------------------

New Features
^^^^^^^^^^^^

- Add a schema and tag for the Stokes frame. [#164]

- Added ``WCS.pixel_shape`` property. [#233]


Bug Fixes
^^^^^^^^^

- Update util.isnumerical(...) to recognize big-endian types as numeric. [#225]

- Fixed issue in unified WCS API (APE14) for transforms that use
  ``Quantity``. [#222]

- Fixed WCS API issues when ``output_frame`` is 1D, e.g. ``Spectral`` only. [#232]


0.10.0 (12/20/2018)
-------------------

New Features
^^^^^^^^^^^^

- Initializing a ``WCS`` object with a ``pipeline`` list now keeps
  the complete ``CoordinateFrame`` objects in the ``WCS.pipeline``.
  The effect is that a ``WCS`` object can now be initialized with
  a ``pipeline`` from a different ``WCS`` object. [#174]

- Implement support for astropy APE 14
  (https://doi.org/10.5281/zenodo.1188875). [#146]

- Added a ``wcs_from_[points`` function which creates a WCS object
  two matching sets of points ``(x,y)`` and ``(ra, dec)``. [#42]


0.9.0 (2018-05-23)
------------------

New Features
^^^^^^^^^^^^

- Added a ``TemporalFrame`` to represent relative or absolute time axes. [#125]

- Removed deprecated ``grid_from_domain`` function and ``WCS.domain`` property. [#119]

- Support for Python 2.x, 3.0, 3.1, 3.2, 3.3 and 3.4 was removed. [#119]

- Add a ``coordinate_to_quantity`` method to ``CoordinateFrame`` which handles
  converting rich coordinate input to numerical values. It is an inverse of the
  ``coordinates`` method. [#133]

- Add a ``StokesFrame`` which converts from 'I', 'Q', 'U', 'V' to 0-3. [#133]

- Support serializing the base ``CoordinateFrame`` class to asdf, by making
  a specific tag and schema for ``Frame2D``. [#150]

- Generalized the footrpint calculation to all output axes. [#167]


API Changes
^^^^^^^^^^^

- The argument ``output="numerical_plus"`` was replaced by a bool
  argument ``with_units``. [#156]

- Added a new flag ``axis_type`` to the footprint method. It controls what
  type of footprint to calculate. [#167]

Bug Fixes
^^^^^^^^^

- Fixed a bug in ``bounding_box`` definition when the WCS has only one axis. [#117]

- Fixed a bug in ``grid_from_bounding_box`` which caused the grid to be larger than
  the image in cases when the bounding box is on the edges of an image. [#121]


0.8.0 (2017-11-02)
------------------

- ``LabelMapperRange`` now returns ``LabelMapperRange._no_label`` when the key is
  not within any range. [#71]

- ``LabelMapperDict`` now returns ``LabelMapperDict._no_label`` when the key does
  not match. [#72]

- Replace ``domain`` with ``bounding_box``. [#74]

- Added a ``LabelMapper`` model where ``mapper`` is an instance of
  `~astropy.modeling.core.Model`. [#78]

- Evaluating a WCS with bounding box was moved to ``astropy.modeling``. [#86]

- RegionsSelector now handles the case when a label does not have a corresponding
  transform and returns RegionsSelector.undefined_transform_value. [#86]

- GWCS now deals with axes types which are neither celestial nor spectral as "unknown"
  and creates a transform equivalent to the FITS linear transform. [#92]

0.7 (2016-12-23)
----------------

New Features
^^^^^^^^^^^^
- Added ``wcs_from_fiducial`` function to wcstools. [#34]
- Added ``domain`` to the WCS object. [#36]
- Added ``grid_from_domain`` function. [#36]
- The WCS object can return now an `~astropy.coordinates.SkyCoord`
  or `~astropy.units.Quantity` object. This is triggered by a new
  parameter to the ``__call__`` method, ``output`` which takes values
  of "numericals" (default) or "numericals_plus".    [#64]

API_Changes
^^^^^^^^^^^
- Added ``atol`` argument to ``LabelMapperDict``, representing the absolute tolerance [#29]
- The ``CoordinateFrame.transform_to`` method was removed [#64]

Bug Fixes
^^^^^^^^^
- Fixed a bug in ``LabelMapperDict`` where a wrong index was used.[#29]
- Changed the order of the inputs when ``LabelMapperArray`` is evaluated as
  the inputs are supposed to be image coordinates. [#29]
- Renamed variables in read_wcs_from_header to match loop variable [#63]

0.5.1 (2016-02-01)
------------------

Bug Fixes
^^^^^^^^^

- Added ASDF requirement to setup. [#30]
- Import OrderedDict from collections, not from astropy. [#32]

0.5 (2015-12-28)
----------------

Initial release on PYPI.

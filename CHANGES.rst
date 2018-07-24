0.10.0 (Unreleased)
-------------------

New Features
^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^


0.9.0 (2017-05-23)
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

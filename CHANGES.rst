0.9.0 (Unreleased)
------------------

New Features
^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

- Fixed a bug in ``bounding_box`` definition when the WCs has only one axis. [#117]

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

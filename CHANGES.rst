0.6 (unreleased)
----------------

New Features
^^^^^^^^^^^^
- Added ``wcs_from_fiducial`` function to wcstools. [#34]
- Added ``domain`` to the WCS object. [#36]
- Added ``grid_from_domain`` function. [#36]

API_Changes
^^^^^^^^^^^
- Added ``atol`` argument to ``LabelMapperDict``, representing the absolute tolerance [#29]

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

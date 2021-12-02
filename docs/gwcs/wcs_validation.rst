.. _wcs_validation:

WCS validation
==============


The WCS is validated when an object is read in or written to a file.
However, this happens transparently to the end user and knowing
the details of the validation machinery is not necessary to use or
construct a WCS object.

GWCS uses the
`Advanced Scientific Data Format <https://asdf-standard.readthedocs.io/en/latest/>`_
(ASDF) to serialize and deserialize GWCS objects (including transformations
and frames) and to provide validation that the serialization is correct.
ASDF makes use of abstract data type definitions called ``schemas``.
The serialization and deserialization happens in classes, referred to as
``converters`` defined in ``gwcs.converters.*`` modules. Most of the schemas
available for the WCS object, coordinate frames and some WCS specific transforms
live in the
`asdf-wcs-schemas package <http://asdf-wcs-schemas.readthedocs.io/en/latest>`_.

Packages using GWCS may create their own transforms and schemas and register
them as an ``Asdf Extension``. If those are of general use, it is recommended
they be included in `asdf-astropy <https://github.com/astropy/asdf-astropy>`_.

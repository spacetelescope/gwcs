.. _wcs_validation:

WCS validation
==============


The WCS is validated when an object is read in or written to a file.
However, this happens transparently to the end user and knowing
the details of the validation machinery is not necessary to use or
construct a WCS object. 

GWCS uses the `Advanced Scientific Data Format <https://asdf-standard.readthedocs.io/en/latest/>`__ (ASDF)
to validate the transforms, coordinate frames and the overall WCS object structure.
ASDF makes use of abstract data type
definitions called ``schemas``. The serialization and deserialization happens in classes,
referred to as ``tags``. Most of the transform schemas live in the ``asdf-standard`` package while most of the transform tags live in ``astropy``. :ref:`gwcs-schemas` are available for the WCS object, coordinate frames and some WCS specific transforms. 

Packages using GWCS may create their own transforms and schemas and register them as an ``Asdf Extension``. If those are of general use, it is recommended they be included in astropy.



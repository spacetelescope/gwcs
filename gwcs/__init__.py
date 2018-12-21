# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
   GWCS - Generalized World Coordinate System
   ==========================================

Generalized World Coordinate System (GWCS) is an `Astropy`_ affiliated package providing tools for managing the World Coordinate System of astronomical data.

GWCS takes a general approach to the problem of expressing transformations between pixel and world coordinates. It supports a data model which includes the entire transformation pipeline from input coordinates (detector by default) to world coordinates. It is tightly integrated with `Astropy`_.

- Transforms are instances of ``astropy.Model``. They can be chained, joined or combined with arithmetic operators using the flexible framework of compound models in `astropy.modeling`_.
- Celestial coordinates are instances of ``astropy.SkyCoord`` and are transformed to other standard celestial frames using `astropy.coordinates`_.
- Time coordinates are represented by ``astropy.Time`` and can be further manipulated using the tools in `astropy.time`_
- Spectral coordinates are ``astropy.Quantity`` objects and can be converted to other units using the tools in `astropy.units`_.

For complete features and usage examples see the `documentation`_ site.

Note
----

GWCS support only Python 3.


Installation
------------

To install::

    pip install gwcs

To clone from github and install the master branch::

    git clone https://github.com/spacetelescope/gwcs.git
    cd gwcs
    python setup.py install


Contributing Code, Documentation, or Feedback
---------------------------------------------

We welcome feedback and contributions to the project. Contributions of
code, documentation, or general feedback are all appreciated. Please
follow the `contributing guidelines <CONTRIBUTING.md>`__ to submit an
issue or a pull request.

We strive to provide a welcoming community to all of our users by
abiding to the `Code of Conduct <CODE_OF_CONDUCT.md>`__.


.. _Astropy: http://www.astropy.org/

.. _astropy.time: http://docs.astropy.org/en/stable/time/
.. _astropy.modeling: http://docs.astropy.org/en/stable/modeling/
.. _astropy.units: http://docs.astropy.org/en/stable/units/
.. _astropy.coordinates: http://docs.astropy.org/en/stable/coordinates/
.. _documentation: http://gwcs.readthedocs.org/en/latest/

"""

import sys
if sys.version_info < (3, 5):
    raise ImportError("GWCS does not support Python 2.x, 3.0, 3.1, 3.2, 3.3 or 3.4."
                      "Beginning with GWCS 0.9, Python 3.5 and above is required.")


# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import * # noqa
# ----------------------------------------------------------------------------


# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .wcs import *   # noqa
    from .wcstools import *   # noqa
    from .coordinate_frames import *  # noqa
    from .selector import *   # noqa

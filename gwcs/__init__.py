# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

import sys
if sys.version_info < (3, 5):
    raise ImportError("GWCS does not support Python 2.x, 3.0, 3.1, 3.2, 3.3 or 3.4."
                      "Beginning with GWCS 0.9, Python 3.5 and above is required.")


# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------


# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .wcs import *
    from .wcstools import *
    from .coordinate_frames import *
    from .selector import *

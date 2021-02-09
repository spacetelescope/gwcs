GWCS - Generalized World Coordinate System
==========================================

.. image:: https://github.com/spacetelescope/gwcs/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/gwcs/actions
    :alt: CI Status
    
.. image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :target: https://docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/spacetelescope/gwcs/branch/master/graph/badge.svg?token=JtHal6Jbta
    :target: https://codecov.io/gh/spacetelescope/gwcs
    :alt: Code coverage

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
    
.. image:: https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat
    :target: http://www.stsci.edu
    :alt: Powered by STScI Badge

Generalized World Coordinate System (GWCS) is an `Astropy`_ affiliated package providing tools for managing the World Coordinate System of astronomical data.

GWCS takes a general approach to the problem of expressing transformations between pixel and world coordinates. It supports a data model which includes the entire transformation pipeline from input coordinates (detector by default) to world coordinates. It is tightly integrated with `Astropy`_.

- Transforms are instances of ``astropy.Model``. They can be chained, joined or combined with arithmetic operators using the flexible framework of compound models in `astropy.modeling`_.
- Celestial coordinates are instances of ``astropy.SkyCoord`` and are transformed to other standard celestial frames using `astropy.coordinates`_.
- Time coordinates are represented by ``astropy.Time`` and can be further manipulated using the tools in `astropy.time`_
- Spectral coordinates are ``astropy.Quantity`` objects and can be converted to other units using the tools in `astropy.units`_.

For complete features and usage examples see the `documentation`_ site.

Note
----
Beginning with version 0.9 GWCS requires Python 3.5 and above.


Installation
------------

To install::

    pip install gwcs  # Make sure pip >= 9.0.1 is used.

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


Citing GWCS
-----------

.. image:: https://zenodo.org/badge/29208937.svg
   :target: https://zenodo.org/badge/latestdoi/29208937

If you use GWCS, please cite the package via its Zenodo record.

.. _Astropy: http://www.astropy.org/

.. _astropy.time: http://docs.astropy.org/en/stable/time/
.. _astropy.modeling: http://docs.astropy.org/en/stable/modeling/
.. _astropy.units: http://docs.astropy.org/en/stable/units/
.. _astropy.coordinates: http://docs.astropy.org/en/stable/coordinates/
.. _documentation: http://gwcs.readthedocs.org/en/latest/

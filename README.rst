.. 
   GWCS - Generalized World Coordinate System
   ==========================================

.. raw:: html

   <h1 align="center">GWCS - Generalized World Coordinate System </h1>
   <p align="center">
     <a href='https://gwcs.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/gwcs/badge/?version=latest' alt='Documentation Status'></a>
     <a href="https://travis-ci.org/spacetelescope/gwcs"><img src="https://travis-ci.org/spacetelescope/gwcs.svg?branch=master" alt="Build Status"></a>
     <a href="https://coveralls.io/github/spacetelescope/gwcs?branch=master"><img src="https://coveralls.io/repos/github/spacetelescope/gwcs/badge.svg?branch=master" alt="Coverage Status"></a>
     <img src="https://img.shields.io/pypi/l/gwcs.svg" alt="license">
     <a href="http://www.stsci.edu"><img src="https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat" alt="stsci"></a>
     <a href="http://www.astropy.org/"><img src="http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat" alt="astropy"></a>
   </p>


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

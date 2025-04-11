.. _gwcs:

GWCS Documentation
==================

`GWCS <https://github.com/spacetelescope/gwcs>`__ is a package for managing
the World Coordinate System (WCS) of astronomical data.


Installation
------------

`gwcs <https://github.com/spacetelescope/gwcs>`__ requires:

- `numpy <http://www.numpy.org/>`__

- `astropy <http://www.astropy.org/>`__

- `asdf <https://asdf.readthedocs.io/en/latest/>`__

To install from source::

    git clone https://github.com/spacetelescope/gwcs.git
    cd gwcs
    python setup.py install

To install the latest release::

    pip install gwcs

The latest release of GWCS is also available as a conda package via `conda-forge <https://github.com/conda-forge/gwcs-feedstock>`__.


Concepts
--------

.. toctree::
  :maxdepth: 2

  gwcs/overview.rst

Getting Started
---------------

.. toctree::
  :maxdepth: 2

  gwcs/tutorials.rst


Using ``gwcs``
--------------

.. toctree::
  :maxdepth: 2

  gwcs/guides.rst


References
-----------

.. toctree::
  :maxdepth: 2

  gwcs/api.rst


See also
--------

- `The modeling  package in astropy
  <http://docs.astropy.org/en/stable/modeling/>`__

- `The coordinates package in astropy
  <http://docs.astropy.org/en/stable/coordinates/>`__

- `The Advanced Scientific Data Format (ASDF) standard
  <https://asdf-standard.readthedocs.io/>`__
  and its `Python implementation
  <https://asdf.readthedocs.io/>`__

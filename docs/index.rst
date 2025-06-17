.. _gwcs:

GWCS Documentation
==================

.. _installing-gwcs:

Getting GWCS
------------

To install the latest release:

.. code-block:: shell

    pip install gwcs

To install as a conda package from `conda-forge <https://github.com/conda-forge/gwcs-feedstock>`__:

.. code-block:: shell

    conda install -c conda-forge gwcs

To install the latest development version from source (not generally recommended
unless one needs a very new feature or bug fix):

.. code-block:: shell

  pip install git+https://github.com/spacetelescope/gwcs.git

If you wish to install directly from source with the ability to edit the source code:

.. code-block:: shell

  git clone https://github.com/spacetelescope/gwcs.git
  cd gwcs
  pip install -e .

Introductions
-------------

For Users
.........

.. toctree::
  :maxdepth: 2

  gwcs/user_introduction.rst

For Developers
..............

.. toctree::
  :maxdepth: 2

  gwcs/developer_introduction.rst


Other Examples
--------------

.. toctree::
  :maxdepth: 2

  gwcs/imaging_with_distortion.rst
  gwcs/ifu.rst



Using ``gwcs``
--------------

.. toctree::
  :maxdepth: 2

  gwcs/wcs_ape.rst
  gwcs/using_wcs.rst
  gwcs/wcstools.rst
  gwcs/pure_asdf.rst
  gwcs/wcs_validation.rst
  gwcs/points_to_wcs.rst
  gwcs/fits_analog.rst



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

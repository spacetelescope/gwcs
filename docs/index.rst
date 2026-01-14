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

To install the latest development version from source (not generally recommendedß
unless one needs a very new feature or bug fix):

.. code-block:: shell

  pip install git+https://github.com/spacetelescope/gwcs.git


Introductions
-------------

For Users
*********

.. toctree::
  :maxdepth: 3

  gwcs/user_introduction.rst


Constructing GWCS Models
************************

.. toctree::
  :maxdepth: 3

  gwcs/constructing_gwcs_models.rst
  gwcs/imaging_with_distortion.rst
  gwcs/ifu.rst


Advanced User Topics
--------------------

.. toctree::
  :maxdepth: 2

  gwcs/wcs_ape.rst
  gwcs/native_api.rst
  gwcs/native_vs_shared.rst
  gwcs/using_wcs.rst
  gwcs/wcstools.rst
  gwcs/pure_asdf.rst
  gwcs/wcs_validation.rst
  gwcs/points_to_wcs.rst
  gwcs/fits_analog.rst


Reference
---------

.. toctree::
  :maxdepth: 3

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

.. _ape14:

Common Interface for World Coordinate System - APE 14
=====================================================

To improve interoperability between packages, the Astropy Project and other interested parties have collaboratively defined a standardized application
programming interface (API) for world coordinate system objects to be used
in Python. This API is described in the Astropy Proposal for Enhancements (APE) 14:
`A shared Python interface for World Coordinate Systems
<https://doi.org/10.5281/zenodo.1188874>`_.

The base classes that define the low- (`~astropy.wcs.wcsapi.BaseLowLevelWCS`) and high- (:class:`~astropy.wcs.wcsapi.BaseHighLevelWCS`) level APIs are in astropy.
GWCS implements both APIs. Once a gWCS object is created the API methods will be available. It is recommended that applications use the ``Common API`` to
ensure transparent use of ``GWCS`` and ``FITSWCS`` objects.

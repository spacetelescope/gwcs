.. _legacy_vs_shared:

Differences in functionality between the Legacy and Shared interfaces
=====================================================================

The Shared API is meant to abstract out the details of the WCS implementation and
provide a uniform way to evaluate the transforms regardless of the underlying implementations in different WCS packages.
As such it is the "lowest common denominator" of the existing WCS libraries that support it,
currently ``astropy.wcs`` and ``gwcs``.

Because GWCS aims to be flexible and general, there is functionality that can
only be achieved with the Legacy Interface. Some of the differences are listed below:

- The Shared Interface methods call under the hood the Legacy methods with their default
  options. One of the consequences is that the transform evaluation happens always within the
  ``bounding_box`` of the WCS and values outside it are set to ``np.nan``.

- The Shared Interface methods work only on the total transform while GWCS
  can execute transforms between intermediate frames.

- Since the Shared Interface methods do not accept keyword arguments, if the ``numerical_inverse``
  method is invoked, it runs with default parameters.

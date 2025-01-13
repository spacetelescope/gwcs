WCS User Tools
==============

`~gwcs.wcstools.grid_from_bounding_box` is a function which returns a grid of input points based on the bounding_box of the WCS.

  >>> from gwcs.wcstools import grid_from_bounding_box
  >>> bounding_box = ((0, 4096), (0, 2048))
  >>> x, y = grid_from_bounding_box(bounding_box)
  >>> ra, dec = w(x, y)   # doctest: +SKIP
  
The `~gwcs.wcstools` module contains functions of general usability.

`~gwcs.wcstools.wcs_from_fiducial` is a function which given a fiducial in some coordinate system,
returns a WCS object.

  >>> from gwcs.wcstools import wcs_from_fiducial
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from astropy.modeling import models

To create a WCS from a pointing on the sky, as a minimum pass a sky coordinate and a projection to the function.
  >>> fiducial = coord.SkyCoord(5.46 * u.deg, -72.2 * u.deg, frame='icrs')
  >>> tan = models.Pix2Sky_TAN()

Any additional transforms are prepended to the projection and sky rotation.

  >>> trans = models.Shift(-2048) & models.Shift(-1024) | models.Scale(1.38*10**-5) & models.Scale(1.38*10**-5)
  >>> w = wcs_from_fiducial(fiducial, projection=tan, transform=trans)
  >>> w(2048, 1024)  # doctest: +FLOAT_CMP
      (5.46, -72.2)


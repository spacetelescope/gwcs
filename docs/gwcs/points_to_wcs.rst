
.. _wcs_from_points_example:

Fitting a WCS to input pixels & sky positions
=============================================

Suppose we have an image where we have centroid positions for a number of sources, and we have matched these 
positions to an external catalog to obtain (RA, Dec). If this data is missing or has inaccurate WCS information,
it is useful to fit or re-fit a GWCS object with this matched list of coordinate pairs to be able to transform
between pixel and sky. 

This example shows how to use the `~gwcs.wcstools.wcs_from_points` tool to fit a WCS to a matched set of 
pixel and sky positions.  Along with arrays of the (x,y) pixel position in the image and the matched sky coordinates,
the fiducial point for the projection must be supplied as a `~astropy.coordinates.SkyCoord` object. Additionally,
the projection type must be specified from the available projections in `~astropy.modeling.projections.projcode`.

Geometric distortion can also be fit to the input coordinates - the distortion type (2D polynomial, chebyshev, legendre) and 
the degree can be supplied to fit this component of the model.

The following example will show how to fit a WCS, including a 4th degree 2D polynomial, to a set of input pixel positions of 
sources in an image and their corresponding positions on the sky obtained from a catalog. 

Import the wcs_from_points function,

  >>> from gwcs.wcstools import wcs_from_points
	
along with some useful general imports.

  >>> from astropy.coordinates import SkyCoord
  >>> from astropy.io import ascii
  >>> import astropy.units as u
  >>> import numpy as np
  
A collection of 20 matched coordinate pairs in x, y, RA, and Dec stored in two arrays, will be used to fit the WCS information. The function requires tuples of arrays.

  >>> xy = (np.array([2810.156, 2810.156,  650.236, 1820.927, 3425.779, 2750.369,
  ...                 212.422, 1146.91 ,   27.055, 2100.888,  648.149,   22.212,
  ...                 2003.314,  727.098,  248.91 ,  409.998, 1986.931,  128.925,
  ...                 1106.654, 1502.67 ]),
  ...       np.array([1670.347, 1670.347,  360.325,  165.663,  900.922,  700.148,
  ...                 1416.235, 1372.364,  398.823,  580.316,  317.952,  733.984,
  ...                 339.024,  234.29 , 1241.608,  293.545, 1794.522, 1365.706,
  ...                 583.135,   25.306]))
  >>> radec = (np.array([246.75001315, 246.75001315, 246.72033646, 246.72303144,
  ...                    246.74164072, 246.73540614, 246.73379121, 246.73761455,
  ...	        	 246.7179495 , 246.73051123, 246.71970072, 246.7228646 ,
  ...			 246.72647213, 246.7188386 , 246.7314031 , 246.71821002,
  ...			 246.74785534, 246.73265223, 246.72579817, 246.71943263]),
  ...	       np.array([43.48690547,  43.48690547,  43.46792989,  43.48075238,
  ...		         43.49560501,  43.48903538,  43.46045875,  43.47030776,
  ...			 43.46132376,  43.48252763,  43.46802566,  43.46035331,
  ...			 43.48218262,  43.46908299,  43.46131665,  43.46560591,
  ...			 43.47791234,  43.45973025,  43.47208325,  43.47779988]))
  
  
We can now choose the reference point on the sky for the projection. This is passed in 
as a `~astropy.coordinates.SkyCoord` object so that information about the celestial frame and units is given as well.
The input world coordinates are passed in as unitless arrays, and so are assumed to be of the same unit and frame 
as the fiducial point. 
 
 >>> proj_point = SkyCoord(246.7368408, 43.480712949, frame = 'icrs', unit = (u.deg,u.deg))
 
We can now call the function that returns a GWCS object corresponding to the best fit parameters
that relate the input pixels and sky coordinates with a TAN projection centered at the reference point
we specified, with a distortion model (degree 4 polynomial). This function will return a GWCS object that 
can be used to transform between coordinate frames.
 
  >>> gwcs_obj = wcs_from_points(xy, radec, proj_point)

This GWCS object contains parameters for a TAN projection, rotation, scale, skew and a polynomial fit to x and y 
that represent the best-fit to the input coordinates. With WCS information associated with the data now, we can
easily work in both pixel and sky space, and transform between frames. 

The GWCS object, which by default when called executes for forward transformation,
can be used to convert coordinates from pixel to world.

  >>> gwcs_obj(36.235,642.215)    # doctest: +FLOAT_CMP
  (246.72158004206716, 43.46075091731673)
  
Or equivalently 
  >>> gwcs_obj.forward_transform(36.235,642.215)  # doctest: +FLOAT_CMP
  (246.72158004206716, 43.46075091731673)


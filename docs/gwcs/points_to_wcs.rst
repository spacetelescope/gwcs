<<<<<<< HEAD
.. _wcs_from_points_example:
=======
.. _user_api:
>>>>>>> 5ebcca0ea6ca16e38bb32fc8b0540b22d30591aa

Fitting a WCS to input pixels & sky positions
=============================================

<<<<<<< HEAD
Suppose we have an image where we have centroid positions for a number of sources, and we have matched these 
=======
Suppose we have an image where we have centroid positions for a handful of sources, and we have matched these 
>>>>>>> 5ebcca0ea6ca16e38bb32fc8b0540b22d30591aa
positions to an external catalog to obtain (RA, Dec). If this data is missing or has inaccurate WCS information,
it is useful to fit or re-fit a GWCS object with this matched list of coordinate pairs to be able to transform
between pixel and sky. 

<<<<<<< HEAD
This example shows how to use the `~gwcs.wcstools.wcs_from_points` tool to fit a WCS to a matched set of 
pixel and sky positions.  Along with arrays of the (x,y) pixel position in the image and the matched sky coordinates,
the fiducial point for the projection must be supplied as a `~astropy.coordinates.SkyCoord` object. Additionally,
=======
This example shows how to use the wcs_from_point tool to fit a WCS to a matched set of 
pixel and sky positions.  Along with arrays of the (x,y) pixel position in the image and the matched sky coordinates,
the fiducial point for the projection must be supplied as a SkyCoord object. Additionally,
>>>>>>> 5ebcca0ea6ca16e38bb32fc8b0540b22d30591aa
the projection type must be specified from the available projections in `~astropy.modeling.projections.projcode`.

Geometric distortion can also be fit to the input coordinates - the distortion type (2D polynomial, chebyshev, legendre) and 
the degree can be supplied to fit this component of the model.

The following example will show how to fit a WCS, including a 4th degree 2D polynomial, to a set of input pixel positions of 
sources in an image and their corresponding positions on the sky obtained from a catalog. 

<<<<<<< HEAD
Import the wcs_from_points function,
  >>> from gwcs.wcstools import wcs_from_points
	
along with some useful general imports.

  >>> from astropy.coordinates import SkyCoord
  >>> from astropy.io import ascii
  >>> import astropy.units as u
  >>> import numpy as np
  
A collection of 20 matched coordinate pairs in x, y, RA, and Dec stored in a csv file, will be used to fit the WCS information.
We can read this file in as an astropy table so the data is accessible. If you would like to try this example yourself, the file xy_sky_coords.csv 
is in the docs/gwcs/static_example_files subdirectory within the gwcs source code.

  >>> tab = ascii.read('gwcs/docs/gwcs/static_example_files/xy_sky_coords.csv')
  >>> print(tab)
	   x        y          ra          dec    
	-------- -------- ------------ -----------
	2810.156 1670.347 246.75001315 43.48690547
	2810.156 1670.347 246.75001315 43.48690547
	 650.236  360.325 246.72033646 43.46792989
	1820.927  165.663 246.72303144 43.48075238
	...
  
Unpack the table into arrays for each coordinate. 
  >>> x, y, ra, dec = tab['x'], tab['y'], tab['ra'], tab['dec']
  
The function requires tuples of arrays, so make tuples of (x,y) and (ra, dec) to pass to the function.  

  >>> xy = (x,y)
  >>> world_coordinates = (ra,dec)
  
We can now choose the reference point on the sky for the projection. This is passed in 
as a `~astropy.coordinates.SkyCoord` object so that information about the celestial frame and units is given as well.
The input world coordinates are passed in as unitless arrays, and so are assumed to be of the same unit and frame 
as the fiducial point. 
 
 >>> proj_point = SkyCoord(246.7368408, 43.480712949, frame = 'icrs', unit = (u.deg,u.deg))
 
By default, `~gwcs.wcstools.wcs_from_points` fits a degree 4 polynomial to the points to represent geometric distortion in the image. This 
behavior can be modified by setting the keyword arguments ``degree`` and ``polynomial_type``. Additionally, this function defaults 
to using the TAN projection from projections, so we do not need to set this explicitly in this case. If you are using a 
different projection type from projections, that must be passed to the function as one of the projections in `~astropy.modeling.projections.projcode`.
=======
.. doctest-skip::
   
  >>> from gwcs import wcs_from_point
  >>> import numpy as np
  >>> from astropy.coordinates import SkyCoord
  
  The matched set of input pixels and sky coordinates are passed in as tuples of arrays, matched by index. 
  The xy array is a tuple (x coordinates, y coordinates), and world_coordinates is (RA, Dec).
  
  >>> xy = (np.array(2810.156, 650.236, 1820.927, 3425.779, 2750.369), np.array(1670.347, 360.325, 165.663, 900.922, 700.148))
  >>> world_coordinates = (np.array([ 246.75001315,  246.72033646,  246.72303144,  246.74164072, 246.73540614]), np.array([ 43.48690547,  43.46792989,  43.48075238,  43.49560501,  43.48903538]))
 						   
We can now choose the reference point on the sky for the projection. This is passed in 
as a SkyCoord object so that information about the celestial frame and units is given as well.
 
 >>> proj_point = SkyCoord(246.7368408, 43.480712949, frame = 'icrs', unit = (u.deg,u.deg))
 
By default, wcs_from_point fits a degree 4 polynomial to the points to represent geometric distortion in the image. This 
behavior can be modified by setting the keyword arguments 'degree' and 'polynomial_type'. Additionally, this function defaults 
to using the TAN projection from projections, so we do not need to set this explicitley in this case. If you are using a 
different projection type from projections, that must be passed to the function. 
>>>>>>> 5ebcca0ea6ca16e38bb32fc8b0540b22d30591aa

We can now call the function that returns a GWCS object corresponding to the best fit parameters
that relate the input pixels and sky coordinates with a TAN projection centered at the reference point
we specified, with a distortion model (degree 4 polynomial). This function will return a GWCS object that 
can be used to transform between coordinate frames.
 
<<<<<<< HEAD
  >>> gwcs_obj = wcs_from_points(xy, world_coordinates, proj_point)

This GWCS object contains parameters for a TAN projection, rotation, scale, skew and a polynomial fit to x and y 
that represent the best-fit to the input coordinates. With WCS information associated with the data now, we can
easily work in both pixel and sky space, and transform between frames. 

The GWCS object, which by default when called executes for forward transformation,
can be used to convert coordinates from pixel to world.

  >>> gwcs_obj(36.235,642.215)
  (246.72158004206716, 43.46075091731673)
  
Or equivalently 
  >>> gwcs_obj.forward_transform(36.235,642.215)
  (246.72158004206716, 43.46075091731673) 
  
=======
 >>> wcs_from_points(xy, world_coordinates, proj_point)
 
	
>>>>>>> 5ebcca0ea6ca16e38bb32fc8b0540b22d30591aa

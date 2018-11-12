.. _user_api:

Fitting a WCS to input pixels & sky positions
=============================================

Suppose we have an image where we have centroid positions for a handful of sources, and we have matched these 
positions to an external catalog to obtain (RA, Dec). If this data is missing or has inaccurate WCS information,
it is useful to fit or re-fit a GWCS object with this matched list of coordinate pairs to be able to transform
between pixel and sky. 

This example shows how to use the wcs_from_point tool to fit a WCS to a matched set of 
pixel and sky positions.  Along with arrays of the (x,y) pixel position in the image and the matched sky coordinates,
the fiducial point for the projection must be supplied as a SkyCoord object. Additionally,
the projection type must be specified from the available projections in `~astropy.modeling.projections.projcode`.

Geometric distortion can also be fit to the input coordinates - the distortion type (2D polynomial, chebyshev, legendre) and 
the degree can be supplied to fit this component of the model.

The following example will show how to fit a WCS, including a 4th degree 2D polynomial, to a set of input pixel positions of 
sources in an image and their corresponding positions on the sky obtained from a catalog. 

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

We can now call the function that returns a GWCS object corresponding to the best fit parameters
that relate the input pixels and sky coordinates with a TAN projection centered at the reference point
we specified, with a distortion model (degree 4 polynomial). This function will return a GWCS object that 
can be used to transform between coordinate frames.
 
 >>> wcs_from_points(xy, world_coordinates, proj_point)
 
	

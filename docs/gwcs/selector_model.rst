An IFU Example
==============

``GWCS`` has a special `~gwcs.selector.SelectorModel` class which provides a mapping of transforms to
regions on the detector or to other quantities and the means to switch between or select a transform for a certain label. We show here how this can be used to describe the WCS of an IFU observation.

For this example we look at an IFU with 6 slits.
In general each slit has a WCS asosciated with it. The output coordinate frame for each slit is a composite frame with two frames, `~gwcs.coordinate_frames.CelestialFrame` and `~gwcs.coordinate_frames.SpectralFrame`. For each slit the WCS object transforms pixel coordinates to world coordinates (RA, DEC, lambda).
For the sake of brevity we assime the transform for each slit has been created. A detailed example of how to create a transform is in the ref: imaging_example.

In order to use the `~gwcs.selector.SelectorModel` we need a list of labels
and a mask. Labels are integers or strings (less efficient) which map to the location of slits in pixel space. Each label has a transform associated with it. The mask is an array with the size of the detector where each item in the array corresponds to a pixel on the detector and its value is the slice number (label) this pixel belongs to.

.. image:: ifu-regions.png

The image above shows the projection of the 6 slits on the detector and represents the mask used in the example. Pixels, labeled "0" do not belong to any slit.

Assuming the transforms from pixel to world coordinates for each slit are named "p2w_#", where "#" is the slit label, the pixel to world transform for the entire IFU can be created:

  >>> slit_labels = [1, 2, 3, 4, 5, 6]
  >>> slit_transforms = [p2w_1, p2w_2, p2w_3, p2w_4, p2w_5, p2w_6]
  >>> forward = gwcs.RegionsSelector(mask, labels=slit_labels, transforms=slit_transforms, undefined_transform_value=np.nan)

We have chosen in this example to set the world coordinate value for pixels which do not belong to any slit to NaN but any number is an acceptable value.

Next we need to create the output coordinate system - a `~gwcs.coordinate_frames.CompositeCoordinateFrame` which consistes of two frames: a `~gwcs.coordinate_frames.CelestialFrame` with an `~astropy.coordinates.ICRS` reference frame and  a `~gwcs.coordinate_frames.SpectralFrame` with a `~gwcs.coordinate_frames.spectral_builtin_frames.Wavelength` reference frame.

  >>> celestial = gwcs.CelestialFrame(coord.ICRS())
  >>> spec = gwcs.SpectralFrame(gwcs.spectral_builtin_frames.Wavelength(), unit=[u.micron], axes_names=['lambda'])
  >>> output_system = gwcs.CompositeFrame([celestial, spec])
  >>> print output_system.unit
  [Unit("deg"), Unit("deg"), Unit("micron")]


The WCS for the IFU observation is created by passing the input and output coordinate systems and the transform between them. To transform from pixel to world coordinates we simply call the WCS object. ``x`` and ``y`` below are coordinates in the detector pixel space.

  >>> wifu = gwcs.WCS(input_coordinate_system='detector', output_coordinate_system=output_system, forward_transform=forward)
  >>> x, y = 1, 2
  >>> result = wifu(x, y)
  >>> print result
  (<SkyCoord (ICRS): (ra, dec) in deg (5.63043056, -72.05454345)>,
  <Wavelength Coordinate (reference_position=BARYCENTER): (lam) in m (0.00000344)>)


Because the transform is an instance of `~gwcs.selector.RegionsSelectorModel`, we can
pass a ``region_id`` to the WCS function and perform the transform for a particular slice number.
For example, to transform coordinates in slice 4:

  >>> transform(x, y, 4)
  (<SkyCoord (ICRS): (ra, dec) in deg (5.63024693, -72.05452058)>,
  <Wavelength Coordinate (reference_position=BARYCENTER): (lam) in m (0.00000344)>)

Note that ``x``, ``y`` now are pixels in the coordinate system associated with the 4th slice,
not the entire detector.



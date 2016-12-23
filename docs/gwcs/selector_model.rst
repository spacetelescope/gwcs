Discontinuous WCS - An IFU Example
==================================

There are a couple of models in GWCS which help with managing discontinuous WCSs.
Given (x, y) pixel indices, `~gwcs.selector.LabelMapperArray` returns a label (int or str)
which uniquely identifies a location in a frame. `~gwcs.selector.RegionsSelector`
maps different transforms to different locations in the frame of `~gwcs.selector._LabelMapper`.

An example use case is an IFU observation. The locations on the detector where slices are
projected are labeled with integer numbers. Given an (x, y) pixel, the `~gwcs.selector.LabelMapperArray`
returns the label of the IFU slice which this pixel belongs to. Assuming each slice has its own WCS
transformation, `~gwcs.selector.RegionsSelector` takes as input an instance of `~gwcs.selector.LabelMapperArray`
and maps labels to transforms. A step by step example of constructing the WCS for an IFU with 6 slits follows.

  >>> import numpy as np
  >>> from astropy.modeling.models import Shift, Scale, Mapping
  >>> from astropy import coordinates as coord
  >>> from astropy import units as u
  >>> from gwcs import wcs, selector
  >>> from gwcs import coordinate_frames as cf

The output frame is common for all slits and is a composite frame with two subframes,
`~gwcs.coordinate_frames.CelestialFrame` and `~gwcs.coordinate_frames.SpectralFrame`.

  >>> sky_frame = cf.CelestialFrame(name='icrs', reference_frame=coord.ICRS(), axes_order=(0, 2))
  >>> spec_frame = cf.SpectralFrame(name='wave', unit=(u.micron,), axes_order=(1,), axes_names=('lambda',))
  >>> cframe = cf.CompositeFrame([sky_frame, spec_frame], name='world')

The input frame can be a string (default is 'detector') or a subclass of `~gwcs.coordinate_frames.CoordinateFrame`
in which allows additional functionality (see example below).

  >>> det = cf.Frame2D(name='detector')

All slits have the same input and output frames, however each slit has a different model transforming
from pixels to world coordinates (RA, lambda, dec). For the sake of brevity this example uses a simple
shift transform for each slit. Detailed examples of how to create more realistic transforms
are here (imaging) and here  the ref: spectral_example.

  >>> transforms = {}
  >>> for i in range(1, 7):
          transforms[i] = Mapping([0, 0, 1]) | Shift(i * 0.1) & Shift(i * 0.2) & Scale(i * 0.1)

One way to initialize `~gwcs.selector.LabelMapperArray` is to pass it the shape of the array and the vertices
of each slit on the detector {label: vertices} see :meth: `~gwcs.selector.LabelMapperArray.from_vertices`.
In this example the mask is an array with the size of the detector where each item in the array
corresponds to a pixel on the detector and its value is the slice number (label) this pixel
belongs to.

.. image:: ifu-regions.png

The image above shows the projection of the 6 slits on the detector. Pixels, with a label of 0 do
not belong to any slit. Assuming the array is stored in
`ASDF <https://asdf-standard.readthedocs.io/en/latest>`__ format, create the mask:

  >>> from asdf import AsdfFile
  >>> f = AsdfFile.open('mask.asdf')
  >>> data = f.tree['mask']
  >>> mask = selector.LabelMapperArray(data)

For more information on using the `ASDF standard <https://asdf-standard.readthedocs.io/en/latest/>`__ format
see `asdf <https://asdf.readthedocs.io/en/latest/>`__

Create the pixel to world transform for the entire IFU:

  >>> regions_transform = selector.RegionsSelector(inputs=['x','y'],
                                                   outputs=['ra', 'dec', 'lam'],
                                                   selector=transforms,
                                                   label_mapper=mask,
                                                   undefined_transform_value=np.nan)

The WCS object now can evaluate simultaneously the transforms of all slices

  >>> wifu = wcs.WCS(forward_transform=regions_transform, output_frame=cframe, input_frame=det)
  >>> x, y = mask.mapper.shape
  >>> x, y = np.mgrid[:x, :y]
  >>> r, d, l = wifu(x, y)

or of single slices.

  >>> wifu.forward_transform.set_input(4)(1, 2)
      (1.4, 1.8, 0.8)

The :meth:`~gwcs.selector.RegionsSelector.set_input` method returns the forward_transform for
a specific label.

The above commands return numerical values. The :meth: `~gwcs.coordinate_frames.CoordinateFrame.coordinates`
is functionally equivalent to the above commands bu returns coordinate objects:

  >>> wifu(10, 200)
      (10.3, 10.6, 60.00000000000001)
  >>> wifu.available_frames
      ['detector', 'world']

  >>> wifu.output_frame.coordinates(10, 200)
      (<SkyCoord (ICRS): (ra, dec) in deg
          (10.3, 60.0)>, <Quantity 10.6 micron>)

Frames provide additional information:

  >>> print(wifu.output_frame.axes_type)
      [u'SPATIAL', u'SPECTRAL', u'SPATIAL']
  >>> print(wifu.output_frame.axes_names)
      [u'ra', 'lambda', u'dec']
  >>> print(wifu.output_frame.unit)
      [Unit("deg"), Unit("micron"), Unit("deg")]

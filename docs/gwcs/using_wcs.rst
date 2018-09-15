Using the WCS object
====================

Let's expand the WCS created in :ref:`getting-started` by adding a distortion.

First create polynomial transform to represent distortion:

  >>> import numpy as np
  >>> from astropy.modeling.models import (Polynomial2D, Shift, Scale, Rotation2D,
  ...       Pix2Sky_TAN, RotateNative2Celestial, Mapping)
  >>> polyx = Polynomial2D(4)
  >>> polyx.parameters = np.arange(15) * .1
  >>> polyy = Polynomial2D(4)
  >>> polyy.parameters = np.arange(15) * .2
  >>> distortion = (Mapping((0, 1, 0, 1)) | polyx & polyy).rename("distortion")
  >>> det2sky = (Shift(-10.5) & Shift(-13.2) | Rotation2D(0.0023) | \
  ...            Scale(.01) & Scale(.04) | Pix2Sky_TAN() | \
  ...            RotateNative2Celestial(5.6, -72.05, 180)).rename("det2sky")

Create an intermediate frame. The distortion transforms positions on the
detector into this frame.

  >>> from astropy import units as u
  >>> from astropy import coordinates as coord
  >>> from gwcs import coordinate_frames as cf
  >>> focal_frame = cf.Frame2D(name="focal_frame", unit=(u.arcsec, u.arcsec))
  >>> detector_frame = cf.Frame2D(name="detector", unit=(u.pix, u.pix))
  >>> sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')

Create the WCS pipeline and initialize the WCS:

  >>> from gwcs import wcs
  >>> pipeline = [(detector_frame, distortion),
  ...             (focal_frame, det2sky),
  ...             (sky_frame, None)
  ...             ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
      From    Transform
  ----------- ----------
     detector distortion
  focal_frame    det2sky
         icrs       None

To see what frames are defined:

   >>> print(wcsobj.available_frames)
       ['detector', 'focal_frame', 'icrs']
   >>> wcsobj.input_frame
       <Frame2D(name="detector", unit=(Unit("pix"), Unit("pix")), axes_names=('x', 'y'),
       axes_order=(0, 1))>
   >>> wcsobj.output_frame
       <CelestialFrame(name="icrs", unit=(Unit("deg"), Unit("deg")), axes_names=('lon', 'lat'),
       axes_order=(0, 1), reference_frame=<ICRS Frame>)>

Because the ``output_frame`` is a `~gwcs.coordinate_frames.CoordinateFrame` object we can get
the result of the WCS transform as an `astropy.coordinates.SkyCoord` object and transform
them to other standard coordinate frames supported by `astropy.coordinates`.

  >>> skycoord = wcsobj(1, 2, with_units=True)
  >>> print(skycoord) # doctest: +SKIP
  <SkyCoord (ICRS): (ra, dec) in deg
      (6.62759055, -68.75445668)>
  >>> print(skycoord.transform_to("galactic")) # doctest: +SKIP
  <SkyCoord (Galactic): (l, b) in deg
      (306.31586901, -48.20968112)>

Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

  >>> distortion = wcsobj.get_transform('detector', 'focal_frame')
  >>> distortion(1, 2)    # doctest: +FLOAT_CMP
      (47.8, 95.60)

Transforms in the pipeline can be replaced by new transforms.

  >>> from astropy.modeling.models import Shift
  >>> new_transform = Shift(1) & Shift(1.5) | distortion
  >>> wcsobj.set_transform('detector', 'focal_frame', new_transform)
  >>> wcsobj(1, 2)
      (10.338562883899195, -42.331828785194055)

A transform can be inserted before or after a frame in the pipeline.

  >>> from astropy.modeling.models import Scale
  >>> scale = Scale(2) & Scale(1)
  >>> wcsobj.insert_transform('icrs', scale, after=False)
  >>> wcsobj(1, 2)
      (20.67712576779839, -42.331828785194055)

The WCS object has an attribute ``domain`` which describes the range of
acceptable values for each input axis.

  >>> wcsobj.bounding_box = ((0, 2048), (0, 1000))

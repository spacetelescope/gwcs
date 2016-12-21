Using the WCS object
====================

Let's expand the WCS created in :ref:`getting-started` by adding a distortion.

First create polynomial transform to represent distortion:

  >>> polyx = Polynomial2D(4)
  >>> polyx.parameters = np.random.randn(15)
  >>> polyy = Polynomial2D(4)
  >>> polyy.parameters = np.random.randn(15)
  >>> distortion = (Mapping((0, 1, 0, 1)) | polyx & polyy).rename("distortion")

Create an intermediate frame. The distortion transforms positions on the
detector into this frame.

  >>> focal_frame = cf.Frame2D(name="focal_frame", unit=(u.arcsec, u.arcsec))

Create the WCS pipeline and initialize the WCS:

  >>> pipeline = [(detector_frame, distortion),
                  (focal_frame, det2sky),
                  (sky_frame, None)
                  ]
  >>> wcsobj = wcs.WCS(pipeline)
  >>> print(wcsobj)
      From        Transform
      ----------- ----------
      detector     distortion
      focal_frame  focal2sky
      icrs         None

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

  >>> skycoord = wcsobj(1, 2, output="numericals_plus")
  >>> print(skycoord)
      <SkyCoord (ICRS): (ra, dec) in deg
          ( 4.97563356, -72.54530634)>
  >>> print(skycoord.transform_to("galactic"))
      <SkyCoord (Galactic): (l, b) in deg
          ( 306.23201951, -44.38032023)>

Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

  >>> distortion = wcsobj.get_transform('detector', 'focal')
  >>> distortion(1, 2)
      (4.807433286098964, 4.924746607074259)

Transforms in the pipeline can be replaced by new transforms.

  >>> new_transform = Shift(1) & Shift(1.5) | distortion
  >>> wcsobj.set_transform('detector', 'focal_frame', new_transform)
  >>> wcsobj(1, 2)
      (7.641677379945592, -71.18890415491595)

A transform can be inserted before or after a frame in the pipeline.

  >>> scale = Scale(2) & Scale(1)
  >>> wcsobj.insert_transform('icrs', scale, after=False)
  >>> wcsobj(1, 2)
      (15.283354759891184, -71.18890415491595)

The WCS object has an attribute ``domain`` which describes the range of
acceptable values for each input axis.

  >>> wcsobj.domain = [{'lower': 0, 'upper': 2048, 'includes_lower': True, 'includes_upper': False},
                       {'lower': 0, 'upper': 1000, 'includes_lower': True, 'includes_upper': False}]

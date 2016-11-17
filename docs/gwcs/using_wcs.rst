Using the WCS object
====================

Let's use the WCS object created in `Getting Started`_ in order to show the interface.

To see what frames are defined:

   >>> print(wcsobj.available_frames)
       ['detector', 'focal', 'icrs']

Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

  >>> distortion = wcsobj.get_transform('detector', 'focal')
  >>> distortion(1, 2)
      (13.4, 0.)

Transforms in the pipeline can be replaced by new transforms.

  >>> new_transform = Shift(1) & Shift(1.5) | distortion
  >>> wcsobj.set_transform('detector', 'focal', new_transform)
  >>> wcsobj(1, 2)
      (5.257230028926096, -72.53171157138964)

A transform can be inserted before or after a frame in the pipeline.

  >>> scale = Scale(2) & Scale(1)
  >>> w.insert_transform('icrs', scale, after=False)
  >>> w(1, 2)
      (10.514460057852192, -72.53171157138964)


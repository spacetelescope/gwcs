.. _user_api:

Using the WCS object
====================
   
This section uses the ``imaging_wcs.asdf`` created in :ref:`imaging_example`
to read in a WCS object and demo its methods.

.. doctest-skip::
   
  >>> import asdf
  >>> asdf_file = asdf.open("imaging_wcs.asdf")
  >>> wcsobj = asdf_file.tree["wcs"]
  >>> print(wcsobj)    # doctest: +SKIP
           From          Transform    
  ----------------- ----------------
           detector       distortion
  undistorted_frame linear_transform
               icrs             None

To see what frames are defined:

.. doctest-skip::

   >>> print(wcsobj.available_frames)
       ['detector', 'undistorted_frame', 'icrs']
   >>> wcsobj.input_frame
        <Frame2D(name="detector", unit=(Unit("pix"), Unit("pix")), axes_names=('x', 'y'), axes_order=(0, 1))>
    >>> wcsobj.output_frame
        <CelestialFrame(name="icrs", unit=(Unit("deg"), Unit("deg")), axes_names=('lon', 'lat'), axes_order=(0, 1), reference_frame=<ICRS Frame>)>

Because the ``output_frame`` is a `~gwcs.coordinate_frames.CoordinateFrame` object we can get
the result of the WCS transform as an `~astropy.coordinates.SkyCoord` object and transform
them to other standard coordinate frames supported by `astropy.coordinates`.

.. doctest-skip::
   
  >>> skycoord = wcsobj(1, 2, with_units=True)
  >>> print(skycoord)
  <SkyCoord (ICRS): (ra, dec) in deg
    (5.52886119, -72.05285219)>
  >>> print(skycoord.transform_to("galactic"))
  <SkyCoord (Galactic): (l, b) in deg
    (306.11346489, -44.89382103)>

The WCS object has an attribute :attr:`~gwcs.WCS.bounding_box`
(default value of ``None``) which describes the range of
acceptable values for each input axis.

.. doctest-skip::
   
  >>> wcsobj.bounding_box = ((0, 2048), (0, 1000))
  >>> wcsobj((2,3), (1020, 980))
      array([nan, 133.48248429]), array([nan, -11.24021056])
      
The WCS object accepts a boolean flag called ``with_bounding_box`` with default value of
``True``. Output values which are outside the ``bounding_box`` are set to ``NaN``.
There are cases when this is not desirable and ``with_bounding_box=False`` should be passes.

Calling the :meth:`~gwcs.WCS.footprint` returns the footprint on the sky.

.. doctest-skip::
   
   >>> wcsobj.footprint()
   
Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

.. doctest-skip::

   
  >>> dist = wcsobj.get_transform('detector', 'undistorted_frame')
  >>> dist(1, 2)    # doctest: +FLOAT_CMP
      (47.8, 95.60)

Transforms in the pipeline can be replaced by new transforms.

.. doctest-skip::
   
  >>> new_transform = models.Shift(1) & models.Shift(1.5) | distortion
  >>> wcsobj.set_transform('detector', 'focal_frame', new_transform)
  >>> wcsobj(1, 2)         # doctest: +FLOAT_CMP
      (5.5583005430002785, -72.06028278184611)


A transform can be inserted before or after a frame in the pipeline.

.. doctest-skip::
   
  >>> scale = models.Scale(2) & models.Scale(1)
  >>> wcsobj.insert_transform('icrs', scale, after=False)
  >>> wcsobj(1, 2)          # doctest: +FLOAT_CMP
      (11.116601086000557, -72.06028278184611)


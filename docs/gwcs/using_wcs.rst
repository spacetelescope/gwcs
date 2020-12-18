.. _using_wcs_examples:

Using the WCS object
====================

This section uses the ``imaging_wcs_wdist.asdf`` created in :ref:`imaging_example`
to read in a WCS object and demo its methods.

.. doctest-skip::

  >>> import asdf
  >>> asdf_file = asdf.open("imaging_wcs_wdist.asdf")
  >>> wcsobj = asdf_file.tree["wcs"]
  >>> print(wcsobj)    # doctest: +SKIP
           From          Transform
  ----------------- ----------------
           detector       distortion
  undistorted_frame linear_transform
               icrs             None


Inspecting Available Coordinate Frames
--------------------------------------

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
      (5.50090023, -72.04553535)>
  >>> print(skycoord.transform_to("galactic"))
  <SkyCoord (Galactic): (l, b) in deg
      (306.12713109, -44.8996588)>

Using Bounding Box
------------------

The WCS object has an attribute :attr:`~gwcs.WCS.bounding_box`
(default value of ``None``) which describes the range of
acceptable values for each input axis.

.. doctest-skip::

  >>> wcsobj.bounding_box = ((0, 2048), (0, 1000))
  >>> wcsobj((2,3), (1020, 980))
  [array([       nan, 5.54527989]), array([         nan, -72.06454341])]

The WCS object accepts a boolean flag called ``with_bounding_box`` with default value of
``True``. Output values which are outside the ``bounding_box`` are set to ``NaN``.
There are cases when this is not desirable and ``with_bounding_box=False`` should be passes.

Calling the :meth:`~gwcs.WCS.footprint` returns the footprint on the sky.

.. doctest-skip::

   >>> wcsobj.footprint()


Manipulating Transforms
-----------------------

Some methods allow managing the transforms in a more detailed manner.

Transforms between frames can be retrieved and evaluated separately.

.. doctest-skip::

  >>> dist = wcsobj.get_transform('detector', 'undistorted_frame')
  >>> dist(1, 2)    # doctest: +FLOAT_CMP
  (-292.4150238489997, -616.8680129899999)

Transforms in the pipeline can be replaced by new transforms.

.. doctest-skip::

  >>> new_transform = models.Shift(1) & models.Shift(1.5) | distortion
  >>> wcsobj.set_transform('detector', 'undistorted_frame', new_transform)
  >>> wcsobj(1, 2)         # doctest: +FLOAT_CMP
  (5.501064280097802, -72.04557376712566)

A transform can be inserted before or after a frame in the pipeline.

.. doctest-skip::

  >>> scale = models.Scale(2) & models.Scale(1)
  >>> wcsobj.insert_transform('icrs', scale, after=False)
  >>> wcsobj(1, 2)          # doctest: +FLOAT_CMP
  (11.002128560195604, -72.04557376712566)


Inverse Transformations
-----------------------

Often, it is useful to be able to compute inverse transformation that converts
coordinates from the output frame back to the coordinates in the input frame.

In this section, for illustration purpose, we will be using the same 2D imaging
WCS from ``imaging_wcs_wdist.asdf`` created in :ref:`imaging_example` whose
forward transformation converts image coordinates to world coordinates and
inverse transformation converts world coordinates back to image coordinates.

.. doctest-skip::

  >>> wcsobj = asdf.open(get_pkg_data_filename('imaging_wcs_wdist.asdf')).tree['wcs']

The most general method available for computing inverse coordinate
transformation is the `WCS.invert() <gwcs.wcs.WCS.invert>`
method. This method uses automatic or user-supplied analytical inverses whenever
available to convert coordinates from the output frame to the input frame.
When analytical inverse is not available as is the case for the ``wcsobj`` above,
a numerical solution will be attempted using
`WCS.numerical_inverse() <gwcs.wcs.WCS.numerical_inverse>`.

Default parameters used by `WCS.numerical_inverse() <gwcs.wcs.WCS.numerical_inverse>`
or `WCS.invert() <gwcs.wcs.WCS.invert>` methods should be acceptable in
most situations:

.. doctest-skip::

  >>> world = wcsobj(350, 200)
  >>> print(wcsobj.invert(*world))  # convert a single point
  (349.9999994163172, 200.00000017679295)
  >>> world = wcsobj([2, 350, -5000], [2, 200, 6000])
  >>> print(wcsobj.invert(*world))  # convert multiple points at once
  (array([ 2.00000000e+00,  3.49999999e+02, -5.00000000e+03]), array([1.99999972e+00, 2.00000002e+02, 6.00000000e+03])

By default, parameter ``quiet`` is set to `True` in `WCS.numerical_inverse() <gwcs.wcs.WCS.numerical_inverse>`
and so it will return results "as is" without warning us about possible loss
of accuracy or about divergence of the iterative process.

In order to catch these kind of errors that can occur during numerical
inversion, we need to turn off ``quiet`` mode and be prepared to catch
`gwcs.wcs.WCS.NoConvergence` exceptions. In the next example, let's also add a
point far away from the image for which numerical inverse fails.

.. doctest-skip::

  >>> from gwcs import NoConvergence
  >>> world = wcsobj([-85000, 2, 350, 3333, -5000], [-55000, 2, 200, 1111, 6000],
  ...                with_bounding_box=False)
  >>> try:
  ...     x, y = wcsobj.invert(*world, quiet=False, maxiter=40,
  ...                          detect_divergence=True, with_bounding_box=False)
  ... except NoConvergence as e:
  ...     print(f"Indices of diverging points: {e.divergent}")
  ...     print(f"Indices of poorly converging points: {e.slow_conv}")
  ...     print(f"Best solution:\n{e.best_solution}")
  ...     print(f"Achieved accuracy:\n{e.accuracy}")
  Indices of diverging points: [0]
  Indices of poorly converging points: [4]
  Best solution:
  [[ 1.38600585e+11  6.77595594e+11]
   [ 2.00000000e+00  1.99999972e+00]
   [ 3.49999999e+02  2.00000002e+02]
   [ 3.33300000e+03  1.11100000e+03]
   [-4.99999985e+03  5.99999985e+03]]
  Achieved accuracy:
  [[8.56497375e+02 5.09216089e+03]
   [6.57962988e-06 3.70445289e-07]
   [5.31656943e-06 2.72052603e-10]
   [6.81557583e-06 1.06560533e-06]
   [3.96365344e-04 6.41822468e-05]]

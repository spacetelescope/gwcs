.. _legacy_api:

Legacy Interface
================

The Native Interface is the original GWCS interface that allows complete access to GWCS functionality.

There are three main methods performing transformations and additional ones to help
with WCS introspection.

``forward()`` method evaluates the WCS transformation from the input frame (first frame in the transform) to the output coordinate frame. For convenience, the same can be achieved by invoking the WCS object as a function.

.. doctest-skip::

  >>> from gwcs import examples
  >>> wcsobj = examples.gwcs_3d_spatial_wave()
  >>> result = wcsobj(1, 1, 1)
  >>> print(result)
  (2.0, 3.0, 2.0)


The ``invert`` method evaluates the backward transform.

.. doctest-skip::

  >>> wcsobj.invert(*result)
  (1.0, 1.0, 1.0)

GWCS keeps track of units defined in the transforms, the inputs (if quantities) and
the coordinate frames. The type of the result matches the type of the inputs. If the input
is Quantities the result is quantities, even if the transforms do not support units.
In this case the units defined in the coordinate frames are used.
The Legacy API does not work with High Level Objects. These are supported only by the
High Level API.

It is possible to evaluate any transform between two intermediate frames using the ``transform`` method.

.. doctest-skip::

  >>> from astropy import units as u
  >>> from gwcs import examples
  >>> wcsobj = examples.gwcs_with_pipeline_celestial()
  >>> print(wcsobj.available_frames)
  ['input', 'celestial', 'output']
  >>> wcsobj.celestial
  <CelestialFrame(name="celestial", unit=(Unit("arcsec"), Unit("deg")), axes_names=('lon', 'lat'),
  ... axes_order=(0, 1), reference_frame=<ICRS Frame>)>
  >>> print(wcsobj.transform("celestial", "input", 60*u.arcsec, 60*u.deg))
  (<Quantity 3. pix>, <Quantity 4. pix>)


In this example ``from_frame`` comes after ``to_frame`` in the general WCS pipeline, essentially
evaluating the inverse of the transform between ``input`` and ``celestial``. Also, typing the
name of any frame in the pipeline shows some of the attributes of the frame.

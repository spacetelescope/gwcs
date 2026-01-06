.. _legacy_api:

Legacy Interface
================

The Legacy Interface is the original GWCS interface, before the Shared API was defined.

There are three main methods performing transformations and additional ones to help
with WCS introspection.

To evaluate the ``forward`` transform the WCS object can be called as a function.

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

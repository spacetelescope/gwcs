

.. _http://stsci.edu/schemas/gwcs/frame-1.0.0:

frame-1.0.0: The base class of all coordinate frames.
=====================================================

:soft:`Type:` object.

The base class of all coordinate frames.


These objects are designed to be nested in arbitrary ways to build up
transformation pipelines out of a number of low-level pieces.


:category:`Properties:`



  .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/name:

  :entry:`name`

  :soft:`Type:` string. Required.

  

  A user-friendly name for the frame.
  



  .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_order:

  :entry:`axes_order`

  :soft:`Type:` array :soft:`of` ( integer ).

  

  The order of the axes.
  

  :category:`Items:`



    .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_order/items:

    :soft:`Type:` integer.

    

    



  .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_names:

  :entry:`axes_names`

  :soft:`Type:` array :soft:`of` ( string :soft:`or` null ).

  

  The name of each axis in this frame.
  

  :category:`Items:`



    .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_names/items:

    :soft:`Type:` string :soft:`or` null.

    

    

    :category:`Any of:`



      .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_names/items/anyOf/0:

      :entry:`—`

      :soft:`Type:` string.

      

      



      .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/axes_names/items/anyOf/1:

      :entry:`—`

      :soft:`Type:` null.

      

      



  .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/reference_frame:

  :entry:`reference_frame`

  :soft:`Type:` :doc:`baseframe-1.0.0 <tag:astropy.org:astropy/coordinates/frames/baseframe-1.0.0>`.

  

  The reference frame.
  



  .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/unit:

  :entry:`unit`

  :soft:`Type:` array :soft:`of` ( :doc:`unit-1.0.0 <tag:stsci.edu:asdf/unit/unit-1.0.0>` ).

  

  Units for each axis.
  

  :category:`Items:`



    .. _http://stsci.edu/schemas/gwcs/frame-1.0.0/properties/unit/items:

    :soft:`Type:` :doc:`unit-1.0.0 <tag:stsci.edu:asdf/unit/unit-1.0.0>`.

    

    

:category:`Examples:`

A celestial frame in the ICRS reference frame.
::

  !<tag:stsci.edu:gwcs/celestial_frame-1.0.0>
    axes_names: [lon, lat]
    name: CelestialFrame
    reference_frame: !<tag:astropy.org:astropy/coordinates/frames/icrs-1.1.0>
      frame_attributes: {}
    unit: [!unit/unit-1.0.0 deg, !unit/unit-1.0.0 deg]
  

A pixel frame in three dimensions
::

  !<tag:stsci.edu:gwcs/frame-1.0.0>
    axes_names: [raster position, slit position, wavelength]
    axes_order: [0, 1, 2]
    axes_type: [SPATIAL, SPATIAL, SPECTRAL]
    name: pixel
    naxes: 3
    unit: [!unit/unit-1.0.0 pixel, !unit/unit-1.0.0 pixel, !unit/unit-1.0.0 pixel]
  

.. only:: html

   :download:`Original schema in YAML <frame-1.0.0.yaml>`



.. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0:

frame2d-1.0.0: Represents a 2D frame.
=====================================

:soft:`Type:` object :soft:`and` :doc:`frame-1.0.0 <frame-1.0.0>`.

Represents a 2D frame.




:category:`All of:`



  .. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0/allOf/0:

  :entry:`0`

  :soft:`Type:` object.

  

  

  :category:`Properties:`



    .. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0/allOf/0/properties/axes_names:

    :entry:`axes_names`

    :soft:`Type:` any.

    

    



    .. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0/allOf/0/properties/axes_order:

    :entry:`axes_order`

    :soft:`Type:` any.

    

    



    .. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0/allOf/0/properties/unit:

    :entry:`unit`

    :soft:`Type:` any.

    

    



  .. _http://stsci.edu/schemas/gwcs/frame2d-1.0.0/allOf/1:

  :entry:`1`

  :soft:`Type:` :doc:`frame-1.0.0 <frame-1.0.0>`.

  

  

:category:`Examples:`

A two dimensional spatial frame
::

  !<tag:stsci.edu:gwcs/frame2d-1.0.0>
    axes_names: [lon, lat]
    name: Frame2D
    unit: [!unit/unit-1.0.0 pixel, !unit/unit-1.0.0 pixel]
  


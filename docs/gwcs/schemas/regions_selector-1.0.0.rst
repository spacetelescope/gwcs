

.. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0:

regions_selector-1.0.0: Represents a discontinuous transform.
=============================================================

:soft:`Type:` `transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>` :soft:`and` object.

Represents a discontinuous transform.


Maps regions to transgorms and evaluates the transforms with the corresponding inputs.


:category:`All of:`



  .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/0:

  :entry:`0`

  :soft:`Type:` `transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

  

  



  .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1:

  :entry:`1`

  :soft:`Type:` object.

  

  

  :category:`Properties:`



    .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/label_mapper:

    :entry:`label_mapper`

    :soft:`Type:` :doc:`label_mapper-1.0.0 <./label_mapper-1.0.0>`. Required.

    

    An instance of
    [label_mapper-1.1.0](ref:http://stsci.edu/schemas/gwcs/label_mapper-1.0.0)
    



    .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/inputs:

    :entry:`inputs`

    :soft:`Type:` array :soft:`of` ( string ). Required.

    

    Names of inputs.
    

    :category:`Items:`



      .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/inputs/items:

      :soft:`Type:` string.

      

      



    .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/outputs:

    :entry:`outputs`

    :soft:`Type:` array :soft:`of` ( string ). Required.

    

    Names of outputs.
    

    :category:`Items:`



      .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/outputs/items:

      :soft:`Type:` string.

      

      



    .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/selector:

    :entry:`selector`

    :soft:`Type:` object. Required.

    

    A mapping of regions to trransforms.
    

    :category:`Properties:`



      .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/selector/properties/labels:

      :entry:`labels`

      :soft:`Type:` array :soft:`of` ( integer or string ).

      

      An array of unique region labels.
      

      :category:`Items:`



        .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/selector/properties/labels/items:

        :soft:`Type:` integer or string.

        

        



      .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/selector/properties/transforms:

      :entry:`transforms`

      :soft:`Type:` array :soft:`of` ( `transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>` ).

      

      A transform for each region. The order should match the order of labels.
      

      :category:`Items:`



        .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/selector/properties/transforms/items:

        :soft:`Type:` `transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

        

        



    .. _http://stsci.edu/schemas/gwcs/regions_selector-1.0.0/allOf/1/properties/undefined_transform_value:

    :entry:`undefined_transform_value`

    :soft:`Type:` number.

    

    Value to be returned if there's no transform defined for the inputs.
    

:category:`Examples:`

Create a regions_selector schema for 2 regions, labeled "1" and "2".::

  !<tag:stsci.edu:gwcs/regions_selector-1.0.0>
    inputs: [x, y]
    label_mapper: !<tag:stsci.edu:gwcs/label_mapper-1.0.0>
      mapper: !core/ndarray-1.0.0
        datatype: int8
        data:
        - [0, 1, 1, 0, 2, 0]
        - [0, 1, 1, 0, 2, 0]
        - [0, 1, 1, 0, 2, 0]
        - [0, 1, 1, 0, 2, 0]
        - [0, 1, 1, 0, 2, 0]
        datatype: int64
        shape: [5, 6]
    no_label: 0
    outputs: [ra, dec, lam]
    selector: !!omap
    - !!omap
      labels: [1, 2]
    - !!omap
      transforms:
      - !transform/compose-1.1.0
        forward:
        - !transform/remap_axes-1.1.0
          mapping: [0, 1, 1]
        - !transform/concatenate-1.1.0
          forward:
          - !transform/concatenate-1.1.0
            forward:
            - !transform/shift-1.1.0 {offset: 1.0}
            - !transform/shift-1.1.0 {offset: 2.0}
          - !transform/shift-1.1.0 {offset: 3.0}
      - !transform/compose-1.1.0
        forward:
        - !transform/remap_axes-1.1.0
          mapping: [0, 1, 1]
        - !transform/concatenate-1.1.0
          forward:
          - !transform/concatenate-1.1.0
            forward:
            - !transform/scale-1.1.0 {factor: 2.0}
            - !transform/scale-1.1.0 {factor: 3.0}
          - !transform/scale-1.1.0 {factor: 3.0}
    undefined_transform_value: .nan
  


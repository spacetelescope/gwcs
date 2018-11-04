

.. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0:

label_mapper-1.0.0: Represents a mapping from a coordinate value to a label.
============================================================================

:soft:`Type:` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>` :soft:`and` object.

Represents a mapping from a coordinate value to a label.


A label mapper instance maps inputs to a label.  It is used together
with
[regions_selector](ref:http://stsci.edu/schemas/gwcs/regions_selector-1.0.0). The
[label_mapper](ref:http://stsci.edu/schemas/gwcs/label_mapper-1.0.0)
returns the label corresponding to given inputs. The
[regions_selector](ref:http://stsci.edu/schemas/gwcsregions_selector-1.0.0)
returns the transform corresponding to this label. This maps inputs
(e.g. pixels on a detector) to transforms uniquely.


:category:`All of:`



  .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/0:

  :entry:`0`

  :soft:`Type:` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

  

  



  .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1:

  :entry:`1`

  :soft:`Type:` object.

  

  

  :category:`Properties:`



    .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper:

    :entry:`mapper`

    :soft:`Type:` :doc:`ndarray-1.0.0 <tag:stsci.edu:asdf/core/ndarray-1.0.0>` :soft:`or` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>` :soft:`or` object. Required.

    

    A mapping of inputs to labels.
    In the general case this is a `astropy.modeling.core.Model`.
    
    It could be a numpy array with the shape of the detector/observation.
    Pixel values are of type integer or string and represent
    region labels. Pixels which are not within any region have value ``no_label``.
    
    It could be a dictionary which maps tuples to labels or floating point numbers to labels.
    

    :category:`Any of:`



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/0:

      :entry:`—`

      :soft:`Type:` :doc:`ndarray-1.0.0 <tag:stsci.edu:asdf/core/ndarray-1.0.0>`.

      

      



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/1:

      :entry:`—`

      :soft:`Type:` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

      

      



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2:

      :entry:`—`

      :soft:`Type:` object.

      

      

      :category:`Properties:`



        .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/labels:

        :entry:`labels`

        :soft:`Type:` array :soft:`of` ( number :soft:`or` array :soft:`of` ( number ) ).

        

        

        :category:`Items:`



          .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/labels/items:

          :soft:`Type:` number :soft:`or` array :soft:`of` ( number ).

          

          

          :category:`Any of:`



            .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/labels/items/anyOf/0:

            :entry:`—`

            :soft:`Type:` number.

            

            



            .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/labels/items/anyOf/1:

            :entry:`—`

            :soft:`Type:` array :soft:`of` ( number ).

            

            

            :category:`Items:`



              .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/labels/items/anyOf/1/items:

              :soft:`Type:` number.

              

              



        .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/models:

        :entry:`models`

        :soft:`Type:` array :soft:`of` ( :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>` ).

        

        

        :category:`Items:`



          .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/mapper/anyOf/2/properties/models/items:

          :soft:`Type:` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

          

          



    .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/inputs:

    :entry:`inputs`

    :soft:`Type:` array :soft:`of` ( string ).

    

    Names of inputs.
    

    :category:`Items:`



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/inputs/items:

      :soft:`Type:` string.

      

      



    .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/inputs_mapping:

    :entry:`inputs_mapping`

    :soft:`Type:` :doc:`transform-1.1.0 <tag:stsci.edu:asdf/transform/transform-1.1.0>`.

    

    [mapping](ref:http://stsci.edu/schemas/asdf/transform/remap-axes-1.1.0)
    



    .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/atol:

    :entry:`atol`

    :soft:`Type:` number.

    

    absolute tolerance to compare keys in mapper.
    



    .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/no_label:

    :entry:`no_label`

    :soft:`Type:` number :soft:`or` string.

    

    Fill in value for missing output.
    

    :category:`Any of:`



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/no_label/anyOf/0:

      :entry:`—`

      :soft:`Type:` number.

      

      



      .. _http://stsci.edu/schemas/gwcs/label_mapper-1.0.0/allOf/1/properties/no_label/anyOf/1:

      :entry:`—`

      :soft:`Type:` string.

      

      

:category:`Examples:`

Map array indices are to labels.::

  !<tag:stsci.edu:gwcs/label_mapper-1.0.0>
    mapper: !core/ndarray-1.0.0
      data:
      - [1, 0, 2]
      - [1, 0, 2]
      - [1, 0, 2]
      datatype: int64
      shape: [3, 3]
      no_label: 0
  

Map numbers dictionary to transforms which return labels.::

  !<tag:stsci.edu:gwcs/label_mapper-1.0.0>
    atol: 1.0e-08
    inputs: [x, y]
    inputs_mapping: !transform/remap_axes-1.1.0
        mapping: [0]
        n_inputs: 2
    mapper: !!omap
      - !!omap
        labels: [-1.67833272, -1.9580548, -1.118888]
      - !!omap
        models:
        - !transform/shift-1.1.0 {offset: 6.0}
        - !transform/shift-1.1.0 {offset: 2.0}
        - !transform/shift-1.1.0 {offset: 4.0}
    no_label: 0
  

Map a number within a range of numbers to transforms which return labels.::

  !<tag:stsci.edu:gwcs/label_mapper-1.0.0>
    mapper: !!omap
    - !!omap
      labels:
      - [3.2, 4.1]
      - [2.67, 2.98]
      - [1.95, 2.3]
    - !!omap
      models:
      - !transform/shift-1.1.0 {offset: 6.0}
      - !transform/shift-1.1.0 {offset: 2.0}
      - !transform/shift-1.1.0 {offset: 4.0}
    inputs: [x, y]
    inputs_mapping: !transform/remap_axes-1.1.0
      mapping: [0]
      n_inputs: 2
  

.. only:: html

   :download:`Original schema in YAML <label_mapper-1.0.0.yaml>`

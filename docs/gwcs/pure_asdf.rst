.. _pure_asdf:

Listing of ``imaging_wcs.asdf``
===============================

Listing of ``imaging_wcs.asdf``::
  

  #ASDF 1.0.0
  #ASDF_STANDARD 1.2.0
  %YAML 1.1
  %TAG ! tag:stsci.edu:asdf/
  --- !core/asdf-1.1.0
  asdf_library: !core/software-1.0.0 {author: Space Telescope Science Institute,
  homepage: 'http://github.com/spacetelescope/asdf', name: asdf, version: 2.2.0.dev1526}
  history:
    extensions:
    - !core/extension_metadata-1.0.0
      extension_class: asdf.extension.BuiltinExtension
      software: {name: asdf, version: 2.2.0.dev1526}
    - !core/extension_metadata-1.0.0
      extension_class: astropy.io.misc.asdf.extension.AstropyExtension
      software: {name: astropy, version: 3.2.dev23222}
    - !core/extension_metadata-1.0.0
      extension_class: astropy.io.misc.asdf.extension.AstropyAsdfExtension
      software: {name: astropy, version: 3.2.dev23222}
    - !core/extension_metadata-1.0.0
      extension_class: gwcs.extension.GWCSExtension
      software: {name: gwcs, version: 0.10.dev417}
  wcs: !<tag:stsci.edu:gwcs/wcs-1.0.0>
    name: ''
    steps:
    - !<tag:stsci.edu:gwcs/step-1.0.0>
      frame: !<tag:stsci.edu:gwcs/frame2d-1.0.0>
        axes_names: [x, y]
        name: detector
        unit: [!unit/unit-1.0.0 pixel, !unit/unit-1.0.0 pixel]
      transform: !transform/compose-1.1.0
        forward:
        - !transform/remap_axes-1.1.0
          mapping: [0, 1, 0, 1]
        - !transform/concatenate-1.1.0
          forward:
          - !transform/polynomial-1.1.0
            coefficients: !core/ndarray-1.0.0
              data:
              - [0.0, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8]
              - [0.1, 0.9, 1.0, 1.1, 0.0]
              - [0.2, 1.2000000000000002, 1.3, 0.0, 0.0]
              - [0.30000000000000004, 1.4000000000000001, 0.0, 0.0, 0.0]
              - [0.4, 0.0, 0.0, 0.0, 0.0]
              datatype: float64
              shape: [5, 5]
          - !transform/polynomial-1.1.0
            coefficients: !core/ndarray-1.0.0
              data:
              - [0.0, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6]
              - [0.2, 1.8, 2.0, 2.2, 0.0]
              - [0.4, 2.4000000000000004, 2.6, 0.0, 0.0]
              - [0.6000000000000001, 2.8000000000000003, 0.0, 0.0, 0.0]
              - [0.8, 0.0, 0.0, 0.0, 0.0]
              datatype: float64
              shape: [5, 5]
    - !<tag:stsci.edu:gwcs/step-1.0.0>
      frame: !<tag:stsci.edu:gwcs/frame2d-1.0.0>
        axes_names: [undist_x, undist_y]
        name: undistorted_frame
        unit: [!unit/unit-1.0.0 pixel, !unit/unit-1.0.0 pixel]
      transform: !transform/compose-1.1.0
        forward:
        - !transform/compose-1.1.0
          forward:
          - !transform/compose-1.1.0
            forward:
            - !transform/concatenate-1.1.0
              forward:
              - !transform/shift-1.2.0 {offset: -2048.0}
              - !transform/shift-1.2.0 {offset: -1024.0}
            - !transform/affine-1.2.0
              inverse: !transform/affine-1.2.0
                matrix: !core/ndarray-1.0.0
                  data:
                  - [65488.318039522, 30828.31712434267]
                  - [26012.509548778366, -66838.34993781192]
                  datatype: float64
                  shape: [2, 2]
                translation: !core/ndarray-1.0.0
                  data: [0.0, 0.0]
                  datatype: float64
                  shape: [2]
              matrix: !core/ndarray-1.0.0
                data:
                - [1.290551569736e-05, 5.9525007864732e-06]
                - [5.0226382102765e-06, -1.2644844123757e-05]
                datatype: float64
                shape: [2, 2]
              translation: !core/ndarray-1.0.0
                data: [0.0, 0.0]
                datatype: float64
                shape: [2]
          - !transform/gnomonic-1.1.0 {direction: pix2sky}
        - !transform/rotate3d-1.2.0 {phi: 5.63056810618, psi: 180.0, theta: -72.05457184279}
        inverse: !transform/compose-1.1.0
          forward:
          - !transform/rotate3d-1.2.0 {direction: celestial2native, phi: 5.63056810618,
            psi: 180.0, theta: -72.05457184279}
          - !transform/compose-1.1.0
            forward:
            - !transform/gnomonic-1.1.0 {direction: sky2pix}
            - !transform/compose-1.1.0
              forward:
              - !transform/affine-1.2.0
                matrix: !core/ndarray-1.0.0
                  data:
                  - [65488.318039522, 30828.31712434267]
                  - [26012.509548778366, -66838.34993781192]
                  datatype: float64
                  shape: [2, 2]
                translation: !core/ndarray-1.0.0
                  data: [0.0, 0.0]
                  datatype: float64
                  shape: [2]
              - !transform/concatenate-1.1.0
                forward:
                - !transform/shift-1.2.0 {offset: 2048.0}
                - !transform/shift-1.2.0 {offset: 1024.0}
        name: linear_transform
    - !<tag:stsci.edu:gwcs/step-1.0.0>
      frame: !<tag:stsci.edu:gwcs/celestial_frame-1.0.0>
        axes_names: [lon, lat]
        name: icrs
        reference_frame: !<tag:astropy.org:astropy/coordinates/frames/icrs-1.1.0>
          frame_attributes: {}
        unit: [!unit/unit-1.0.0 deg, !unit/unit-1.0.0 deg]
  ...

  

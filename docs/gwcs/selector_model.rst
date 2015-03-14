An IFU Example
==============

``GWCS`` has a special `~gwcs.selector.SelectorModel` class which provides a mapping of transforms to
regions on the detector or to other quantities and the means to switch between or select
certain transforms. An example where this can be useful is the WCS of an IFU observation.

The example describes how to create a WCS for an IFU with 21 slits.
In general each slit has a WCS asosciated with it. Each individual WCS transforms from
detector pixels to a composite output coordinate frame with two frames [CelestialFrame, SpectralFrame].
For the sake of brevity we assime the WCS object for each slit has been created.
In order to use the `~gwcs.selector.SelectorModel` we need two more things - a list of labels
and a mask. The labels can be integers or strings (less efficient) and they are used to
create the detector mask as each pixel has associated label.

(insert an image of the projection of an IFU on the detector)


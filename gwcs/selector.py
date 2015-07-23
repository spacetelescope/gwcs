# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import copy
import numpy as np
import warnings

from astropy.extern import six
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter

from . import region
from .utils import RegionError


__all__ = ['SelectorMask', 'RegionsSelector']


def _toindex(value):
    """
    Convert value to an int or an int array.

    Input coordinates should be turned into integers
    corresponding to the center of the pixel.
    They are used to index the mask.

    Examples
    --------
    >>> _toindex(np.array([-0.5, 0.49999]))
    array([0, 0])
    >>> _toindex(np.array([0.5, 1.49999]))
    array([1, 1])
    >>> _toindex(np.array([1.5, 2.49999]))
    array([2, 2])
    """
    indx = np.empty(value.shape, dtype=np.int32)
    indx = np.floor(value + 0.5, out=indx)
    return indx


class SelectorMask(Model):

    """
    A mask model to be used with the `~gwcs.selector.RegionsSelector` transform.

    For an IFU observation, the values of the mask
    correspond to the region (slice)  label.

    Parameters
    ----------
    mask : ndarray
        An array of integers or strings where the values
        correspond to a transform label in `~gwcs.selector.RegionsSelector` model.
        If a transform is not defined the value shoul dbe set to 0 or " ".
    """

    inputs = ('x', 'y')
    outputs = ('z')

    linear = False
    fittable = False

    def __init__(self, mask):
        if mask.dtype.type is not np.unicode_:
            self._mask = np.asanyarray(mask, dtype=np.int)
        else:
            self._mask = mask
        if mask.dtype.type is np.string_:
            self._no_transform_value = ""
        else:
            self._no_transform_value = 0
        super(SelectorMask, self).__init__()

    @property
    def mask(self):
        return self._mask

    @property
    def no_transform_value(self):
        return self._no_transform_value

    def evaluate(self, x, y):

        indx = _toindex(x)
        indy = _toindex(y)
        return self.mask[indx, indy]

    @classmethod
    def from_vertices(cls, mask_shape, regions):
        """
        Create a `~gwcs.selector.SelectorMask` from
        polygon vertices read in from a json file.

        Parameters
        ----------
        mask_shape : tuple
            shape of mask array
        regions: dict
            {region_label : list_of_polygon_vertices}

            The keys in this dictionary should match the region labels
            in `~gwcs.selector.RegionsSelector`.

            The list of vertices is ordered in such a way that when traversed in a
            counterclockwise direction, the enclosed area is the polygon.
            The last vertex must coincide with the first vertex, minimum
            4 vertices are needed to define a triangle.

        Returns
        -------
        mask : `~gwcs.selector.SelectorMask`
            Mask to be used with `~gwcs.selector.SelectorModel`.

        Examples
        -------_
        mask = region.create_regions_mask_from_json((300,300), 'regions.json',
        'region_schema.json')
        """
        labels = np.array(list(regions.keys()))
        mask = np.zeros(mask_shape, dtype=labels.dtype)

        for rid, vert in regions.items():
            pol = region.Polygon(rid, vert)
            mask = pol.scan(mask)

        return cls(mask)


class RegionsSelector(Model):

    """
    A model which maps regions to their corresponding transforms.
    It evaluates the model matching inputs to the correct region/transform.

    Parameters
    ----------
    inputs : list of str
        Names of the inputs.
    outputs : list of str
        Names of the outputs.
    selector : dict
        Mapping of region labels to transforms.
        Labels can be of type int or str, transforms are of type `~astropy.modeling.core.Model`
    mask : `~gwcs.selector.SelectorMask`
        Mask with region labels.
    undefined_transform_value : float, np.nan (default)
        Value to be returned if there's no transform defined for the inputs.
    """
    _param_names = ()

    linear = False
    fittable = False

    def __init__(self, inputs, outputs, selector, mask, undefined_transform_value=np.nan):
        self._inputs = inputs
        self._outputs = outputs
        self.mask = mask.copy()
        self._undefined_transform_value = undefined_transform_value
        self._selector = copy.deepcopy(selector)

        if " " in selector.keys() or 0 in selector.keys():
            raise ValueError('"0" and " " are not allowed as keys.')

        super(RegionsSelector, self).__init__(n_models=1)
        # make sure that keys in mapping match labels in mask
        labels_mask = self.labels_from_mask(mask.mask)
        if not np.in1d(labels_mask, list(self._selector.keys()), assume_unique=True).all() or \
           not np.in1d(list(self._selector.keys()), labels_mask, assume_unique=True).all():
            raise ValueError("Labels don't match regions_mask.")

    @staticmethod
    def labels_from_mask(regions_mask):
        """
        Parameters
        ----------
        regions_mask : ndarray
            An array where regions are indicated by int or str labels.
            " " and 0 indicate a pixel on the detector which is not within any region.
            Evaluating the model in these locations returns NaN or
            ``undefined_transform_value`` if provided.
        """
        labels = np.unique(regions_mask).tolist()
        try:
            labels.remove(0)
        except ValueError:
            pass
        try:
            labels.remove('')
        except ValueError:
            pass
        return labels

    @staticmethod
    def get_unique_regions(mask):
        unique_regions = np.unique(mask).tolist()

        try:
            unique_regions.remove(0)
            unique_regions.remove('')
        except ValueError:
            pass
        try:
            unique_regions.remove("")
        except ValueError:
            pass
        return unique_regions

    def set_input(self, rid):
        """
        Sets one of the inputs and returns a transform associated with it.
        """
        def _eval_input(x, y):
            return self._selector[rid](x, y)
        return _eval_input

    def evaluate(self, x, y):
        """
        Parameters
        ----------
        x : float or ndarray
            Input pixel coordinate.
        y : float or ndarray
            Input pixel coordinate.
        """
        # Get the region labels corresponding to these inputs
        indx = _toindex(x)
        indy = _toindex(y)
        rids = self.mask(indx, indy).flatten()
        # Raise an error if all pixels are outside regions
        if (rids == self.mask.no_transform_value).all():
            raise RegionError("The input positions are not inside any region.")

        # Create output arrays and set any pixels not withing regions to
        # "undefined_transform_value"
        no_trans_ind = (rids == self.mask.no_transform_value).nonzero()
        outputs = [np.empty(rids.shape) for n in range(self.n_outputs)]
        for out in outputs:
            out[no_trans_ind] = self.undefined_transform_value

        # Compute the transformations
        x = x.flatten()
        y = y.flatten()
        uniq = self.get_unique_regions(rids)
        for rid in uniq:
            ind = (rids == rid)
            result = self._selector[rid](x[ind], y[ind])
            for j in range(self.n_outputs):
                outputs[j][ind] = result[j]
        return outputs

    def __call__(self, *inputs, **kwargs):
        """
        Evaluate this model using the given input(s) and the parameter values
        that were specified when the model was instantiated.
        """
        import itertools

        parameters = self._param_sets(raw=True)
        evaluate = self.evaluate
        inputs, format_info = self.prepare_inputs(*inputs, **kwargs)
        outputs = evaluate(*itertools.chain(inputs, parameters))

        if self.n_outputs == 1:
            outputs = (outputs,)
        return self.prepare_outputs(format_info, *outputs, **kwargs)

    def inverse(self):
        """
        The inverse exists if all transforms have an inverse
        and the mask has an inverse.
        """
        selector_inverse = copy.deepcopy(self._selector)
        for tr in selector_inverse:
            selector_inverse[tr] = selector_inverse[tr].inverse
        try:
            mask = self.mask.inverse
        except NotImplementedError:
            raise
        return self.__class__(self.outputs, self.inputs, selector_inverse,
                              mask, self.undefined_transform_value)

    @property
    def undefined_transform_value(self):
        return self._undefined_transform_value

    @undefined_transform_value.setter
    def undefined_transform_value(self, value):
        self._undefined_transform_value = value

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @property
    def outputs(self):
        """The name(s) of the output(s) of the model."""
        return self._outputs

    @property
    def selector(self):
        return self._selector

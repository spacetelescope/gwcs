# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The classes in this module create discontinuous transforms.
The main class is `RegionsSelector`. It maps regions to transforms.
Regions are well defined spaces in the same frame as the inputs to
RegionsSelector. They are assigned unique labels (int or str).

The module also defines classes which map inputs to regions.
An instance of one of the LabelMapper classes is passed as a parameter to RegionsSelector.
Its purpose is to return the labels of the  regions within which
each input is located. The labels are used by RegionsSelector to
pick the corresponding transforms. Finally, RegionsSelector
evaluates the transforms using the matchin inputs.

The base class _LabelMapper can be subclassed to create other
label mappers. The inputs to RegionsSelector are passed to the
LabelMapper instance which must return a label.


"""
from __future__ import absolute_import, division, unicode_literals, print_function

import copy
import numbers
import numpy as np
import warnings

from astropy.extern import six
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter

from . import region
from .utils import RegionError


__all__ = ['LabelMapperArray', 'LabelMapperDict', 'LabelMapperRange', 'RegionsSelector']


def get_unique_regions(regions):
    if isinstance(regions, np.ndarray):
        unique_regions = np.unique(regions).tolist()

        try:
            unique_regions.remove(0)
            unique_regions.remove('')
        except ValueError:
            pass
        try:
            unique_regions.remove("")
        except ValueError:
            pass
    elif isinstance(regions, dict):
        unique_regions = []
        for key in regions.keys():
            unique_regions.append(regions[key](key))
    else:
        raise TypeError("Unable to get unique regions.")
    return unique_regions


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


class _LabelMapper(Model):
    """
    Maps location to a label.

    Labels are strings or numbers which uniquely identify a location.
    For example, labels may represent slices of an IFU or names of spherical polygons.

    Parameters
    ----------
    mapper : object
        A python structure which represents the labels.
        Look at subclasses for examples.
    no_label : str or int
        "" or 0
        A return value for a location which has no corresponding label.
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
    """
    def __init__(self, mapper, no_label, inputs_mapping=None):
        self._no_label = no_label
        self._inputs_mapping = inputs_mapping
        self._mapper = mapper
        super(_LabelMapper, self).__init__()

    @property
    def mapper(self):
        return self._mapper

    @property
    def inputs_mapping(self):
        return self._inputs_mapping

    @property
    def no_label(self):
        return self._no_label

    def get_labels(self):
        """
        Return the labels in the mapper.
        """
        raise NotImplementedError("Subclasses should impement this.")


class LabelMapperArray(_LabelMapper):

    """
    Maps array locations to labels.

    Parameters
    ----------
    mapper : ndarray
        An array of integers or strings where the values
        correspond to a label in `~gwcs.selector.RegionsSelector` model.
        If a transform is not defined the value should be set to 0 or " ".
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.

    Use case:
    For an IFU observation, the array represents the detector and its
    values correspond to the IFU slice  label.

    """

    inputs = ('x', 'y')
    outputs = ('label',)

    linear = False
    fittable = False

    def __init__(self, mapper, inputs_mapping=None):
        if mapper.dtype.type is not np.unicode_:
            mapper = np.asanyarray(mapper, dtype=np.int)
            _no_label = 0
        else:
            _no_label = ""

        super(LabelMapperArray, self).__init__(mapper, _no_label)

    def get_labels(self):
        labels = list(np.unique(self._mapper))

        if isinstance(labels[0], numbers.Number):
            if 0 in labels:
                labels.remove(0)
        else:
            if "" in labels:
                labels.remove("")
        return labels

    def evaluate(self, *args):
        args = [_toindex(a) for a in args]
        return self._mapper[args]

    @classmethod
    def from_vertices(cls, shape, regions):
        """
        Create a `~gwcs.selector.LabelMapperArray` from
        polygon vertices stores in a dict.

        Parameters
        ----------
        shape : tuple
            shape of mapper array
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
        mapper : `~gwcs.selector.LabelMapperArray`
            This models is used with `~gwcs.selector.RegionsSelector`.
            A model which takes the same inputs as `~gwcs.selector.RegionsSelector`
            and returns a label.



        Examples
        --------
        regions = {1: [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
                   2: [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
                   3: [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
                   4: [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
                  }
        mapper = selector.LabelMapperArray.from_vertices((2400, 2400), regions)

        """
        labels = np.array(list(regions.keys()))
        mask = np.zeros(shape, dtype=labels.dtype)

        for rid, vert in regions.items():
            pol = region.Polygon(rid, vert)
            mask = pol.scan(mask)

        return cls(mask)


class LabelMapperDict(_LabelMapper):

    """
    Maps a number to a transform. The transforms take as input the key
    and return a label.

    Use case: inverse transforms of an IFU.
    For an IFU observation, the keys are constant angles (corresponding to a slice)
    and values are transforms which return a slice label.

    Parameters
    ----------
    inputs : tuple of str
        Names for the inputs, e.g. ('alpha', 'beta', lam')
    mapper : dict
        Maps key values to transforms.
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
    """
    standard_broadcasting = False
    outputs = ('labels',)

    linear = False
    fittable = False

    def __init__(self, inputs, mapper, inputs_mapping=None):
        self.inputs = inputs
        # try to determine if slices are numbers of strings
        if isinstance(mapper[list(mapper.keys())[0]](*np.arange(len(inputs))), numbers.Number):
            _no_label = 0
        else:
            _no_label = ""
        super(LabelMapperDict, self).__init__(mapper, _no_label, inputs_mapping)

    def get_labels(self):
        labels = [self._mapper[key](key) for key in self._mapper.keys()]
        return labels

    def evaluate(self, *args):
        shape = args[0].shape
        args = [a.flatten() for a in args]
        if self.inputs_mapping is not None:
            keys = self._inputs_mapping.evaluate(*args)
        else:
            keys = args
        res = np.empty(shape, dtype=type(self._no_label))
        unique = np.unique(keys)
        for key in unique:
            ind = (keys == key)
            inputs = [a[ind] for a in args]
            mapper_keys = list(self.mapper.keys())
            key_ind = np.nonzero([np.allclose(key, mk) for mk in mapper_keys])[0]
            if key_ind.size > 1:
                raise ValueError("More than one key found.")
            elif key_ind.size == 0:
                res[ind] = self._no_label
            else:
                res[ind] = self.mapper[key](*inputs)
        res.shape = shape
        return res


class LabelMapperRange(_LabelMapper):

    """
    The structure this class uses maps a range of values to a transform.
    Given an in put value it checks which range the value falls in and returns
    the corresponding transform. When evaluated the transform returns a label.

    This class is to be used as an argument to `~gwcs.selector.RegionsSelector`.
    All inputs passed to `~gwcs.selector.RegionsSelector` are passed to
    this class. ``inputs_mapping`` is used to filter which input is to be used
    to determine the range. Transforms may use an instance of `~astropy.modeling.models.Mapping`
    and/or `~astropy.modeling.models.Identity` to filter or change the order of their inputs.

    Example: Pick a transform based on wavelength range.
    For an IFU observation, the keys are (lambda_min, lambda_max) pairs
    and values are transforms which return a label corresponding to a slice.
    This label is used by `~gwcs.selector.RegionsSelector` to evaluate the transform
    corresponding to this slice.


    Parameters
    ----------
    inputs : tuple of str
        Names for the inputs, e.g. ('alpha', 'beta', 'lambda')
    mapper : dict
        Maps key values to transforms.
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
    """
    standard_broadcasting = False
    outputs = ('labels',)

    linear = False
    fittable = False

    def __init__(self, inputs, mapper, inputs_mapping=None):
        self.inputs = inputs
        # try to determine if slices are numbers of strings
        #if isinstance(mapper[list(mapper.keys())[0]](*np.arange(len(inputs))), numbers.Number):
        _no_label = 0
        #else:
        #    _no_label = ""
        super(LabelMapperRange, self).__init__(mapper, _no_label, inputs_mapping)

    def get_labels(self):
        labels = [self._mapper[key](key) for key in self._mapper.keys()]
        return labels

    # move this to utils?
    def _find_range(self, value_range, value):
        """
        Returns the index of the range which holds value.

        Parameters
        ----------
        value_range : np.ndarray
            an (2, 2) array of non-overlapping (min, max) values
        value : number
            The value to
        """
        a, b = value_range[:,0], value_range[:,1]
        ind = np.logical_and(value > a, value < b).nonzero()[0]
        if ind.size > 1:
            raise ValueError("There are overlapping ranges.")
        elif ind.size == 0:
            return None
        else:
            return ind.item()

    def evaluate(self, *args):
        shape = args[0].shape
        args = [a.flatten() for a in args]
        if self.inputs_mapping is not None:
            keys = self._inputs_mapping.evaluate(*args)
        else:
            keys = args
        res = np.empty(shape, dtype=type(self._no_label)) # this creates a S1 type array
        unique = np.unique(keys)
        value_range = np.array(self.mapper.keys())
        for key in unique:
            ind = (keys == key)#[0]
            print('ind', ind)
            print('a', a)
            print('args', args)
            inputs = [a[ind] for a in args]
            range_ind = self._find_range(value_range, key)
            #filter out values which are not within any region
            if range_ind is not None:
                transform = self.mapper[list(self.mapper.keys())[range_ind]]
                res[ind] = transform(*inputs)
            else:
                res[ind] = self._no_label
            res.shape = shape
        return res


class RegionsSelector(Model):

    """
    A model which maps locations to their corresponding transforms.
    It uses an instance of `_LabelMapper` to map inputs to the correct region.

    Parameters
    ----------
    inputs : list of str
        Names of the inputs.
    outputs : list of str
        Names of the outputs.
    selector : dict
        Mapping of region labels to transforms.
        Labels can be of type int or str, transforms are of type `~astropy.modeling.core.Model`.
    label_mapper : a subclass of `~gwcs.selector._LabelMapper`
        A model which maps locations to region labels.
    undefined_transform_value : float, np.nan (default)
        Value to be returned if there's no transform defined for the inputs.
    """
    standard_broadcasting = False
    _param_names = ()

    linear = False
    fittable = False

    def __init__(self, inputs, outputs, selector, label_mapper, undefined_transform_value=np.nan, mapping=None):
        self._mapping = mapping
        self._inputs = inputs
        self._outputs = outputs
        self.label_mapper = label_mapper
        self._undefined_transform_value = undefined_transform_value
        self._selector = selector #copy.deepcopy(selector)

        if " " in selector.keys() or 0 in selector.keys():
            raise ValueError('"0" and " " are not allowed as keys.')

        super(RegionsSelector, self).__init__(n_models=1)
        # make sure that keys in mapping match labels in mask
        #labels = label_mapper.get_labels()
        #if not np.in1d(labels, list(self._selector.keys()), assume_unique=True).all() or \
           #not np.in1d(list(self._selector.keys()), labels, assume_unique=True).all():
            #raise ValueError("Selector labels don't match labels in mapper.")

    def set_input(self, rid):
        """
        Sets one of the inputs and returns a transform associated with it.
        """
        def _eval_input(x, y):
            return self._selector[rid](x, y)
        return _eval_input

    def evaluate(self, *args):
        """
        Parameters
        ----------
        args : float or ndarray
            Input pixel coordinate, one input for each dimension.

        """
        # Get the region labels corresponding to these inputs
        rids = self.label_mapper(*args).flatten()
        # Raise an error if all pixels are outside regions
        if (rids == self.label_mapper.no_label).all():
            raise RegionError("The input positions are not inside any region.")

        # Create output arrays and set any pixels not within regions to
        # "undefined_transform_value"
        no_trans_ind = (rids == self.label_mapper.no_label).nonzero()
        outputs = [np.empty(rids.shape) for n in range(self.n_outputs)]
        for out in outputs:
            out[no_trans_ind] = self.undefined_transform_value

        # Compute the transformations
        args = [a.flatten() for a in args]
        uniq = get_unique_regions(rids)

        for rid in uniq:
            ind = (rids == rid)
            inputs = [a[ind] for a in args]
            result = self._selector[rid](*inputs)
            for j in range(self.n_outputs):
                outputs[j][ind] = result[j]
        return outputs

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

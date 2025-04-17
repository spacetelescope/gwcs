# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The classes in this module create discontinuous transforms.

The main class is `RegionsSelector`. It maps inputs to transforms
and evaluates the transforms on the corresponding inputs.
Regions are well defined spaces in the same frame as the inputs.
Regions are assigned unique labels (int or str). The region
labels are used as a proxy between inputs and transforms.
An example is the location of IFU slices in the detector frame.

`RegionsSelector` uses two structures:
  - A mapping of inputs to labels - "label_mapper"
  - A mapping of labels to transforms - "transform_selector"

A "label_mapper" is also a transform, a subclass of `astropy.modeling.Model`,
which returns the labels corresponding to the inputs.

An instance of a ``LabelMapper`` class is passed to `RegionsSelector`.
The labels are used by `RegionsSelector` to match inputs to transforms.
Finally, `RegionsSelector` evaluates the transforms on the corresponding inputs.
Label mappers and transforms take the same inputs as
`RegionsSelector`. The inputs should be filtered appropriately using the
``inputs_mapping`` argument which is ian instance of
`~astropy.modeling.mappings.Mapping`.  The transforms in "transform_selector"
should have the same number of inputs and outputs.

This is illustrated below using two regions, labeled 1 and 2


.. tabs::
    .. tab:: Diagram

        .. graphviz::
            :align: center

            digraph regions {
                node [shape=box, style=filled, fillcolor=lightyellow, fontname="Helvetica", margin="0.3,0.3"];

                subgraph cluster_detector {
                    label="Detector";
                    style=filled;
                    color=lightgray;
                    fontname="Helvetica";

                    region1 [label="1", shape=box, height=2, width=1];
                    region2 [label="2", shape=box, height=1, width=1];

                    {rank=same; region1 region2}
                }
            }

    .. tab:: ASCII

        .. code-block::

            +-----------+
            | +-+       |
            | | |  +-+  |
            | |1|  |2|  |
            | | |  +-+  |
            | +-+       |
            +-----------+


.. tabs::
    .. tab:: Diagram

        .. graphviz::
            :align: center

            digraph selector_workflow {
                splines="ortho"
                node [shape=box, style=filled, fillcolor=lightblue, fontname="Arial", margin="0.3,0.3"];

                // Nodes
                in [shape=point, width=0, height=0];
                inputs [label="inputs", fillcolor=lightgreen];
                label_mapper [label="label mapper"];
                label [label="label"];
                transform_selector [label="transform_selector"];
                transform [label="transform"];
                outputs [label="outputs", fillcolor=lightgreen];

                // Invisible nodes for alignment
                invisible1 [style=invis, shape=point];
                invisible2 [style=invis, shape=point];

                // Main flow
                in ->inputs -> transform -> outputs;
                inputs -> invisible1 [style=invis];
                inputs -> label_mapper [constraint=false];
                label_mapper -> label;
                label -> transform_selector;
                transform_selector -> transform;
                invisible2 -> transform [style=invis];

                // Layout adjustments
                {rank=same; label_mapper; invisible1;}
                {rank=same; transform_selector; invisible2;}
            }

    .. tab:: ASCII

        .. code-block::

                               +--------------+
                               | label mapper |
                               +--------------+
                                 ^       |
                                 |       V
                       ----------|   +-------+
                       |             | label |
                     +--------+      +-------+
            --->     | inputs |          |
                     +--------+          V
                          |          +--------------------+
                          |          | transform_selector |
                          |          +--------------------+
                          V                  |
                     +-----------+           |
                     | transform |<-----------
                     +------------+
                          |
                          V
                     +---------+
                     | outputs |
                     +---------+


The base class _LabelMapper can be subclassed to create other
label mappers.
"""  # noqa: E501

import contextlib
import warnings

import numpy as np
from astropy.modeling import models as astmodels
from astropy.modeling.core import Model

from . import region
from .utils import RegionError, _toindex

__all__ = [
    "LabelMapper",
    "LabelMapperArray",
    "LabelMapperDict",
    "LabelMapperRange",
    "RegionsSelector",
]


def get_unique_regions(regions):
    regions = np.asarray(regions)
    if isinstance(regions, np.ndarray):
        unique_regions = np.unique(regions).tolist()

        try:
            unique_regions.remove(0)
            unique_regions.remove("")
        except ValueError:
            pass
        with contextlib.suppress(ValueError):
            unique_regions.remove("")
    elif isinstance(regions, dict):
        unique_regions = []
        for key in regions:
            unique_regions.append(regions[key](key))
    else:
        msg = "Unable to get unique regions."
        raise TypeError(msg)
    return unique_regions


class LabelMapperArrayIndexingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class _LabelMapper(Model):
    """
    Maps inputs to regions. Returns the region labels corresponding to the inputs.

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
    name : str
        The name of this transform.
    """

    def __init__(self, mapper, no_label, inputs_mapping=None, name=None, **kwargs):
        self._no_label = no_label
        self._inputs_mapping = inputs_mapping
        self._mapper = mapper
        super().__init__(name=name, **kwargs)

    @property
    def mapper(self):
        return self._mapper

    @property
    def inputs_mapping(self):
        return self._inputs_mapping

    @property
    def no_label(self):
        return self._no_label

    def evaluate(self, *args):
        msg = "Subclasses should implement this method."
        raise NotImplementedError(msg)


class LabelMapperArray(_LabelMapper):
    """
    Maps array locations to labels.

    Parameters
    ----------
    mapper : ndarray
        An array of integers or strings where the values
        correspond to a label in `~gwcs.selector.RegionsSelector` model.
        For pixels for which the transform is not defined the value should
        be set to 0 or " ".
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order
        so that the output of it is (x, y) values to index the array.
    name : str
        The name of this transform.

    Use case:
    For an IFU observation, the array represents the detector and its
    values correspond to the IFU slice label.

    """

    n_inputs = 2
    n_outputs = 1

    linear = False
    fittable = False

    def __init__(self, mapper, inputs_mapping=None, name=None, **kwargs):
        if mapper.dtype.type is not np.str_:
            mapper = np.asanyarray(mapper, dtype=int)
            _no_label = 0
        else:
            _no_label = ""
        super().__init__(mapper, _no_label, name=name, **kwargs)
        self.inputs = ("x", "y")
        self.outputs = ("label",)

    def evaluate(self, *args):
        args = tuple([_toindex(a) for a in args])
        try:
            result = self._mapper[args[::-1]]
        except IndexError as e:
            raise LabelMapperArrayIndexingError(e) from e
        return result

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
        >>> regions = {1: [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
        ...            2: [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
        ...            3: [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
        ...            4: [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
        ...           }
        >>> mapper = LabelMapperArray.from_vertices((2400, 2400), regions)

        """  # noqa: E501
        labels = np.array(list(regions.keys()))
        mask = np.zeros(shape, dtype=labels.dtype)

        for rid, vert in regions.items():
            pol = region.Polygon(rid, vert)
            mask = pol.scan(mask)

        return cls(mask)


class LabelMapperDict(_LabelMapper):
    """
    Maps a number to a transform, which when evaluated returns a label.

    Use case: inverse transforms of an IFU.
    For an IFU observation, the keys are constant angles (corresponding to a slice)
    and values are transforms which return a slice number.

    Parameters
    ----------
    inputs : tuple of str
        Names for the inputs, e.g. ('alpha', 'beta', lam')
    mapper : dict
        Maps key values to transforms.
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
        It returns a number which is one of the keys of ``mapper``.
    atol : float
        Absolute tolerance when comparing inputs to ``mapper.keys``.
        It is passed to np.isclose.
    name : str
        The name of this transform.
    """

    standard_broadcasting = False

    linear = False
    fittable = False

    n_outputs = 1

    def __init__(
        self, inputs, mapper, inputs_mapping=None, atol=10**-8, name=None, **kwargs
    ):
        self._atol = atol
        _no_label = 0
        self._inputs = inputs
        self._n_inputs = len(inputs)
        if not all(m.n_outputs == 1 for m in mapper.values()):
            msg = "All transforms in mapper must have one output."
            raise TypeError(msg)
        self._input_units_strict = dict.fromkeys(self._inputs, False)
        self._input_units_allow_dimensionless = dict.fromkeys(self._inputs, False)
        super().__init__(mapper, _no_label, inputs_mapping, name=name, **kwargs)
        self.outputs = ("labels",)

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        self._inputs = val

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, val):
        self._atol = val

    def evaluate(self, *args):
        shape = args[0].shape
        args = [a.flatten() for a in args]
        # if n_inputs > 1, determine which one is to be used as keys
        if self.inputs_mapping is not None:
            keys = self._inputs_mapping.evaluate(*args)
        else:
            keys = args
        keys = keys.flatten()
        # create an empty array for the results
        res = np.zeros(keys.shape) + self._no_label
        # If this is part of a combined transform, some of the inputs
        # may be NaNs.
        # Set NaNs to the ``_no_label`` value
        mapper_keys = list(self.mapper.keys())
        # Loop over the keys in mapper and compare to inputs.
        # Find the indices where they are within ``atol``
        # and evaluate the transform to get the corresponding label.
        for key in mapper_keys:
            ind = np.isclose(key, keys, atol=self._atol)
            inputs = [a[ind] for a in args]
            res[ind] = self.mapper[key](*inputs)
        res.shape = shape
        return res


class LabelMapperRange(_LabelMapper):
    """
    The structure this class uses maps a range of values to a transform.
    Given an input value it finds the range the value falls in and returns
    the corresponding transform. When evaluated the transform returns a label.

    Example: Pick a transform based on wavelength range.
    For an IFU observation, the keys are (lambda_min, lambda_max) tuples
    and values are transforms which return a label corresponding to a slice.

    Parameters
    ----------
    inputs : tuple of str
        Names for the inputs, e.g. ('alpha', 'beta', 'lambda')
    mapper : dict
        Maps tuples of length 2 to transforms.
    inputs_mapping : `~astropy.modeling.mappings.Mapping`
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
    atol : float
        Absolute tolerance when comparing inputs to ``mapper.keys``.
        It is passed to np.isclose.
    name : str
        The name of this transform.
    """

    standard_broadcasting = False

    n_outputs = 1

    linear = False
    fittable = False

    def __init__(self, inputs, mapper, inputs_mapping=None, name=None, **kwargs):
        if self._has_overlapping(np.array(list(mapper.keys()))):
            msg = "Overlapping ranges of values are not supported."
            raise ValueError(msg)
        self._inputs = inputs
        self._n_inputs = len(inputs)
        _no_label = 0
        if not all(m.n_outputs == 1 for m in mapper.values()):
            msg = "All transforms in mapper must have one output."
            raise TypeError(msg)
        self._input_units_strict = dict.fromkeys(self._inputs, False)
        self._input_units_allow_dimensionless = dict.fromkeys(self._inputs, False)
        super().__init__(mapper, _no_label, inputs_mapping, name=name, **kwargs)
        self.outputs = ("labels",)

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        self._inputs = val

    @staticmethod
    def _has_overlapping(ranges):
        """
        Test a list of tuple representing ranges of values has no overlapping ranges.
        """
        d = dict(ranges)
        start = ranges[:, 0]
        end = ranges[:, 1]
        start.sort()
        values = [[v, d[v]] for v in start]
        values = np.array(values)
        start = np.roll(values[:, 0], -1)
        end = values[:, 1]
        return bool(any((end - start)[:-1] > 0) or any(start[-1] > end))

    # move this to utils?
    def _find_range(self, value_range, value):
        """
        Returns the index of the tuple which holds value.

        Parameters
        ----------
        value_range : np.ndarray
            an (2, 2) array of non-overlapping (min, max) values
        value : float
            The value

        Returns
        -------
        ind : int
           Index of the tuple which defines a range holding the input value.
           None, if the input value is not within any available range.
        """
        a, b = value_range[:, 0], value_range[:, 1]
        ind = np.logical_and(value >= a, value <= b).nonzero()[0]
        if ind.size > 1:
            msg = "There are overlapping ranges."
            raise ValueError(msg)
        if ind.size == 0:
            return None
        return ind.item()

    def evaluate(self, *args):
        shape = args[0].shape
        args = [a.flatten() for a in args]
        if self.inputs_mapping is not None:
            keys = self._inputs_mapping.evaluate(*args)
        else:
            keys = args
        keys = keys.flatten()
        # Define an array for the results.
        res = np.zeros(keys.shape) + self._no_label
        nan_ind = np.isnan(keys)
        res[nan_ind] = self._no_label
        value_ranges = list(self.mapper.keys())
        # For each tuple in mapper, find the indices of the inputs
        # which fall within the range it defines.
        for val_range in value_ranges:
            temp = keys.copy()
            temp[nan_ind] = np.nan
            temp = np.where(
                np.logical_or(temp <= val_range[0], temp >= val_range[1]), np.nan, temp
            )
            ind = ~np.isnan(temp)

            if ind.any():
                inputs = [a[ind] for a in args]
                res[ind] = self.mapper[tuple(val_range)](*inputs)
            else:
                continue
        res.shape = shape
        if len(np.nonzero(res)[0]) == 0:
            warnings.warn(
                f"All data is outside the valid range - {self.name}.", stacklevel=2
            )
        return res


class RegionsSelector(Model):
    """
    This model defines discontinuous transforms.
    It maps inputs to their corresponding transforms.
    It uses an instance of `_LabelMapper` as a proxy to map inputs to
    the correct region.

    Parameters
    ----------
    inputs : list of str
        Names of the inputs.
    outputs : list of str
        Names of the outputs.
    selector : dict
        Mapping of region labels to transforms.
        Labels can be of type int or str, transforms are of type
        `~astropy.modeling.Model`.
    label_mapper : a subclass of `~gwcs.selector._LabelMapper`
        A model which maps locations to region labels.
    undefined_transform_value : float, np.nan (default)
        Value to be returned if there's no transform defined for the inputs.
    name : str
        The name of this transform.
    """

    standard_broadcasting = False

    linear = False
    fittable = False

    def __init__(
        self,
        inputs,
        outputs,
        selector,
        label_mapper,
        undefined_transform_value=np.nan,
        name=None,
        **kwargs,
    ):
        self._inputs = inputs
        self._outputs = outputs
        self._n_inputs = len(inputs)
        self._n_outputs = len(outputs)
        self.label_mapper = label_mapper
        self._undefined_transform_value = undefined_transform_value
        self._selector = selector  # copy.deepcopy(selector)

        if " " in selector or 0 in selector:
            msg = '"0" and " " are not allowed as keys.'
            raise ValueError(msg)
        self._input_units_strict = dict.fromkeys(self._inputs, False)
        self._input_units_allow_dimensionless = dict.fromkeys(self._inputs, False)
        super().__init__(n_models=1, name=name, **kwargs)
        # Validate uses_quantity at init time for nicer error message
        _ = self.uses_quantity

    @property
    def uses_quantity(self):
        all_uses_quantity = [t.uses_quantity for t in self._selector.values()]
        not_all_uses_quantity = [not uq for uq in all_uses_quantity]
        if all(all_uses_quantity):
            return True
        if not_all_uses_quantity:
            return False
        msg = (
            "You can not mix models which use quantity and do not use "
            "quantity inside a RegionSelector"
        )
        raise ValueError(msg)

    def set_input(self, rid):
        """
        Sets one of the inputs and returns a transform associated with it.
        """
        if rid in self._selector:
            return self._selector[rid]
        msg = f"Region {rid} not found"
        raise RegionError(msg)

    def inverse(self):
        if self.label_mapper.inverse is not None:
            try:
                transforms_inv = {}
                for rid in self._selector:
                    transforms_inv[rid] = self._selector[rid].inverse
            except AttributeError as err:
                msg = (
                    "The inverse of all regions must be defined"
                    "for RegionsSelector to have an inverse."
                )
                raise NotImplementedError(msg) from err
            return self.__class__(
                self.outputs, self.inputs, transforms_inv, self.label_mapper.inverse
            )
        msg = (
            "The label mapper must have an inverse "
            "for RegionsSelector to have an inverse."
        )
        raise NotImplementedError(msg)

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
            warnings.warn(
                "The input positions are not inside any region.", stacklevel=2
            )

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
            ind = rids == rid
            inputs = [a[ind] for a in args]
            if rid in self._selector:
                result = self._selector[rid](*inputs)
            else:
                # If there's no transform for a label, return np.nan
                result = [
                    np.empty(inputs[0].shape) + self._undefined_transform_value
                    for i in range(self.n_outputs)
                ]
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
    def outputs(self):
        """The name(s) of the output(s) of the model."""
        return self._outputs

    @property
    def selector(self):
        return self._selector

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        self._inputs = val

    @outputs.setter
    def outputs(self, val):
        """
        The name(s) of the output variable(s).
        """
        self._outputs = val

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs


class LabelMapper(_LabelMapper):
    """
    Maps inputs to regions. Returns the region labels corresponding to the inputs.

    Labels are strings or numbers which uniquely identify a location.
    For example, labels may represent slices of an IFU or names of spherical polygons.

    Parameters
    ----------
    mapper : `~astropy.modeling.Model`
        A function which returns a region.
    no_label : str or int
        "" or 0
        A return value for a location which has no corresponding label.
    inputs_mapping : `~astropy.modeling.mappings.Mapping` or tuple
        An optional Mapping model to be prepended to the LabelMapper
        with the purpose to filter the inputs or change their order.
        If tuple, a `~astropy.modeling.mappings.Mapping` model will be created from it.
    name : str
        The name of this transform.
    """

    n_outputs = 1

    def __init__(
        self, inputs, mapper, no_label=np.nan, inputs_mapping=None, name=None, **kwargs
    ):
        self._no_label = no_label
        self._inputs = inputs
        self._n_inputs = len(inputs)
        self._outputs = tuple([f"x{ind}" for ind in list(range(mapper.n_outputs))])
        if isinstance(inputs_mapping, tuple):
            inputs_mapping = astmodels.Mapping(inputs_mapping)
        elif inputs_mapping is not None and not isinstance(
            inputs_mapping, astmodels.Mapping
        ):
            msg = "inputs_mapping must be an instance of astropy.modeling.Mapping."
            raise TypeError(msg)

        self._inputs_mapping = inputs_mapping
        self._mapper = mapper
        self._input_units_strict = dict.fromkeys(self._inputs, False)
        self._input_units_allow_dimensionless = dict.fromkeys(self._inputs, False)
        super(_LabelMapper, self).__init__(name=name, **kwargs)
        self.outputs = ("label",)

    @property
    def inputs(self):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        """
        The name(s) of the input variable(s) on which a model is evaluated.
        """
        self._inputs = val

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def mapper(self):
        return self._mapper

    @property
    def inputs_mapping(self):
        return self._inputs_mapping

    @property
    def no_label(self):
        return self._no_label

    def evaluate(self, *args):
        if self.inputs_mapping is not None:
            args = self.inputs_mapping(*args)
        if self.n_outputs == 1:
            args = [args]
        res = self.mapper(*args)
        if np.isscalar(res):
            res = np.array([res])
        return np.array(res)

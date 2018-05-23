import numpy as np

import astropy.units as u
from astropy.modeling.core import Model, _prepare_inputs_single_model

__all__ = ['LookupTable']


class LookupTable(Model):
    """
    This model indexes the lookup table with the rounded value.
    The primary difference between this model and ``Tabular`` models is that it
    supports non-numeric values in the lookup table.

    .. note::
        Any units on the input value are ignored.

    Parameters
    ----------
    lookup_table : `~astropy.units.Quantity` or `numpy.ndarray`
        The lookup table.
    index_unit : `~astropy.units.Unit`, optional
        The unit of the index of the lookup table, used for the reverse
        lookups. Defaults to `astropy.units.pixel`

    """

    linear = False
    fittable = False
    standard_broadcasting = False
    separable = True

    inputs = ('x',)
    outputs = ('y',)

    @property
    def return_units(self):
        if hasattr(self.lookup_table, 'unit'):
            return {'y': self.lookup_table.unit}
        else:
            return None

    def __init__(self, lookup_table, index_unit=u.pix, name=None):
        super().__init__(name=name)
        self.index_unit = index_unit

        if not isinstance(lookup_table, u.Quantity):
            lookup_table = np.asarray(lookup_table)

        self.lookup_table = lookup_table

    def evaluate(self, point):
        ind = np.asarray(np.round(point), dtype=int)
        return self.lookup_table[ind]

    @property
    def inverse(self):
        return ReverseLookupTable(self.lookup_table, self.index_unit)


class ReverseLookupTable(LookupTable):
    """
    The inverse lookup table.

    This model takes input which is equal to one of the values in the lookup
    table, and returns it's index. For numerical lookup tables it does this by
    using ``isclose`` with ``rtol=1e-15`` to avoid issues with floating point
    precision.
    """

    linear = False
    fittable = False
    standard_broadcasting = False
    separable = True

    inputs = ('x',)
    outputs = ('y',)

    _inverse = None

    @property
    def return_units(self):
        return None

    def prepare_inputs(self, *inputs, model_set_axis=None, equivalencies=None,
                       **kwargs):
        """
        Override this to allow non-numerical inputs for values in non-numerical
        lookup tables.
        """

        inputs = [np.asanyarray(_input) for _input in inputs]

        if issubclass(inputs[0].dtype.type, np.number):
            return super().prepare_inputs(*inputs, model_set_axis=None,
                                          equivalencies=None, **kwargs)
        else:
            return _prepare_inputs_single_model(self, [], inputs, **kwargs)

    def evaluate(self, point):
        if issubclass(self.lookup_table.dtype.type, np.number):
            if isinstance(self.lookup_table, u.Quantity):
                isclose = u.isclose
            else:
                isclose = np.isclose

            inds = np.where(isclose(self.lookup_table, point, rtol=1e-15))[0]

        else:
            inds = np.where(self.lookup_table == point)[0]

        if not inds.size:
            raise ValueError("Value not found in lookup table")

        # There is some strange interaction between inverse and return units,
        # where this needs to be done here and not with return units.
        if hasattr(self.lookup_table, 'unit'):
            inds = u.Quantity(inds, self.index_unit)

        return inds

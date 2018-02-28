import numpy as np

import astropy.units as u
from astropy.modeling.core import Model

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
    """

    linear = False
    fittable = False
    standard_broadcasting = False

    inputs = ('x',)
    outputs = ('y',)

    @property
    def return_units(self):
        if not isinstance(self.lookup_table, u.Quantity):
            return None
        else:
            return {'y': self.lookup_table.unit}

    def __init__(self, lookup_table):
        super().__init__()

        if not isinstance(lookup_table, u.Quantity):
            lookup_table = np.asarray(lookup_table)

        self.lookup_table = lookup_table

    def evaluate(self, point):
        ind = int(np.asarray(np.round(point)))
        return self.lookup_table[ind]

    @property
    def inverse(self):
        return _ReverseLookupTable(self.lookup_table)

    def prepare_inputs(self, *inputs, model_set_axis=None, equivalencies=None,
                       **kwargs):
        """
        Override this to allow non-numerical inputs.
        """

        inputs = [np.asanyarray(_input) for _input in inputs]

        if issubclass(inputs[0].dtype.type, np.number):
            return super().prepare_inputs(*inputs, model_set_axis=None,
                                          equivalencies=None, **kwargs)
        else:
            return inputs, ((1,),)


def _unquantify_allclose_arguments(actual, desired, rtol=1e-5, atol=None):
    from astropy.tests.helper import _unquantify_allclose_arguments
    return _unquantify_allclose_arguments(actual, desired, rtol, atol)


class _ReverseLookupTable(LookupTable):
    """
    This model indexes the lookup table with the rounded value.
    The primary difference between this model and ``Tabular`` models is that it
    supports non-numeric values in the lookup table.
    .. note::
        Any units on the input value are ignored.
    Parameters
    ----------
    lookup_table : `~astropy.units.Quantity` or `numpy.ndarray`
    """

    linear = False
    fittable = False
    standard_broadcasting = False

    inputs = ('x',)
    outputs = ('y',)

    _inverse = None

    @property
    def return_units(self):
        if not isinstance(self.lookup_table, u.Quantity):
            return None
        else:
            return {'y': u.pixel}

    def evaluate(self, point):
        if issubclass(self.lookup_table.dtype.type, np.number):
            if isinstance(self.lookup_table, u.Quantity):
                isclose = lambda *args, **kwargs: np.isclose(*_unquantify_allclose_arguments(*args, **kwargs))
            else:
                isclose = np.isclose

            inds = np.where(isclose(self.lookup_table, point, rtol=1e-15))[0]

        else:
            inds = np.where(self.lookup_table == point)[0]

        if not inds.size:
            raise ValueError("Value not found in lookup table")
        return inds

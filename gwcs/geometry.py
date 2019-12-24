# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import numpy as np
from astropy.modeling.core import Model


__all__ = ['ToDirectionCosines', 'FromDirectionCosines']


class ToDirectionCosines(Model):
    """
    Transform a vector to direction cosines.
    """
    _separable = False

    n_inputs = 3
    n_outputs = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('cosa', 'cosb', 'cosc', 'length')

    def evaluate(self, x, y, z):
        vabs = np.sqrt(1. + x**2 + y**2)
        cosa = x / vabs
        cosb = y / vabs
        cosc = 1. / vabs
        return cosa, cosb, cosc, vabs

    def inverse(self):
        return FromDirectionCosines()


class FromDirectionCosines(Model):
    """
    Transform directional cosines to vector.
    """
    _separable = False

    n_inputs = 4
    n_outputs = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('cosa', 'cosb', 'cosc', 'length')
        self.outputs = ('x', 'y', 'z')

    def evaluate(self, cosa, cosb, cosc, length):

        return cosa * length, cosb * length, cosc * length

    def inverse(self):
        return ToDirectionCosines()

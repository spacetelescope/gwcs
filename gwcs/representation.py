# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

from astropy import units as u
from astropy.utils import OrderedDict
from astropy.coordinates import BaseRepresentation
from astropy.utils.compat.numpy import broadcast_arrays


__all__ = ['Cartesian1DRepresentation', 'Cartesian2DRepresentation']


class Cartesian1DRepresentation(BaseRepresentation):

    """
    Representation of a one dimensional cartesian coordinate.

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        The coordinate along the axis.

    copy : bool, optional
        If True arrays will be copied rather than referenced.
    """

    attr_classes = OrderedDict([('x', u.Quantity)])

    def __init__(self, x, copy=True):

        if not isinstance(x, self.attr_classes['x']):
            raise TypeError('x should be a {0}'.format(self.attr_classes['x'].__name__))

        x = self.attr_classes['x'](x, copy=copy)

        self._x = x

    @property
    def x(self):
        """
        The x component of the point(s).
        """
        return self._x

    #@classmethod
    # def from_cartesian(cls, other):
        # return other

    # def to_cartesian(self):
        # return self


class Cartesian2DRepresentation(BaseRepresentation):

    """
    Representation of a two dimensional cartesian coordinate system

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        The coordinate along the X axis.
    y : `~astropy.units.Quantity`
        The coordinate along the Y axis.
    copy : bool, optional
        If True arrays will be copied rather than referenced.
    """

    attr_classes = OrderedDict([('x', u.Quantity),
                                ('y', u.Quantity)])

    def __init__(self, x, y, copy=True):

        if not isinstance(x, self.attr_classes['x']):
            raise TypeError('x should be a {0}'.format(self.attr_classes['x'].__name__))
        if not isinstance(y, self.attr_classes['y']):
            raise TypeError('y should be a {0}'.format(self.attr_classes['y'].__name__))

        x = self.attr_classes['x'](x, copy=copy)
        y = self.attr_classes['y'](y, copy=copy)

        if not (x.unit.physical_type == y.unit.physical_type):
            raise u.UnitsError("x and y should have matching physical types")

        try:
            x, y = broadcast_arrays(x, y, subok=True)
        except ValueError:
            raise ValueError("Input parameters x and y cannot be broadcast")

        self._x = x
        self._y = y

    @property
    def x(self):
        """
        The x component of the point(s).
        """
        return self._x

    @property
    def y(self):
        """
        The y component of the point(s).
        """
        return self._y

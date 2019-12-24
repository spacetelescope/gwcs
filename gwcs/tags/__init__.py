# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
from astropy import units as u


def _parameter_to_value(param):
    if param.unit is not None:
        return u.Quantity(param)
    return param.value

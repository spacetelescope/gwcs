# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import pytest

import numpy as np

from astropy.modeling.models import Identity
from asdf.tests import helpers

from ... import spectroscopy


sell_glass = spectroscopy.SellmeierGlass(B_coef=[0.58339748, 0.46085267, 3.8915394],
                                         C_coef=[0.00252643, 0.010078333, 1200.556])
sell_zemax = spectroscopy.SellmeierZemax(65, 35, 0, 0, [0.58339748, 0.46085267, 3.8915394],
                                         [0.00252643, 0.010078333, 1200.556], [-2.66e-05, 0.0, 0.0])
snell = spectroscopy.Snell3D()
todircos = spectroscopy.ToDirectionCosines()
fromdircos = spectroscopy.FromDirectionCosines()

transforms = [todircos,
              fromdircos,
              snell,
              sell_glass,
              sell_zemax,
              sell_zemax & todircos| snell & Identity(1) | fromdircos,
              sell_glass & todircos | snell & Identity(1) | fromdircos,
              ]


@pytest.mark.parametrize(('model'), transforms)
def test_transforms(tmpdir, model):
    tree = {'model': model}
    helpers.assert_roundtrip_tree(tree, tmpdir)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import pytest

import numpy as np
from asdf.tests import helpers

from ... import spectroscopy

transforms = [spectroscopy.ToDirectionCosines(),
              spectroscopy.FromDirectionCosines(),
              spectroscopy.Snell3D(1.45),
              spectroscopy.SellmeierGlass(B_coef=[0.58339748, 0.46085267, 3.8915394],
                                          C_coef=[0.00252643, 0.010078333, 1200.556]),
              spectroscopy.SellmeierZemax(65, 35, 0, 0, [0.58339748, 0.46085267, 3.8915394],
                                          [0.00252643, 0.010078333, 1200.556], [-2.66e-05, 0.0, 0.0]),
              ]


@pytest.mark.parametrize(('model'), transforms)
def test_transforms(tmpdir, model):
    tree = {'model': model}
    helpers.assert_roundtrip_tree(tree, tmpdir)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import pytest

import numpy as np
from asdf.tests import helpers

from ... import spectroscopy

transforms = [spectroscopy.ToDirectionCosines(),
              spectroscopy.FromDirectionCosines()
              ]


@pytest.mark.parametrize(('model'), transforms)
def test_transforms(tmpdir, model):
    tree = {'model': model}
    helpers.assert_roundtrip_tree(tree, tmpdir)

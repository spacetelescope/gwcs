# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_equal
from astropy.modeling.models import Mapping, Shift, Scale, Polynomial2D
import asdf

from ...tests.test_region import create_scalar_mapper
from ... import selector


def _assert_mapper_equal(a, b):
    __tracebackhide__ = True

    if a is None:
        return

    assert type(a) is type(b)

    if isinstance(a.mapper, dict):
        assert(a.mapper.__class__ == b.mapper.__class__) # nosec
        assert(all(np.in1d(list(a.mapper), list(b.mapper)))) # nosec
        for k in a.mapper:
            assert (a.mapper[k].__class__ == b.mapper[k].__class__) # nosec
            assert(all(a.mapper[k].parameters == b.mapper[k].parameters))  # nosec
        assert (a.inputs == b.inputs) # nosec
        assert (a.inputs_mapping.mapping == b.inputs_mapping.mapping) # nosec
    else:
        assert_array_equal(a.mapper, b.mapper)


def _assert_selector_equal(a, b):
    __tracebackhide__ = True

    if a is None:
        return

    if isinstance(a, selector.RegionsSelector):
        assert type(a) is type(b)
        _assert_mapper_equal(a.label_mapper, b.label_mapper)
        assert_array_equal(a.inputs, b.inputs)
        assert_array_equal(a.outputs, b.outputs)
        assert_array_equal(a.selector.keys(), b.selector.keys())
        for key in a.selector:
            assert_array_equal(a.selector[key].parameters, b.selector[key].parameters)
        assert_array_equal(a.undefined_transform_value, b.undefined_transform_value)


def assert_selector_roundtrip(s, tmpdir, version=None):
    """
    Assert that a selector can be written to an ASDF file and read back
    in without losing any of its essential properties.
    """
    path = str(tmpdir / "test.asdf")

    with asdf.AsdfFile({"selector": s}, version=version) as af:
        af.write_to(path)

    with asdf.open(path) as af:
        rs = af["selector"]
        if isinstance(s, selector.RegionsSelector):
            _assert_selector_equal(s, rs)
        elif isinstance(s, selector._LabelMapper):
            _assert_mapper_equal(s, rs)
        else:
            assert False


def test_regions_selector(tmpdir):
    m1 = Mapping([0, 1, 1]) | Shift(1) & Shift(2) & Shift(3)
    m2 = Mapping([0, 1, 1]) | Scale(2) & Scale(3) & Scale(3)
    sel = {1: m1, 2: m2}
    a = np.zeros((5, 6), dtype=np.int32)
    a[:, 1:3] = 1
    a[:, 4:5] = 2
    mask = selector.LabelMapperArray(a)
    rs = selector.RegionsSelector(inputs=('x', 'y'), outputs=('ra', 'dec', 'lam'),
                                  selector=sel, label_mapper=mask)
    assert_selector_roundtrip(rs, tmpdir)


def test_LabelMapperArray_str(tmpdir):
    a = np.array([["label1", "", "label2"],
                  ["label1", "", ""],
                  ["label1", "label2", "label2"]])
    mask = selector.LabelMapperArray(a)
    assert_selector_roundtrip(mask, tmpdir)


def test_labelMapperArray_int(tmpdir):

    a = np.array([[1, 0, 2],
                  [1, 0, 0],
                  [1, 2, 2]])
    mask = selector.LabelMapperArray(a)
    assert_selector_roundtrip(mask, tmpdir)


def test_LabelMapperDict(tmpdir):
    dmapper = create_scalar_mapper()
    sel = selector.LabelMapperDict(('x', 'y'), dmapper,
                                   inputs_mapping=Mapping((0,), n_inputs=2), atol=1e-3)
    assert_selector_roundtrip(sel, tmpdir)


def test_LabelMapperRange(tmpdir):
    m = []
    for i in np.arange(9) * .1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(Polynomial2D(2, c0_0=c0_0,
                              c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
    keys = np.array([[4.88, 5.64],
                     [5.75, 6.5],
                     [6.67, 7.47],
                     [7.7, 8.63],
                     [8.83, 9.96],
                     [10.19, 11.49],
                     [11.77, 13.28],
                     [13.33, 15.34],
                     [15.56, 18.09]])
    rmapper = {}
    for k, v in zip(keys, m):
        rmapper[tuple(k)] = v
    sel = selector.LabelMapperRange(('x', 'y'), rmapper,
                                    inputs_mapping=Mapping((0,), n_inputs=2))
    assert_selector_roundtrip(sel, tmpdir)

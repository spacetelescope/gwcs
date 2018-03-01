import pytest
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from gwcs.lookup_table import LookupTable, _ReverseLookupTable


def test_lut_init():
    l1 = LookupTable([1, 2, 3])
    assert l1.return_units is None

    np.testing.assert_allclose(l1.lookup_table, np.array([1, 2, 3]))

    l2 = LookupTable([1, 2, 3] * u.m)
    assert_quantity_allclose(l2.lookup_table, [1, 2, 3] * u.m)
    assert l2.return_units == {'y': u.m}

    l3 = LookupTable(['a', 'b', 'c'])
    assert isinstance(l3.lookup_table, np.ndarray)
    np.all(l3.lookup_table == np.array(['a', 'b', 'c']))

    l4 = LookupTable(np.array(['a', 'b', 'c']))
    assert isinstance(l4.lookup_table, np.ndarray)
    np.all(l4.lookup_table == np.array(['a', 'b', 'c']))


@pytest.mark.parametrize('lut', ([1, 2, 3], [1, 2, 3] * u.m, ['a', 'b', 'c']))
def test_evaluate(lut):
    l = LookupTable(lut)
    ret = l(0)
    assert ret == lut[0]


@pytest.mark.parametrize('lutm',
                         (LookupTable([1, 2, 3]),
                          LookupTable([1, 2, 3] * u.m),
                          LookupTable(['a', 'b', 'c'])))
def test_lut_inverse(lutm):
    inv = lutm.inverse
    assert isinstance(inv, _ReverseLookupTable)
    if isinstance(lutm.lookup_table, u.Quantity):
        assert inv.return_units == {'y': u.pix}
    else:
        assert inv.return_units is None


@pytest.mark.parametrize('lut', ([1, 2, 3], [1, 2, 3] * u.m, ['a', 'b', 'c']))
def test_inverse_evaluate(lut):
    lutm = LookupTable(lut)
    inv = lutm.inverse
    ret = inv(lut[0])
    if isinstance(lutm.lookup_table, u.Quantity):
        assert_quantity_allclose(ret, 0 * u.pix)
    else:
        assert ret == 0


def test_inverse_no_match():
    l2 = LookupTable([1, 2, 3] * u.m)
    with pytest.raises(ValueError):
        l2.inverse(5 * u.m)

    l3 = LookupTable(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        l3.inverse('d')

"""
ASDF tags for spectroscopy related models.
"""

from numpy.testing import assert_allclose

from astropy import units as u
from astropy.units import allclose
from asdf import yamlutil

from ..gwcs_types import GWCSTransformType
from .. spectroscopy import *
from . import _parameter_to_value


__all__ = ['GratingEquationType', 'SellmeierGlassType',
           'SellmeierZemaxType', 'Snell3D']


class SellmeierGlassType(GWCSTransformType):
    name = "sellmeier_glass"
    types = [SellmeierGlass]
    version = "1.1.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        return SellmeierGlass(node['B_coef'], node['C_coef'])

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {'B_coef': _parameter_to_value(model.B_coef),
                'C_coef': _parameter_to_value(model.C_coef)}
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)


class SellmeierZemaxType(GWCSTransformType):
    name = "sellmeier_zemax"
    types = [SellmeierZemax]
    version = "1.1.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        return SellmeierZemax(node['temperature'], node['ref_temperature'],
                              node['ref_pressure'], node['pressure'],
                              node['B_coef'], node['C_coef'], node['D_coef'],
                              node['E_coef'])

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {'B_coef': _parameter_to_value(model.B_coef),
                'C_coef': _parameter_to_value(model.C_coef),
                'D_coef': _parameter_to_value(model.D_coef),
                'E_coef': _parameter_to_value(model.E_coef),
                'temperature': _parameter_to_value(model.temperature),
                'ref_temperature': _parameter_to_value(model.ref_temperature),
                'pressure': _parameter_to_value(model.pressure),
                'ref_pressure': _parameter_to_value(model.ref_pressure)}
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)


class Snell3DType(GWCSTransformType):
    name = "snell3d"
    types = [Snell3D]
    version = "1.1.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        return Snell3D()

    @classmethod
    def to_tree_transform(cls, model, ctx):
        return yamlutil.custom_tree_to_tagged_tree({}, ctx)


class GratingEquationType(GWCSTransformType):
    name = "grating_equation"
    version = '1.1.0'
    types = [AnglesFromGratingEquation3D,
             WavelengthFromGratingEquation]

    @classmethod
    def from_tree_transform(cls, node, ctx):
        groove_density = node['groove_density']
        order = node['order']
        output = node['output']
        if output == "wavelength":
            model = WavelengthFromGratingEquation(groove_density=groove_density,
                                                  spectral_order=order)
        elif output == "angle":
            model = AnglesFromGratingEquation3D(groove_density=groove_density,
                                                spectral_order=order)
        else:
            raise ValueError("Can't create a GratingEquation model with "
                             "output {0}".format(output))
        return model

    @classmethod
    def to_tree_transform(cls, model, ctx):
        if model.groove_density.unit is not None:
            groove_density = u.Quantity(model.groove_density.value,
                                        unit=model.groove_density.unit)
        else:
            groove_density = model.groove_density.value
        node = {'order': model.spectral_order.value,
                'groove_density': groove_density
                }
        if isinstance(model, AnglesFromGratingEquation3D):
            node['output'] = 'angle'
        elif isinstance(model, WavelengthFromGratingEquation):
            node['output'] = 'wavelength'
        else:
            raise TypeError("Can't serialize an instance of {0}"
                            .format(model.__class__.__name__))
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        if isinstance(a, AnglesFromGratingEquation3D):
            assert isinstance(b, AnglesFromGratingEquation3D) # nosec
        elif isinstance(a, WavelengthFromGratingEquation):
            assert isinstance(b, WavelengthFromGratingEquation) # nosec
        allclose(a.groove_density, b.groove_density) # nosec
        assert a.spectral_order.value == b.spectral_order.value # nosec

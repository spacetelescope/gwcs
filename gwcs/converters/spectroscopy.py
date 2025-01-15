"""
ASDF tags for spectroscopy related models.

"""

from asdf_astropy.converters.transform.core import (
    TransformConverterBase,
    parameter_to_value,
)
from astropy import units as u

__all__ = [
    "GratingEquationConverter",
    "SellmeierGlassConverter",
    "SellmeierZemaxConverter",
    "Snell3DConverter",
]


class SellmeierGlassConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/sellmeier_glass-*",)
    types = ("gwcs.spectroscopy.SellmeierGlass",)

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.spectroscopy import SellmeierGlass

        return SellmeierGlass(node["B_coef"], node["C_coef"])

    def to_yaml_tree_transform(self, model, tag, ctx):
        return {
            "B_coef": parameter_to_value(model.B_coef),
            "C_coef": parameter_to_value(model.C_coef),
        }


class SellmeierZemaxConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/sellmeier_zemax-*",)
    types = ("gwcs.spectroscopy.SellmeierZemax",)

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.spectroscopy import SellmeierZemax

        return SellmeierZemax(
            node["temperature"],
            node["ref_temperature"],
            node["ref_pressure"],
            node["pressure"],
            node["B_coef"],
            node["C_coef"],
            node["D_coef"],
            node["E_coef"],
        )

    def to_yaml_tree_transform(self, model, tag, ctx):
        return {
            "B_coef": parameter_to_value(model.B_coef),
            "C_coef": parameter_to_value(model.C_coef),
            "D_coef": parameter_to_value(model.D_coef),
            "E_coef": parameter_to_value(model.E_coef),
            "temperature": parameter_to_value(model.temperature),
            "ref_temperature": parameter_to_value(model.ref_temperature),
            "pressure": parameter_to_value(model.pressure),
            "ref_pressure": parameter_to_value(model.ref_pressure),
        }


class Snell3DConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/snell3d-*",)
    types = ("gwcs.spectroscopy.Snell3D",)

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.spectroscopy import Snell3D

        return Snell3D()

    def to_yaml_tree_transform(self, model, tag, ctx):
        return {}


class GratingEquationConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/grating_equation-*",)
    types = (
        "gwcs.spectroscopy.AnglesFromGratingEquation3D",
        "gwcs.spectroscopy.WavelengthFromGratingEquation",
    )

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.spectroscopy import (
            AnglesFromGratingEquation3D,
            WavelengthFromGratingEquation,
        )

        groove_density = node["groove_density"]
        order = node["order"]
        output = node["output"]
        if output == "wavelength":
            model = WavelengthFromGratingEquation(
                groove_density=groove_density, spectral_order=order
            )
        elif output == "angle":
            model = AnglesFromGratingEquation3D(
                groove_density=groove_density, spectral_order=order
            )
        else:
            msg = f"Can't create a GratingEquation model with output {output}"
            raise ValueError(msg)
        return model

    def to_yaml_tree_transform(self, model, tag, ctx):
        from gwcs.spectroscopy import (
            AnglesFromGratingEquation3D,
            WavelengthFromGratingEquation,
        )

        if model.groove_density.unit is not None:
            groove_density = u.Quantity(
                model.groove_density.value, unit=model.groove_density.unit
            )
        else:
            groove_density = model.groove_density.value
        node = {"order": model.spectral_order.value, "groove_density": groove_density}
        if isinstance(model, AnglesFromGratingEquation3D):
            node["output"] = "angle"
        elif isinstance(model, WavelengthFromGratingEquation):
            node["output"] = "wavelength"
        else:
            msg = f"Can't serialize an instance of {model.__class__.__name__}"
            raise TypeError(msg)
        return node

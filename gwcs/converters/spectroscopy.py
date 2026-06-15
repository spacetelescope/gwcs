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
                groove_density=groove_density,
                spectral_order=order,
                reference_pixel=node.get("reference_pixel", 0),
                reference_wavelength=node.get("reference_wavelength", 0),
                dispersion=node.get("dispersion", 0),
                incident_angle=node.get("incident_angle", 0),
                refractive_index=node.get("refractive_index", 1),
                refractive_index_derivative=node.get("refractive_index_derivative", 0),
                out_of_plane_angle=node.get("out_of_plane_angle", 0),
                camera_angle=node.get("camera_angle", 0),
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
            if model.reference_pixel.value != 0:
                node["reference_pixel"] = model.reference_pixel.value
            if model.reference_wavelength.value != 0:
                if model.reference_wavelength.unit is not None:
                    node["reference_wavelength"] = u.Quantity(
                        model.reference_wavelength.value,
                        unit=model.reference_wavelength.unit,
                    )
                else:
                    node["reference_wavelength"] = model.reference_wavelength.value
            if model.dispersion.value != 0:
                if model.dispersion.unit is not None:
                    node["dispersion"] = u.Quantity(
                        model.dispersion.value,
                        unit=model.dispersion.unit,
                    )
                else:
                    node["dispersion"] = model.dispersion.value
            if model.incident_angle.value != 0:
                if model.incident_angle.unit is not None:
                    node["incident_angle"] = u.Quantity(
                        model.incident_angle.value,
                        unit=model.incident_angle.unit,
                    )
                else:
                    node["incident_angle"] = model.incident_angle.value
            if model.refractive_index.value != 1:
                if model.refractive_index.unit is not None:
                    node["refractive_index"] = u.Quantity(
                        model.refractive_index.value,
                        unit=model.refractive_index.unit,
                    )
                else:
                    node["refractive_index"] = model.refractive_index.value
            if model.refractive_index_derivative.value != 0:
                if model.refractive_index_derivative.unit is not None:
                    node["refractive_index_derivative"] = u.Quantity(
                        model.refractive_index_derivative.value,
                        unit=model.refractive_index_derivative.unit,
                    )
                else:
                    node["refractive_index_derivative"] = (
                        model.refractive_index_derivative.value
                    )
            if model.out_of_plane_angle.value != 0:
                if model.out_of_plane_angle.unit is not None:
                    node["out_of_plane_angle"] = u.Quantity(
                        model.out_of_plane_angle.value,
                        unit=model.out_of_plane_angle.unit,
                    )
                else:
                    node["out_of_plane_angle"] = model.out_of_plane_angle.value
            if model.camera_angle.value != 0:
                if model.camera_angle.unit is not None:
                    node["camera_angle"] = u.Quantity(
                        model.camera_angle.value,
                        unit=model.camera_angle.unit,
                    )
                else:
                    node["camera_angle"] = model.camera_angle.value
        else:
            msg = f"Can't serialize an instance of {model.__class__.__name__}"
            raise TypeError(msg)
        return node

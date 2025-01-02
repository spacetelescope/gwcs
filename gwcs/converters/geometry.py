"""
ASDF tags for geometry related models.

"""

from asdf_astropy.converters.transform.core import TransformConverterBase

__all__ = ["DirectionCosinesConverter", "SphericalCartesianConverter"]


class DirectionCosinesConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/direction_cosines-*",)
    types = (
        "gwcs.geometry.ToDirectionCosines",
        "gwcs.geometry.FromDirectionCosines",
    )

    def from_yaml_tree_transform(self, node, tag, ctx):
        from ..geometry import FromDirectionCosines, ToDirectionCosines

        transform_type = node["transform_type"]
        if transform_type == "to_direction_cosines":
            return ToDirectionCosines()
        if transform_type == "from_direction_cosines":
            return FromDirectionCosines()
        msg = f"Unknown model_type {transform_type}"
        raise TypeError(msg)

    def to_yaml_tree_transform(self, model, tag, ctx):
        from ..geometry import FromDirectionCosines, ToDirectionCosines

        if isinstance(model, FromDirectionCosines):
            transform_type = "from_direction_cosines"
        elif isinstance(model, ToDirectionCosines):
            transform_type = "to_direction_cosines"
        else:
            msg = f"Model of type {model.__class__} is not supported."
            raise TypeError(msg)
        return {"transform_type": transform_type}


class SphericalCartesianConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/spherical_cartesian-*",)
    types = (
        "gwcs.geometry.SphericalToCartesian",
        "gwcs.geometry.CartesianToSpherical",
    )

    def from_yaml_tree_transform(self, node, tag, ctx):
        from ..geometry import CartesianToSpherical, SphericalToCartesian

        transform_type = node["transform_type"]
        wrap_lon_at = node["wrap_lon_at"]
        if transform_type == "spherical_to_cartesian":
            return SphericalToCartesian(wrap_lon_at=wrap_lon_at)
        if transform_type == "cartesian_to_spherical":
            return CartesianToSpherical(wrap_lon_at=wrap_lon_at)
        msg = f"Unknown model_type {transform_type}"
        raise TypeError(msg)

    def to_yaml_tree_transform(self, model, tag, ctx):
        from ..geometry import CartesianToSpherical, SphericalToCartesian

        if isinstance(model, SphericalToCartesian):
            transform_type = "spherical_to_cartesian"
        elif isinstance(model, CartesianToSpherical):
            transform_type = "cartesian_to_spherical"
        else:
            msg = f"Model of type {model.__class__} is not supported."
            raise TypeError(msg)

        return {"transform_type": transform_type, "wrap_lon_at": model.wrap_lon_at}

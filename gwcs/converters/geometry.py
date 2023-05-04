"""
ASDF tags for geometry related models.

"""
from asdf_astropy.converters.transform.core import TransformConverterBase


__all__ = ['DirectionCosinesConverter', 'SphericalCartesianConverter']


class DirectionCosinesConverter(TransformConverterBase):
    tags = ["tag:stsci.edu:gwcs/direction_cosines-*"]
    types = ["gwcs.geometry.ToDirectionCosines",
             "gwcs.geometry.FromDirectionCosines"]

    def from_yaml_tree_transform(self, node, tag, ctx):
        from ..geometry import ToDirectionCosines, FromDirectionCosines
        transform_type = node['transform_type']
        if transform_type == 'to_direction_cosines':
            return ToDirectionCosines()
        elif transform_type == 'from_direction_cosines':
            return FromDirectionCosines()
        else:
            raise TypeError(f"Unknown model_type {transform_type}")

    def to_yaml_tree_transform(self, model, tag, ctx):
        from ..geometry import ToDirectionCosines, FromDirectionCosines
        if isinstance(model, FromDirectionCosines):
            transform_type = 'from_direction_cosines'
        elif isinstance(model, ToDirectionCosines):
            transform_type = 'to_direction_cosines'
        else:
            raise TypeError(f"Model of type {model.__class__} is not supported.")
        node = {'transform_type': transform_type}
        return node


class SphericalCartesianConverter(TransformConverterBase):
    tags = ["tag:stsci.edu:gwcs/spherical_cartesian-*"]
    types = ["gwcs.geometry.SphericalToCartesian",
             "gwcs.geometry.CartesianToSpherical"]

    def from_yaml_tree_transform(self, node, tag, ctx):
        from ..geometry import SphericalToCartesian, CartesianToSpherical
        transform_type = node['transform_type']
        wrap_lon_at = node['wrap_lon_at']
        if transform_type == 'spherical_to_cartesian':
            return SphericalToCartesian(wrap_lon_at=wrap_lon_at)
        elif transform_type == 'cartesian_to_spherical':
            return CartesianToSpherical(wrap_lon_at=wrap_lon_at)
        else:
            raise TypeError(f"Unknown model_type {transform_type}")

    def to_yaml_tree_transform(self, model, tag, ctx):
        from ..geometry import SphericalToCartesian, CartesianToSpherical
        if isinstance(model, SphericalToCartesian):
            transform_type = 'spherical_to_cartesian'
        elif isinstance(model, CartesianToSpherical):
            transform_type = 'cartesian_to_spherical'
        else:
            raise TypeError(f"Model of type {model.__class__} is not supported.")

        node = {
            'transform_type': transform_type,
            'wrap_lon_at': model.wrap_lon_at
        }
        return node

"""
ASDF tags for geometry related models.
"""
from asdf import yamlutil
from ..gwcs_types import GWCSTransformType
from .. geometry import ToDirectionCosines, FromDirectionCosines


__all__ = ['DirectionCosinesType']


class DirectionCosinesType(GWCSTransformType):
    name = "direction_cosines"
    types = [ToDirectionCosines, FromDirectionCosines]
    version = "1.0.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        transform_type = node['transform_type']
        if transform_type == 'to_direction_cosines':
            return ToDirectionCosines()
        elif transform_type == 'from_direction_cosines':
            return FromDirectionCosines()
        else:
            raise TypeError(f"Unknown model_type {transform_type}")

    @classmethod
    def to_tree_transform(cls, model, ctx):
        if isinstance(model, FromDirectionCosines):
            transform_type = 'from_direction_cosines'
        elif isinstance(model, ToDirectionCosines):
            transform_type = 'to_direction_cosines'
        else:
            raise TypeError(f"Model of type {model.__class__} is not supported.")
        node = {'transform_type': transform_type}
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

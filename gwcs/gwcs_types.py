# -*- coding: utf-8 -*-

"""
Defines a ``GWCSType`` used by GWCS.
All types are added automatically to ``_gwcs_types`` and the GWCSExtension.

"""

import six

from astropy.io.misc.asdf.tags.transform.basic import TransformType
from asdf.types import ExtensionTypeMeta, CustomType
from astropy.io.misc.asdf.types import AstropyTypeMeta


__all__ = ['GWCSType', 'GWCSTransformType']


_gwcs_types = set()


class GWCSTransformTypeMeta(AstropyTypeMeta):
    """
    Keeps track of `GWCSType` subclasses that are created so that they can
    be stored automatically by astropy extensions for ASDF.
    """
    def __new__(mcls, name, bases, attrs):
        cls = super(GWCSTransformTypeMeta, mcls).__new__(mcls, name, bases, attrs)
        # Classes using this metaclass are automatically added to the list of
        # jwst types and JWSTExtensions.types.
        if cls.organization == 'stsci.edu' and cls.standard == 'gwcs':
            _gwcs_types.add(cls)

        return cls


class GWCSTypeMeta(ExtensionTypeMeta):
    """
    Keeps track of `GWCSType` subclasses that are created so that they can
    be stored automatically by astropy extensions for ASDF.
    """
    def __new__(mcls, name, bases, attrs):
        cls = super(GWCSTypeMeta, mcls).__new__(mcls, name, bases, attrs)
        # Classes using this metaclass are automatically added to the list of
        # jwst types and JWSTExtensions.types.
        if cls.organization == 'stsci.edu' and cls.standard == 'gwcs':
            _gwcs_types.add(cls)

        return cls


@six.add_metaclass(GWCSTypeMeta)
class GWCSType(CustomType):
    """
    This class represents types that have schemas and tags
    implemented within GWCS.

    """
    organization = 'stsci.edu'
    standard = 'gwcs'


@six.add_metaclass(GWCSTransformTypeMeta)
class GWCSTransformType(TransformType):
    """
    This class represents transform types that have schemas and tags
    implemented within GWCS.

    """
    organization = 'stsci.edu'
    standard = 'gwcs'

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

from asdf.extension import BuiltinExtension
from .tags import LabelMapperType, RegionsSelectorType


class GWCSExtension(BuiltinExtension):
    @property
    def types(self):
        return [LabelMapperType, RegionsSelectorType]

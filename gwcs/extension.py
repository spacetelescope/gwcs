# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

from pyasdf.extension import AsdfExtension
from pyasdf import util, resolver
from .tags import LabelMapperType, RegionsSelectorType


class GWCSExtension(AsdfExtension):
    @property
    def types(self):
        return [LabelMapperType, RegionsSelectorType]

    @property
    def tag_mapping(self):
        return [('tag:stsci.edu:asdf/',
                 'http://stsci.edu/schemas/asdf{tag_suffix}')]

    @property
    def url_mapping(self):
        return resolver.DEFAULT_URL_MAPPING

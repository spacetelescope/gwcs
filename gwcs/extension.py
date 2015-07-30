from pyasdf.extension import AsdfExtension
from pyasdf import util, resolver
from .tags import SelectorMaskType, RegionsSelectorType



class GWCSExtension(AsdfExtension):
    @property
    def types(self):
        return [SelectorMaskType, RegionsSelectorType]

    @property
    def tag_mapping(self):
        return [('tag:stsci.edu:asdf',
                 'http://stsci.edu/schemas/asdf{tag_suffix}')]

    @property
    def url_mapping(self):
        return resolver.DEFAULT_URL_MAPPING


import os.path

from pyasdf.extension import AsdfExtension
from pyasdf import util, resolver
from .tags import LabelMapperType, RegionsSelectorType

#schema_path = os.path.join(os.path.dirname(__file__), 'gwcs', '0.1.0')

class GWCSExtension(AsdfExtension):
    @property
    def types(self):
        #return [SelectorMaskType, RegionsSelectorType]
        return [LabelMapperType, RegionsSelectorType]

    @property
    def tag_mapping(self):
        return [('tag:stsci.edu:asdf',
                 'http://stsci.edu/schemas/asdf{tag_suffix}')]

    @property
    def url_mapping(self):
        return resolver.DEFAULT_URL_MAPPING
        #return [('http://stsci.edu/schemas/gwcs/0.1.0/',
        #         util.filepath_to_url(schema_path) +
        #         '/{url_suffix}.yaml')]

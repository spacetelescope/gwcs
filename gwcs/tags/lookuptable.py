from asdf import yamlutil
from asdf.tests import helpers
from ..gwcs_types import GWCSType

from gwcs.lookup_table import LookupTable

__all__ = ['LookTableType']


class LookTableType(GWCSType):
    name = "lookuptable"
    requires = ['astropy']
    types = [LookupTable]
    version = '1.0.0'

    @classmethod
    def from_tree(cls, node, ctx):
        return LookupTable(node['lookup_table'])

    @classmethod
    def to_tree(cls, obj, ctx):
        return {'lookup_table': yamlutil.custom_tree_to_tagged_tree(obj.lookup_table, ctx)}

import os
import pkg_resources

from astropy.tests.helper import enable_deprecations_as_exceptions

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
#enable_deprecations_as_exceptions()

entry_points = []
for entry_point in pkg_resources.iter_entry_points('pytest11'):
    entry_points.append(entry_point.name)

if "asdf_schema_tester" not in entry_points:
    pytest_plugins = ['asdf.tests.schema_tester']

import os
import pkg_resources

entry_points = []
for entry_point in pkg_resources.iter_entry_points('pytest11'):
    entry_points.append(entry_point.name)

if "asdf_schema_tester" not in entry_points:
    pytest_plugins = ['asdf.tests.schema_tester']

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.helper import enable_deprecations_as_exceptions
from astropy.tests.plugins.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
#enable_deprecations_as_exceptions()


try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['asdf'] = 'asdf'
    del PYTEST_HEADER_MODULES['h5py']
except (NameError, KeyError):
    pass

# This is to figure out the affiliated package version, rather than
# using Astropy's
from gwcs import version

try:
    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version.version
except NameError:   # Needed to support Astropy <= 1.0.0
    pass

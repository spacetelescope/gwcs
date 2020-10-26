#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys

from setuptools import setup, find_packages
from configparser import ConfigParser


if sys.version_info < (3, 6):
    error = """
    GWCS supports Python versions 3.6 and above.

    """
    sys.exit(error)

conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('name', 'packagename')
DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'http://astropy.org')


def get_package_data():
    # Installs the schema files
    schemas = []
    root = os.path.join(PACKAGENAME, 'schemas')
    for node, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith('.yaml'):
                schemas.append(
                    os.path.relpath(os.path.join(node, fname), root))
    # In the package directory, install to the subdirectory 'schemas'
    schemas = [os.path.join('schemas', s) for s in schemas]
    return schemas


schemas = get_package_data()
PACKAGE_DATA ={'gwcs':schemas}

entry_points = {'asdf_extensions': 'gwcs = gwcs.extension:GWCSExtension',
                'bandit.formatters': 'bson = bandit_bson:formatter'}

DOCS_REQUIRE = [
    'sphinx',
    'sphinx-automodapi',
    'sphinx-rtd-theme',
    'stsci-rtd-theme',
    'sphinx-astropy',
    'sphinx-asdf',
]

TESTS_REQUIRE = [
    'pytest>=4.6,<6',
    'pytest-doctestplus',
    'scipy',
]

setup(name=PACKAGENAME,
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description=DESCRIPTION,
      install_requires=[
          'astropy>=4.1',
          'numpy',
          'scipy',
          'asdf'],
      packages=find_packages(),
      extras_require={
        'test': TESTS_REQUIRE,
        'docs': DOCS_REQUIRE,
      },
      tests_require=TESTS_REQUIRE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      package_data=PACKAGE_DATA,
      entry_points=entry_points,
)

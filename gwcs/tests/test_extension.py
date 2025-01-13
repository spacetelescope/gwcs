import io
import warnings

import asdf
import asdf_wcs_schemas
import gwcs.extension

import pytest


@pytest.mark.skipif(asdf_wcs_schemas.__version__ < "0.2.0", reason="version 0.2 provides the new manifests")
def test_empty_extension():
    """
    Test that an empty extension was installed for gwcs 1.0.0
    and that extensions are installed for gwcs 1.0.1 and 1.1.0
    """
    extensions = gwcs.extension.get_extensions()
    assert len(extensions) > 1

    extensions_by_uri = {ext.extension_uri: ext for ext in extensions}

    # check for duplicate uris
    assert len(extensions_by_uri) == len(extensions)

    # check that all 3 versions are installed
    for version in ('1.0.0', '1.0.1', '1.1.0'):
        assert f"asdf://asdf-format.org/astronomy/gwcs/extensions/gwcs-{version}" in extensions_by_uri

    # the 1.0.0 extension should support no tags or types
    legacy = extensions_by_uri["asdf://asdf-format.org/astronomy/gwcs/extensions/gwcs-1.0.0"]
    assert len(legacy.tags) == 0
    assert len(legacy.converters) == 0


def test_open_legacy_without_warning():
    """
    Opening a file produced with extension 1.0.0 should not produce any
    warnings because of the empty extension registered for 1.0.0
    """
    asdf_bytes = b"""#ASDF 1.0.0
#ASDF_STANDARD 1.5.0
%YAML 1.1
%TAG ! tag:stsci.edu:asdf/
--- !core/asdf-1.1.0
asdf_library: !core/software-1.0.0 {author: The ASDF Developers, homepage: 'http://github.com/asdf-format/asdf',
  name: asdf, version: 2.9.2}
history:
  extensions:
  - !core/extension_metadata-1.0.0
    extension_class: asdf.extension._manifest.ManifestExtension
    extension_uri: asdf://asdf-format.org/astronomy/gwcs/extensions/gwcs-1.0.0
    software: !core/software-1.0.0 {name: gwcs, version: 0.18.0}
foo: 1
..."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with asdf.open(io.BytesIO(asdf_bytes)) as af:
            assert af['foo'] == 1

import importlib.metadata


# register asdf_schema_tester pytest plugin if it's not already
# registered
for entry_point in importlib.metadata.entry_points(group='pytest11'):
    if entry_point.name == 'asdf_schema_tester':
        break
else:
    pytest_plugins = ['asdf.tests.schema_tester']

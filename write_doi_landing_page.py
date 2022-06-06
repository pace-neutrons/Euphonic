import json
import sys

import yaml
from cffconvert.cli.cli import cli as cffconvert_main

if len(sys.argv) == 1 or sys.argv[1] == 'latest':
    branch = 'master'
    layout = 'latest'
    landing_page = 'index.markdown'
else:
    branch = f'{sys.argv[1]}'
    layout = 'version'
    landing_page = f'versions/{branch}.markdown'

try:
    cffconvert_main(['--url', f'https://github.com/pace-neutrons/Euphonic/tree/{branch}',
                     '--format', 'schema.org',
                     '--outfile', 'tmp.json'])
except SystemExit:
    pass

with open(f'tmp.json', 'r') as f:
    data = json.load(f)

schema_data = {'schemadotorg': data}
# Link to specific version on readthedocs
if branch == 'master':
    url_subdir = 'latest'
else:
    url_subdir = f'{branch}'
url = schema_data['schemadotorg'].get('url', '')
schema_data['schemadotorg']['url'] = url.replace(f'readthedocs.io', f'readthedocs.io/en/{url_subdir}')

landing_page_content = (
    f'---\n'
    f'layout: {layout}\n'
    f'{yaml.dump(schema_data)}'
    f'---\n')
# Use specific DOI version
if branch == 'master':
    landing_page_content += f'# Euphonic - Latest\n'
else:
    landing_page_content = landing_page_content.replace(
        f'SOFTWARE/EUPHONIC',
        f'SOFTWARE/EUPHONIC/{branch.strip("v")}')

with open(landing_page, 'w') as f:
    f.write(landing_page_content)

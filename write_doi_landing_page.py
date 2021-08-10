import json
import sys

import yaml
from cffconvert.cli import cli as cffconvert_main

if len(sys.argv) == 1 or sys.argv[1] == 'latest':
    #branch = 'master'
    branch = 'cff_test'
    layout = 'latest'
    landing_page = 'index.markdown'
else:
    branch = f'{sys.argv[1]}'
    layout = 'version'
    landing_page = f'versions/{branch}.markdown'

try:
    cffconvert_main(['-u', f'https://github.com/pace-neutrons/Euphonic/tree/{branch}',
                     '-f', 'schema.org',
                     '-of', 'tmp.json'])
except SystemExit:
    pass

with open(f'tmp.json', 'r') as f:
    data = json.load(f)

schema_data = {'schemadotorg': data}
landing_page_content = (
    f'---\n'
    f'layout: {layout}\n'
    f'{yaml.dump(schema_data)}'
    f'---\n')
if branch == 'master':
    landing_page_content += f'# Euphonic - Latest\n'

with open(landing_page, 'w') as f:
    f.write(landing_page_content)

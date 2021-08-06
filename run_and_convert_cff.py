from cffconvert.cli import cli as cffconvert_main
import json
import yaml

froot = 'euphonic_schema'

try:
    cffconvert_main(['-u', 'https://github.com/pace-neutrons/Euphonic',
                     '-f', 'cff',
                     '-of', 'CITATION.cff'])
except SystemExit:
    pass

try:
    cffconvert_main(['-f', 'schema.org', '-of', f'{froot}.json'])
except SystemExit:
    pass

with open(f'{froot}.json', 'r') as f:
    data = json.load(f)

new_data = {'schemadotorg': data}
with open(f'{froot}.yaml', 'w') as f:
    yaml.dump(new_data, f)

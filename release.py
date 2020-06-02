import argparse
import json
import os
import re
import requests
import subprocess
from euphonic import __version__


def main():
    parser = get_parser()
    args = parser.parse_args()

    test = not args.notest
    if args.github:
        release_github(test)

    if args.pypi:
        release_pypi(test)


def release_github(test=True):
    with open('CHANGELOG.rst') as f:
        changelog = f.read()
    euphonic_ver = 'v' + __version__
    changelog_ver = re.findall('\n`(v\d+\.\d+\.\S+)\s', changelog)[0]
    if euphonic_ver != changelog_ver:
        raise Exception((
            f'euphonic.__version__/changelog.rst version mismatch! '
            f'euphonic.__version__: {euphonic_ver} changelog.rst: '
            f'{changelog_ver}'))
    desc = re.search('`v\d+\.\d+\.\S+.*?^-+\n(.*?)^`v', changelog,
                     re.DOTALL | re.MULTILINE).groups()[0].strip()

    payload = {
        "tag_name": changelog_ver,
        "target_commitish": "master",
        "name": changelog_ver,
        "body": desc,
        "draft": False,
        "prerelease": True
    }
    if test:
        print(payload)
    else:
        response = requests.post(
            'https://api.github.com/repos/pace-neutrons/euphonic/releases',
            data=json.dumps(payload),
            headers={"Authorization": "token " + os.environ["GITHUB_TOKEN"]})
        print(response.text)
    

def release_pypi(test=True):
    subprocess.run(['rm','-r','dist'])
    subprocess.run(['rm','-r','build'])
    subprocess.run(['python', 'setup.py', 'sdist'])
    if not test:
        subprocess.run(['twine', 'upload', 'dist/*'])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--github',
        action='store_true',
        help='Release on Github')
    parser.add_argument(
        '--pypi',
        action='store_true',
        help='Release on PyPI')
    parser.add_argument(
        '--notest',
        action='store_true',
        help='Actually send/upload')
    return parser


if __name__ == '__main__':
    main()

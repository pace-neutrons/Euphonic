#! /usr/bin/env python
import argparse
import json
import os
import re

from packaging.version import Version
import requests
import yaml

from euphonic import __version__

POST_TIMEOUT = 5.  # 5s timeout; otherwise failed request can hang


def main():
    parser = get_parser()
    args = parser.parse_args()

    test = not args.notest
    if args.github:
        release_github(test)


def release_github(test=True):
    with open('CHANGELOG.rst', encoding='utf8') as f:
        changelog = f.read()
    with open('CITATION.cff', encoding='utf8') as f:
        citation = yaml.safe_load(f)

    euphonic_ver = __version__
    is_prerelease = Version(euphonic_ver).is_prerelease

    version_dict = {}
    if not is_prerelease:
        version_dict['CHANGELOG.rst'] = re.findall(r'\n`(v\d+\.\d+\.\S+)\s',
                                                   changelog)[0]
    version_dict['CITATION.cff'] = 'v' + citation['version']
    for ver_name, ver in version_dict.items():
        if euphonic_ver != ver:
            msg = (
                f'euphonic.__version__/{ver_name} version mismatch! '
                f'euphonic.__version__: {euphonic_ver} {ver_name}: {ver}'
            )
            raise ValueError(msg)

    if is_prerelease:
        body_re = r'`Unreleased.*?^-+\n(.*?)^`v'
    else:
        body_re = r'`v\d+\.\d+\.\S+.*?^-+\n(.*?)^`v'

    desc = re.search(body_re, changelog,
                     re.DOTALL | re.MULTILINE).groups()[0].strip()

    payload = {
        'tag_name': euphonic_ver,
        'target_commitish': 'master',
        'name': euphonic_ver,
        'body': desc,
        'draft': False,
        'prerelease': is_prerelease,
    }
    if test:
        print(payload)

        if 'GITHUB_TOKEN' in os.environ:
            print('Found GITHUB_TOKEN')
        else:
            print('No GITHUB_TOKEN set')
    else:
        response = requests.post(
            'https://api.github.com/repos/pace-neutrons/euphonic/releases',
            data=json.dumps(payload),
            headers={'Authorization': 'token ' + os.environ['GITHUB_TOKEN']},
            timeout=POST_TIMEOUT)
        print(response.text)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--github',
        action='store_true',
        help='Release on Github')
    parser.add_argument(
        '--notest',
        action='store_true',
        help='Actually send/upload')
    return parser


if __name__ == '__main__':
    main()

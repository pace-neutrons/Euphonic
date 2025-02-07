#! /usr/bin/env python

"""Bump the current version number in CHANGELOG.rst

Unreleased changes will be moved into a section for the new version number
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
import re
from typing import NamedTuple

REPOSITORY_ADDRESS = "https://github.com/pace-neutrons/Euphonic"


@dataclass
class Block:
    tag: str
    previous_tag: str
    content: str


def parse_changelog(changelog_file: Path) -> list[Block]:
    """Read all sections from changelog file

    Each section should begin with two lines in format

      `TAG <repository_address/compare/PREV_TAG...TAG>`_
      --------------------------------------------------

    except for 'Unreleased' which compares with HEAD
    """

    with open(changelog_file) as fd:
        split_text = re.split(r"`(\S+) <\S+/compare/(\S+)\.\.\.\S+>`_\n-+",
                              fd.read())

    # First item is always empty?
    split_text = split_text[1:]

    blocks = [Block(*block_data)
              for block_data in batched(split_text, n=3)]

    for block in blocks:
        block.content = block.content.strip()

    return blocks


def tag_to_header(tag: str, previous_tag: str) -> str:
    compare_tag = "HEAD" if tag == "Unreleased" else tag
    header = f"`{tag} <{REPOSITORY_ADDRESS}/compare/{previous_tag}...{compare_tag}>`_"
    return header + "\n" + len(header) * "-"


def to_text(blocks: list[Block]) -> str:
    return "\n\n".join(
        tag_to_header(block.tag, block.previous_tag) + "\n\n" + block.content
        for block in blocks
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("filename", type=Path, help="Input CHANGELOG.rst file")
    parser.add_argument("tag", type=str, help="Tag for new/updated version")
    parser.add_argument("--replace", action="store_true", help="Write to existing file")

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    blocks = parse_changelog(args.filename)

    if args.replace:
        with open(args.filename, "w") as fd:
            print(to_text(blocks), file=fd)

    else:
        print(to_text(blocks))

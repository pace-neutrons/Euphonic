#! /usr/bin/env python

"""Bump the current version number in CHANGELOG.rst

Unreleased changes will be moved into a section for the new version number
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
import re


REPOSITORY_ADDRESS = "https://github.com/pace-neutrons/Euphonic"


@dataclass
class Block:
    """CHANGELOG data corresponding to a single release"""
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
        split_text = re.split(r"`(\S+) <\S+/compare/(\S+)\.\.\.\S+>`_\n-+", fd.read())

    # First item is always empty?
    split_text = split_text[1:]

    blocks = [Block(*block_data) for block_data in batched(split_text, n=3)]

    for block in blocks:
        block.content = block.content.strip()

    return blocks


def bump_version(blocks: list[Block], tag: str) -> None:
    """Create or update a new or existing block with content from "Unreleased" """

    all_tags = [block.tag for block in blocks]

    if all_tags[0] != "Unreleased":
        raise ValueError("CHANGELOG should always begin with Unreleased")

    if all_tags[1] != tag:
        # This version is not in the CHANGELOG yet
        previous_tag = all_tags[1]
        blocks.insert(1, Block(tag, previous_tag, ""))

    blocks[1].content = "\n\n".join(
        [txt for txt in (blocks[0].content, blocks[1].content) if txt]
    )
    blocks[0].content = ""


def tag_to_header(tag: str, previous_tag: str) -> str:
    """Produce header including github diff link"""
    compare_tag = "HEAD" if tag == "Unreleased" else tag
    header = f"`{tag} <{REPOSITORY_ADDRESS}/compare/{previous_tag}...{compare_tag}>`_"
    return header + "\n" + len(header) * "-"


def to_text(blocks: list[Block]) -> str:
    """Dump blocks to single multiline string"""
    return "\n\n".join(
        tag_to_header(block.tag, block.previous_tag)
        + (f"\n\n{block.content}" if block.content else "")
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
    bump_version(blocks, args.tag)

    if args.replace:
        with open(args.filename, "w") as fd:
            print(to_text(blocks), file=fd)

    else:
        print(to_text(blocks))

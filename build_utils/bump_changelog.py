#! /usr/bin/env python

"""Bump the current version number in CHANGELOG.rst

Unreleased changes will be moved into a section for the new version number
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import re

from packaging.version import Version
from toolz.itertoolz import partition

REPOSITORY_ADDRESS = "https://github.com/pace-neutrons/Euphonic"
LINK_STRING = "`{tag} <" + REPOSITORY_ADDRESS + "/compare/{previous_tag}...{compare_tag}>`_"


@dataclass
class Block:
    """CHANGELOG data corresponding to a single release"""
    tag: str
    previous_tag: str
    content: str

    @staticmethod
    def tag_to_header(tag: str, previous_tag: str) -> str:
        """Produce header including github diff link"""
        compare_tag = "HEAD" if tag == "Unreleased" else tag
        header = LINK_STRING.format(tag=tag,
                                    previous_tag=previous_tag,
                                    compare_tag=compare_tag)
        return header + "\n" + len(header) * "-"

    def __str__(self) -> str:
        txt = self.tag_to_header(self.tag, self.previous_tag)
        if self.content:
            txt += f"\n\n{self.content}"

        return txt


def parse_changelog(changelog_file: Path) -> list[Block]:
    """Read all sections from changelog file

    Each section should begin with two lines in format

      `TAG <repository_address/compare/PREV_TAG...TAG>`_
      --------------------------------------------------

    except for 'Unreleased' which compares with HEAD
    """

    with changelog_file.open(encoding="utf8") as fd:
        split_text = re.split(
            LINK_STRING.replace(".", r"\.").format(
                tag=r"(\S+)", previous_tag=r"(\S+)", compare_tag=r"\S+"
            )
            + r"\n-+",
            fd.read(),
        )

    # First item is always empty?
    split_text = split_text[1:]

    # From Python 3.12 can replace partition with itertools.batched
    blocks = [Block(tag, prev_tag, content) for tag, prev_tag, content in partition(3, split_text)]

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
        txt for txt in (blocks[0].content, blocks[1].content) if txt
    )

    # Update "Unreleased" section
    blocks[0].content = ""
    blocks[0].previous_tag = tag


def to_text(blocks: list[Block]) -> str:
    """Dump blocks to single multiline string"""
    return "\n\n".join(map(str, blocks))


def get_parser() -> ArgumentParser:
    """Use argparse to get user input"""
    parser = ArgumentParser()
    parser.add_argument("filename", type=Path, help="Input CHANGELOG.rst file")
    parser.add_argument("tag", type=str, help="Tag for new/updated version")
    parser.add_argument("--replace", action="store_true", help="Write to existing file")

    return parser



def main() -> None:
    """Entrypoint"""
    args = get_parser().parse_args()

    if Version(args.tag).is_prerelease:
        print("New version is a pre-release, leave CHANGELOG alone.")
        return None

    blocks = parse_changelog(args.filename)
    bump_version(blocks, args.tag)

    if args.replace:
        with open(args.filename, "w", encoding="utf8") as fd:
            print(to_text(blocks), file=fd)

    else:
        print(to_text(blocks))


if __name__ == "__main__":
    main()

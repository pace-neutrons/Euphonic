"""Script to compute version from git tags.

Provides 3 means of updating:

- python version.py

  Print computed version to stdout

- python version.py --dump

  Update version in `euphonic/version.py`

- python version.py --dist

  Update versions in both `euphonic/version.py` and in meson sdist build directory.
"""

from pathlib import Path
import subprocess
import sys
import os

if new_dir := os.getenv("MESON_SOURCE_ROOT"):
    os.chdir(new_dir)

gits = ["git"]
if sys.platform == "win32":
    gits += ["git.cmd", "git.exe"]

match sys.argv:
    case [_, "--dist"]:
        COMMAND = "dist"
    case [_, "--dump"]:
        COMMAND = "dump"
    case [_]:
        COMMAND = "print"

version_file = Path(__file__).parent.parent / "euphonic" / "version.py"

for gitcmd in gits:
    try:
        print(f"Trying {gitcmd} ...", file=sys.stderr)
        proc = subprocess.run([gitcmd, "describe", "--tags", "--dirty"],
                              capture_output=True, check=True, text=True)
    except FileNotFoundError as err:
        print(f"Tried {gitcmd}, File Not Found", file=sys.stderr)
        continue

    except (subprocess.CalledProcessError) as err:
        print(f"Tried {gitcmd}, returned: {err}", file=sys.stderr)
        print(f"Stdout: '{err.stdout.strip()}'", file=sys.stderr)
        print(f"Stderr: '{err.stderr.strip()}'", file=sys.stderr)
        continue

    version, *dirty = proc.stdout.strip().split("-")
    if dirty:
        version += f"+{dirty[0]}.{dirty[1]}{'.dirty' if len(dirty) > 2 and COMMAND != 'dump' else ''}"
    break

else:  # Can't use git
    print("All git implementations failed, reading version file", file=sys.stderr)
    version = version_file.read_text().split("=")[1].strip('"\n ')

match COMMAND:
    case "dist":
        version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")
        dist_path = os.getenv("MESON_DIST_ROOT")
        dist_version_file = Path(dist_path) / "euphonic" / "version.py"
        dist_version_file.parent.mkdir(parents=True, exist_ok=True)
        dist_version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")

    case "dump":
        version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")

    case "print":
        print(version)

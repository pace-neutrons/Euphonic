from pathlib import Path
import subprocess
import sys
import os

gits = ["git"]
if sys.platform == "win32":
    gits += ["git.cmd", "git.exe"]

match sys.argv:
    case [_, "--dist"]:
        type_ = "dist"
    case [_, "--dump"]:
        type_ = "dump"
    case [_]:
        type_ = "print"

version_file = Path(__file__).parent.parent / "euphonic" / "version.py"

for gitcmd in gits:
    try:
        proc = subprocess.run([gitcmd, "describe", "--tags", "--dirty"], capture_output=True, check=True, text=True)
        version, *dirty = proc.stdout.strip().split("-")
        if dirty:
            version += f"+{dirty[0]}.{dirty[1]}{'.dirty' if len(dirty) > 2 and type_ != 'dump' else ''}"
        break

    except (OSError, subprocess.CalledProcessError) as err:
        print(f"Tried {gitcmd}, returned: {err}", file=sys.stderr)
else:  # Can't use git
    version = version_file.read_text().split("=")[1].strip('"\n ')

match type_:
    case "dist":
        version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")
        dist_path = os.getenv("MESON_DIST_ROOT")
        version_file = Path(dist_path) / "euphonic" / "version.py"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")

    case "dump":
        version_file.write_text(f'__version__ = "{version}"', encoding="utf-8")

    case "print":
        print(version)

from pathlib import Path
import subprocess
import sys

gits = ["git"]
if sys.platform == "win32":
    gits += ["git.cmd", "git.exe"]

for gitcmd in gits:
    try:
        proc = subprocess.run([gitcmd, "describe", "--tags", "--dirty"], capture_output=True, check=True, text=True)
        version, *dirty = proc.stdout.strip().split("-")
        if dirty:
            version += f"+{dirty[0]}.{dirty[1]}{'.dirty' if len(dirty) > 2 else ''}"
        break

    except (OSError, subprocess.CalledProcessError):
        continue
else:  # Can't use git
    version_file = Path(__file__).parent.parent / "euphonic" / "version.py"
    version = version_file.read_text().split("=")[1].strip('"\n ')


print(version)

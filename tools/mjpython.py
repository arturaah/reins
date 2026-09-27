"""Launch MuJoCo's macOS Python trampoline with uv's shared-library directory available.

Use the repo Python: .venv/bin/python tools/mjpython.py -m harness sim ... --viewer
Arguments pass through unchanged to the installed mjpython launcher.
"""
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def main():
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")
    if sys.platform == "darwin" and libdir and library and (Path(libdir) / library).is_file():
        # mjpython dlopens the interpreter from its .app bundle, where uv's @rpath no longer resolves.
        previous = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = str(Path(libdir).resolve()) + (":" + previous if previous else "")
    launcher = Path(sys.executable).parent / "mjpython"
    if not launcher.is_file():
        raise SystemExit(f"MuJoCo's mjpython launcher is missing: {launcher}")
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()

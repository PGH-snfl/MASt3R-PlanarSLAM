"""Prefer this project's copied dependencies over another editable checkout."""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
for _relative in ("thirdparty/mast3r", "thirdparty/in3d", "thirdparty/mast3r/asmk"):
    _path = _root / _relative
    # ASMK's installed wheel contains its compiled hamming extension.
    if "asmk" not in _relative and _path.is_dir():
        sys.path.insert(0, str(_path))

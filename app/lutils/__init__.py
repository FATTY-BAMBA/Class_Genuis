from pathlib import Path
from typing import Union
PathLike = Union[str, Path]

def _ensure_parent(p: PathLike):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def openf(p: PathLike, mode="r", encoding="utf-8"):
    if any(x in mode for x in ("w", "a", "+")):
        _ensure_parent(p)
    return open(p, mode, encoding=encoding)

def writef(p: PathLike, data: str, encoding="utf-8"):
    _ensure_parent(p)
    with open(p, "w", encoding=encoding) as f:
        f.write(data)
    return Path(p)

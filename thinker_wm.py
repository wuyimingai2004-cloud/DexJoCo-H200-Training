"""Single import entry for the vendored Thinker-WM source.

Use ``thinker_wm.models`` / ``thinker_wm.runtime`` without global aliases or
sys.path injection. Importing this module does not load model dependencies.
The model source is delivered separately from the portable framework wheel.
"""
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "third_party" / "Thinker-WM"
__path__ = [str(SOURCE_ROOT)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__


def source_path(*parts: str) -> Path:
    """Locate an existing resource inside the delivered model source tree."""
    path = SOURCE_ROOT.joinpath(*parts).resolve()
    if not path.is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError("Thinker-WM resource must remain inside its source tree")
    if not path.exists():
        raise FileNotFoundError(f"Thinker-WM source/resource is not installed: {path}")
    return path

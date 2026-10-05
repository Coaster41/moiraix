"""Unpickling support for Lightning ``.ckpt`` files written by the uni2ts repo.

Those checkpoints pickle hyperparameter objects (e.g. loss functions) under their
``uni2ts.*`` module paths. ``install_uni2ts_aliases()`` makes ``uni2ts.X`` resolve
to the matching ``moiraix`` module so they can be resumed here. HF checkpoints
(``HF_checkpoints/``) don't need this.

Only install it when loading such a checkpoint: once active, ``import uni2ts``
succeeds, which would fool code that probes for a real uni2ts install.
"""

import importlib
import importlib.abc
import importlib.util
import sys

# uni2ts.model.{moiraix,moiraic,moiraie} all live in moiraix.model now.
_PREFIXES = [
    ("uni2ts.model.moiraix", "moiraix.model"),
    ("uni2ts.model.moiraic", "moiraix.model"),
    ("uni2ts.model.moiraie", "moiraix.model"),
    ("uni2ts", "moiraix"),
]


def _target(fullname: str):
    for old, new in _PREFIXES:
        if fullname == old or fullname.startswith(old + "."):
            return new + fullname[len(old) :]
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        new = _target(fullname)
        if new is None or importlib.util.find_spec(new) is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(new), is_package=True
        )


def install_uni2ts_aliases() -> None:
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        # Must precede PathFinder, which would otherwise load a second copy of
        # each module under its uni2ts.* name.
        sys.meta_path.insert(0, _AliasFinder())

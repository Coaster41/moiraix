import pickle
import sys

from moiraix.common import legacy
from moiraix.loss.packed import PackedQuantileMTPLoss


def test_uni2ts_pickles_resolve_to_moiraix():
    # Protocol 0 stores class paths as plain text, so we can forge a pickle
    # exactly as the uni2ts repo would have written it.
    data = pickle.dumps(PackedQuantileMTPLoss(), protocol=0)
    data = data.replace(b"moiraix.loss", b"uni2ts.loss")
    assert b"uni2ts.loss.packed" in data

    legacy.install_uni2ts_aliases()
    try:
        obj = pickle.loads(data)
        assert type(obj) is PackedQuantileMTPLoss  # aliased, not a second copy
    finally:
        sys.meta_path[:] = [
            f for f in sys.meta_path if not isinstance(f, legacy._AliasFinder)
        ]
        for name in [
            m for m in sys.modules if m == "uni2ts" or m.startswith("uni2ts.")
        ]:
            del sys.modules[name]

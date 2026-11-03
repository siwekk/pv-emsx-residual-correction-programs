from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

@contextmanager
def temporary_directory():
    base=Path(os.environ.get('EMSX_TEST_TMP',tempfile.gettempdir())).resolve()
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='emsx-test-',dir=base) as tmp:
        if Path(tmp).resolve().parent != base:
            raise RuntimeError('Unexpected temporary directory target')
        yield tmp

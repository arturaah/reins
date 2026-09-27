import pytest

from harness import config as hcfg


@pytest.fixture
def cfg(tmp_path):
    cfg = hcfg.load()
    cfg['experience']['dir'] = str(tmp_path / 'experience')
    return cfg

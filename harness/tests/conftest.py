import pytest

from harness import config as hcfg


@pytest.fixture
def cfg():
    return hcfg.load()

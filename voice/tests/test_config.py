import os

from voice import config


def clean(monkeypatch):
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    for name, aliases in config.ALIASES.items():
        for key in (name, *aliases):
            monkeypatch.delenv(key, raising=False)


def test_loads_project_env_aliases_from_any_working_directory(tmp_path, monkeypatch):
    clean(monkeypatch)
    (tmp_path / '.env').write_text('OPENAI_KEY=openai-test\nAIC_KEY=aic-test\nUNRELATED=private\n')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    config.load_keys()
    assert os.environ['OPENAI_API_KEY'] == 'openai-test'
    assert os.environ['AIC_SDK_LICENSE'] == 'aic-test'
    assert 'UNRELATED' not in os.environ


def test_environment_alias_beats_file_canonical_and_explicit_file_replaces_default(tmp_path, monkeypatch):
    clean(monkeypatch)
    monkeypatch.setenv('OPENAI_KEY', 'environment-test')
    (tmp_path / '.env').write_text('AIC_KEY=default-test\n')
    custom = tmp_path / 'custom.env'
    custom.write_text('OPENAI_API_KEY=file-test\nAIC_KEY=custom-test\n')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    config.load_keys(custom)
    assert os.environ['OPENAI_API_KEY'] == 'environment-test'
    assert os.environ['AIC_SDK_LICENSE'] == 'custom-test'


def test_legacy_names_and_environment_canonical_remain_supported(tmp_path, monkeypatch):
    clean(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'canonical-test')
    monkeypatch.setenv('OPENAI_KEY', 'alias-test')
    path = tmp_path / '.env'
    path.write_text('OPENAI_API_KEY=legacy-test\nAIC_SDK_LICENSE=license-test\n')
    config.load_keys(path)
    assert os.environ['OPENAI_API_KEY'] == 'canonical-test'
    assert os.environ['AIC_SDK_LICENSE'] == 'license-test'

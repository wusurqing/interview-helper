import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.common.provider_settings import router
from core import config


@pytest.fixture
def legacy_file(monkeypatch, tmp_path):
    target = tmp_path / 'config.json'
    models = [
        {'id': 'vision', 'name': 'Vision', 'model': 'same', 'api_key': 'fixture-key', 'supports_vision': True},
        {'id': 'alias', 'name': 'Alias', 'model': 'same', 'api_key': 'fixture-key'},
        {'id': 'other', 'name': 'Other', 'model': 'other', 'api_key': 'other-key', 'enabled': False},
    ]
    target.write_text(json.dumps({
        'models': models,
        'active_model': 0,
        'review_model_index': 0,
        'assist_model_ids': ['alias', 'vision'],
        'assist_screenshot_model_id': 'vision',
        'written_exam_model_ids': ['vision'],
        'knowledge_analysis': {'model_index': 0},
    }), encoding='utf-8')
    monkeypatch.setattr(config, 'CONFIG_FILE', target)
    monkeypatch.setattr(config, '_config', None)
    return target


def test_provider_migration_groups_credentials_but_preserves_model_aliases(legacy_file):
    cfg = config.get_config()
    providers, models = config.get_provider_layout(cfg)
    assert len(providers) == 2
    assert [model.id for model in models] == ['vision', 'alias', 'other']
    assert models[0].provider_id == models[1].provider_id != models[2].provider_id
    assert cfg.assist_model_ids == ['alias', 'vision']
    assert cfg.active_model_id == cfg.review_model_id == cfg.knowledge_analysis.model_id == 'vision'
    assert 'model_providers' not in json.loads(legacy_file.read_text(encoding='utf-8'))


def test_confirmed_disable_clears_every_binding_and_writes_single_credential(legacy_file):
    before = legacy_file.read_bytes()
    providers, models = config.get_provider_layout(config.get_config())
    revision = config.provider_layout_revision(config.get_config())
    models[0] = models[0].model_copy(update={'enabled': False})
    with pytest.raises(config.ModelListChanged):
        config.save_provider_layout(revision, providers, models, set(), 'vision', lambda *_: True)
    assert legacy_file.read_bytes() == before

    saved = config.save_provider_layout(revision, providers, models, {'vision'}, 'vision', lambda *_: True)
    assert saved.assist_model_ids == ['alias']
    assert saved.assist_screenshot_model_id is None
    assert saved.written_exam_model_ids == []
    assert saved.active_model_id is None
    assert saved.review_model_id is None
    assert saved.knowledge_analysis.model_id is None
    assert not saved.models[0].enabled
    assert legacy_file.with_name('config.json.pre-model-providers.bak').read_bytes() == before
    data = json.loads(legacy_file.read_text(encoding='utf-8'))
    assert len(data['model_providers']) == 2
    assert all('api_key' not in row and 'api_base_url' not in row for row in data['models'])

    config._config = None
    reloaded = config.get_config()
    assert reloaded.get_active_model() is None
    assert reloaded.get_review_model() is None
    assert reloaded.models[1].api_key == 'fixture-key'
    assert reloaded.knowledge_analysis.model_index is None


def test_enable_requires_verification_against_the_actual_provider(legacy_file):
    providers, models = config.get_provider_layout(config.get_config())
    revision = config.provider_layout_revision(config.get_config())
    models[-1] = models[-1].model_copy(update={'enabled': True})
    with pytest.raises(ValueError, match='验证'):
        config.save_provider_layout(revision, providers, models, set(), 'vision', lambda *_: False)
    assert not config.get_config().models[-1].enabled

    saved = config.save_provider_layout(
        revision, providers, models, set(), 'vision',
        lambda provider, model_id: (provider.api_key, model_id) == ('other-key', 'other'),
    )
    assert saved.models[-1].enabled
    assert saved.assist_model_ids == ['alias', 'vision']
    assert saved.written_exam_model_ids == ['vision']
    assert saved.models[-1].verified
    assert not saved.models[0].verified
    config._config = None
    assert config.get_config().models[-1].verified


@pytest.fixture
def client(legacy_file):
    app = FastAPI()
    app.include_router(router, prefix='/api')
    with TestClient(app) as client:
        yield client


def test_snapshot_redacts_credentials_and_lists_uses(client):
    response = client.get('/api/model-providers')
    assert response.status_code == 200
    payload = response.json()
    assert 'fixture-key' not in response.text and 'other-key' not in response.text
    assert all(provider['has_api_key'] for provider in payload['providers'])
    assert payload['usages']['vision'] == ['实时辅助', '截图答题', '笔试', '复盘', '知识分析', '通用默认模型']


def test_invalid_request_does_not_echo_the_submitted_api_key(client):
    response = client.put('/api/model-providers', json=[{'api_key': 'never-echo-this'}])
    assert response.status_code == 422
    assert 'never-echo-this' not in response.text


def test_saved_credential_is_bound_to_the_original_address(client):
    payload = client.get('/api/model-providers').json()
    payload['providers'][0]['api_base_url'] = 'https://new.example.test/v1'
    response = client.put('/api/model-providers', json=payload)
    assert response.status_code == 422
    assert config.get_config().models[0].api_base_url == 'https://api.openai.com/v1'


def test_new_scene_usage_invalidates_confirmation(client):
    payload = client.get('/api/model-providers').json()
    payload['models'][1]['enabled'] = False
    payload['confirmed_remove_ids'] = ['alias']
    config.update_config({'review_model_id': 'alias'})
    response = client.put('/api/model-providers', json=payload)
    assert response.status_code == 409
    assert config.get_config().review_model_id == 'alias'
    assert config.get_config().models[1].enabled


def test_removing_all_models_preserves_empty_configuration_after_restart(client):
    payload = client.get('/api/model-providers').json()
    payload.update(providers=[], models=[], confirmed_remove_ids=['vision', 'alias'], active_model_id=None)
    assert client.put('/api/model-providers', json=payload).status_code == 200
    config._config = None
    assert config.get_config().models == []
    assert config.get_config().active_model is None
    assert config.get_config().review_model_index is None
    assert config.get_config().assist_model_ids == []


def test_legacy_model_write_cannot_enable_a_model_after_migration(client):
    payload = client.get('/api/model-providers').json()
    assert client.put('/api/model-providers', json=payload).status_code == 200
    models = list(config.get_config().models)
    models[2] = models[2].model_copy(update={'enabled': True})
    with pytest.raises(config.ModelListChanged):
        config.update_config({'models': models})
    assert not config.get_config().models[2].enabled


def test_http_cannot_enable_model_by_claiming_verification(client):
    payload = client.get('/api/model-providers').json()
    payload['models'][-1].update(enabled=True, verified=True)
    assert client.put('/api/model-providers', json=payload).status_code == 422
    assert not config.get_config().models[-1].enabled


def test_legacy_write_cannot_remove_an_assigned_model_before_migration(legacy_file):
    models = config.get_config().models
    with pytest.raises(config.ModelListChanged, match='确认'):
        config.update_config({'models': models[1:]})
    assert config.get_config().assist_screenshot_model_id == 'vision'
    assert 'model_providers' not in json.loads(legacy_file.read_text(encoding='utf-8'))


def test_legacy_write_requires_verification_before_enabling(legacy_file):
    models = list(config.get_config().models)
    models[-1] = models[-1].model_copy(update={'enabled': True})
    with pytest.raises(config.InvalidAssistModelSelection, match='验证'):
        config.update_config({'models': models})
    assert not config.get_config().models[-1].enabled


def test_legacy_write_cannot_forward_a_saved_key_to_a_different_disabled_endpoint(legacy_file):
    models = list(config.get_config().models)
    models[-1] = models[-1].model_copy(update={'api_base_url': 'https://other.example.test'})
    with pytest.raises(config.InvalidAssistModelSelection, match='验证'):
        config.update_config({'models': models})
    assert config.get_config().models[-1].api_base_url == 'https://api.openai.com/v1'

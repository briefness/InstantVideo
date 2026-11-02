import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from pipeline.visual_assets import prepare_identity_assets, identity_reference_bindings
from pipeline.generator import VideoGenerator
from test_action_sequence import sequence_story


def assets(tmp_path):
    story = sequence_story()
    directory = tmp_path / 'visual_assets'
    directory.mkdir()
    records = []
    for name, color in [('attacker', 'black'), ('defender', 'blue')]:
        path = directory / f'{name}.png'
        Image.new('RGB', (128, 128), color).save(path)
        records.append({'name': name, 'path': f'visual_assets/{name}.png',
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'version': 'visual-assets-v1', 'characters': records}
    (tmp_path / 'visual_assets.json').write_text(json.dumps(manifest))
    return story, manifest


def test_prepared_identity_available_before_first_clip_and_resume(tmp_path):
    story, _ = assets(tmp_path)
    refs = prepare_identity_assets(tmp_path, story)
    generator = VideoGenerator(str(tmp_path))
    generator.character_refs.update(refs)
    paths, role = generator._build_image_refs(story['shots'][0], None)
    assert role == 'reference_image'
    assert paths == [refs[name] for name in story['shots'][0]['characters']]
    assert prepare_identity_assets(tmp_path, story, has_submissions=True) == refs
    assert json.loads((tmp_path / 'visual_asset_binding.json').read_text())['status'] == 'prepared'


def test_numbering_follows_actual_request_order_and_partial_sets():
    refs = {'Lin': '/a.png', 'Shen': '/b.png'}
    text = identity_reference_bindings(['/b.png', '/a.png'], 'reference_image', refs)
    assert 'Image 1 identifies character "Shen"' in text
    assert 'Image 2 identifies character "Lin"' in text
    assert 'Image 1 identifies character "Shen"' in identity_reference_bindings(['/b.png'], 'reference_image', refs)
    assert identity_reference_bindings(['/a.png'], 'first_frame', refs) == ''


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'unknown', 'duplicate', 'traversal', 'symlink', 'bad_image'])
def test_invalid_supplied_assets_fail_before_any_provider_call(tmp_path, mutation):
    story, manifest = assets(tmp_path)
    item = manifest['characters'][0]
    path = tmp_path / item['path']
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'changed':
        path.write_bytes(b'changed')
    elif mutation == 'unknown':
        item['name'] = 'not-in-story'
    elif mutation == 'duplicate':
        manifest['characters'][1].update(path=item['path'], sha256=item['sha256'])
    elif mutation in {'traversal', 'symlink'}:
        outside = tmp_path / 'outside.png'
        Image.new('RGB', (128, 128)).save(outside)
        if mutation == 'traversal':
            item['path'] = 'visual_assets/../outside.png'
        else:
            path.unlink()
            path.symlink_to(outside)
    elif mutation == 'bad_image':
        path.write_bytes(b'not an image')
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'visual_assets.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare_identity_assets(tmp_path, story)
    assert not (tmp_path / 'visual_asset_binding.json').exists()


def test_bootstrap_is_explicit_and_cannot_gain_assets_silently(tmp_path):
    story = sequence_story()
    assert prepare_identity_assets(tmp_path, story) == {}
    binding = json.loads((tmp_path / 'visual_asset_binding.json').read_text())
    assert binding['status'] == 'text_bootstrap'
    assert set(binding['unbound_characters']) == {'attacker', 'defender'}
    assets(tmp_path)
    with pytest.raises(ValueError, match='不能在恢复时'):
        prepare_identity_assets(tmp_path, story)


def test_removing_or_replacing_manifest_cannot_change_frozen_identity(tmp_path):
    story, manifest = assets(tmp_path)
    prepare_identity_assets(tmp_path, story)
    source = tmp_path / 'visual_assets.json'
    source.unlink()
    with pytest.raises(ValueError, match='不能在恢复时'):
        prepare_identity_assets(tmp_path, story)
    first, second = manifest['characters']
    first['name'], second['name'] = second['name'], first['name']
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='不能在恢复时'):
        prepare_identity_assets(tmp_path, story)


def test_legacy_pending_work_cannot_be_assigned_new_identities(tmp_path):
    story, _ = assets(tmp_path)
    with pytest.raises(ValueError, match='已有视频任务'):
        prepare_identity_assets(tmp_path, story, has_submissions=True)


@pytest.mark.asyncio
async def test_real_generator_compiles_numbered_identity_bindings(tmp_path, monkeypatch):
    story, _ = assets(tmp_path)
    generator = VideoGenerator(str(tmp_path))
    generator.semantic_reviewer = None
    submitted = []
    async def generate(**kwargs):
        submitted.append(kwargs)
        return {'status': 'failed', 'error_type': 'authentication', 'error': 'offline-stop'}
    monkeypatch.setattr(generator.api, 'generate', generate)
    with pytest.raises(RuntimeError, match='生成失败'):
        await generator.generate_all(story)
    assert len(submitted) == 1
    assert len(submitted[0]['image_urls']) == 2
    assert 'Image 1 identifies character "attacker"' in submitted[0]['prompt']
    assert 'Image 2 identifies character "defender"' in submitted[0]['prompt']


@pytest.mark.asyncio
async def test_asset_disappearing_after_preparation_cannot_fall_back_to_text(tmp_path):
    story, _ = assets(tmp_path)
    refs = prepare_identity_assets(tmp_path, story)
    generator = VideoGenerator(str(tmp_path))
    generator.character_refs.update(refs)
    Path(refs['attacker']).unlink()
    with pytest.raises(ValueError):
        await generator._generate_single_shot(story['shots'][0], None, None, story)


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [{'error_type': 'privacy'}, {'error': 'invalid image format'}])
async def test_supplied_identity_cannot_be_removed_by_automatic_fallback(tmp_path, monkeypatch, error):
    story, _ = assets(tmp_path)
    generator = VideoGenerator(str(tmp_path))
    generator.semantic_reviewer = None
    requests = []
    async def generate(**kwargs):
        requests.append(kwargs)
        return {'status': 'failed', **error}
    monkeypatch.setattr(generator.api, 'generate', generate)
    with pytest.raises(RuntimeError, match='生成失败'):
        await generator.generate_all(story)
    assert len(requests) == 1
    assert len(requests[0]['image_urls']) == 2

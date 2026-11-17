"""时序来源与像素绑定的离线检验，不把替身判定当作模型准确率。"""
from copy import deepcopy
import subprocess

import pytest

from pipeline.temporal_evidence import sample_clip, validate_bundle, validate_motion, combine


@pytest.fixture(scope='module')
def opposite_motion(tmp_path_factory):
    directory = tmp_path_factory.mktemp('opposite-motion')
    paths = []
    for index, trajectory in enumerate(('17.6*t', '88-17.6*t')):
        path = directory / f'{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'color=c=black:s=96x64:r=12:d=3', '-f', 'lavfi', '-i',
            'color=c=white:s=8x8:r=12:d=3', '-filter_complex',
            f"[0:v][1:v]overlay=x='if(lt(t,2.5),{trajectory},44)':y=28:shortest=1",
            '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', str(path)], check=True)
        paths.append(path)
    return paths


def test_same_endpoint_preserves_opposite_histories(opposite_motion):
    bundles = [sample_clip(str(path), {}, scope='previous_tail') for path in opposite_motion]
    frames = [b['manifest']['sources'][0]['frames'] for b in bundles]
    assert frames[0][-1]['sha256'] == frames[1][-1]['sha256']
    assert [f['sha256'] for f in frames[0][:-1]] != [f['sha256'] for f in frames[1][:-1]]
    assert all(len(f) == 9 for f in frames)
    assert all(validate_bundle(b) == b['manifest'] for b in bundles)


def test_history_ends_at_exported_prefix_not_discarded_source_tail(opposite_motion):
    bundle = sample_clip(str(opposite_motion[0]), {'media_selection': 'planned-prefix-v1', 'duration': 2},
                         scope='previous_tail')
    assert bundle['manifest']['sources'][0]['end_seconds'] == 1.95


@pytest.mark.parametrize('mutation', ['pixels', 'label', 'order', 'scope', 'digest', 'index'])
def test_invalid_evidence_is_rejected(opposite_motion, mutation):
    bundle = sample_clip(str(opposite_motion[0]), {}, scope='previous_tail')
    source = bundle['manifest']['sources'][0]
    if mutation == 'pixels':
        bundle['content'][1]['image_url']['url'] = 'data:image/jpeg;base64,YmFk'
    elif mutation == 'label':
        bundle['content'][0]['text'] = '{}'
    elif mutation == 'order':
        source['frames'].reverse()
    elif mutation == 'scope':
        bundle['manifest']['sources'].append(deepcopy(source))
    elif mutation == 'digest':
        source['video_sha256'] = ''
    else:
        source['frames'][0]['index'] = True
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_motion_cannot_cite_current_clip_or_one_frame(opposite_motion):
    current = sample_clip(str(opposite_motion[0]), {}, scope='current_clip')
    with pytest.raises(ValueError):
        validate_motion({'summary': 'moving', 'sample_indices': [1, 2]}, current['manifest'])
    prior = sample_clip(str(opposite_motion[0]), {}, scope='previous_tail')
    with pytest.raises(ValueError):
        validate_motion({'summary': 'moving', 'sample_indices': [1]}, prior['manifest'])
    assert len(combine(prior, current)['manifest']['sources']) == 2

"""Validate prepared identity assets before video submission and freeze their binding.

This is local input preparation, not an image-generation service. Absence is
reported as text bootstrap; supplied but invalid assets never silently disappear.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from pipeline.run_state import RunWorkspace


class IdentityAsset(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


class VisualAssetManifest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: str = Field(pattern=r'^visual-assets-v1$')
    characters: list[IdentityAsset] = Field(max_length=16)


def prepare_identity_assets(workspace: Path, storyboard: dict, *, has_submissions: bool = False) -> dict[str, str]:
    workspace = Path(workspace).resolve()
    manifest_path = workspace / 'visual_assets.json'
    binding_path = workspace / 'visual_asset_binding.json'
    catalog = {c['name'] for c in storyboard.get('characters', []) if c.get('reference_mode', 'identity') == 'identity'}
    assets = []
    if manifest_path.exists():
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 64000:
            raise ValueError('视觉素材清单必须是工作区内的小型 JSON 文件')
        assets = VisualAssetManifest.model_validate_json(manifest_path.read_text(encoding='utf-8')).characters
    refs, hashes, records = {}, set(), []
    for asset in assets:
        if asset.name not in catalog or asset.name in refs:
            raise ValueError('视觉素材包含未知、重复或非独立身份的角色')
        candidate = workspace / asset.path
        path = candidate.resolve()
        if (Path(asset.path).is_absolute() or not path.is_relative_to(workspace / 'visual_assets')
                or not path.is_file() or path.stat().st_size > 10 * 1024 * 1024):
            raise ValueError('身份图片必须位于工作区 visual_assets 内，且不超过 10 MiB')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != asset.sha256 or digest in hashes:
            raise ValueError('身份图片内容已变化，或同一图片被绑定给多个角色')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    if (image.format not in {'PNG', 'JPEG'} or min(image.size) < 64
                            or image.width * image.height > 16_000_000):
                        raise ValueError('图片格式或尺寸不适合身份参考')
                    image.verify()
        except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError('身份图片不可解码，未提交视频') from exc
        refs[asset.name] = str(path)
        hashes.add(digest)
        records.append({'name': asset.name, 'path': path.relative_to(workspace).as_posix(), 'sha256': digest})
    binding = {'version': 'visual-assets-v1', 'status': 'prepared' if refs else 'text_bootstrap',
               'characters': sorted(records, key=lambda item: item['name']),
               'unbound_characters': sorted(catalog - refs.keys())}
    if binding_path.exists():
        if binding_path.is_symlink() or binding_path.stat().st_size > 64000:
            raise ValueError('视觉素材绑定记录无效')
        if json.loads(binding_path.read_text(encoding='utf-8')) != binding:
            raise ValueError('视觉素材绑定与原运行不一致，不能在恢复时换人或移除参考')
    elif refs and (has_submissions or any((workspace / 'shots').glob('*_generation.json'))):
        raise ValueError('已有视频任务缺少对应视觉素材来源；不能追加入旧任务')
    else:
        RunWorkspace._atomic_write_json(binding_path, binding)
    return refs


def identity_reference_bindings(paths: list[str], role: str | None, character_refs: dict[str, str]) -> str:
    """Bind actual request order, including partial reference sets, to stable IDs."""
    if role != 'reference_image':
        return ''
    bindings = []
    for index, path in enumerate(paths, 1):
        names = [name for name, ref in character_refs.items() if ref == path]
        if len(names) > 1:
            raise ValueError('同一参考图不能同时承担不同人物的身份')
        if names:
            # IDs are data, not extra prompt instructions.
            bindings.append(f'Image {index} identifies character {json.dumps(names[0], ensure_ascii=False)}')
    return ('[Identity binding — ' + '; '.join(bindings)
            + '; each image controls only its named character, never merge or exchange identities.] ') if bindings else ''

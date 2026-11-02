"""Import an explicitly attributed action script into the normal production flow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pipeline.action_authoring import compile_authored_storyboard
from pipeline.models import DirectorIntentSpec, RunOptions
from pipeline.production_plan import build_production_plan
from pipeline.readiness import ensure_storyboard_ready
from pipeline.run_state import RunWorkspace
from pipeline.storyboard import _compile_storyboard_contract
from pipeline.storyboard_authoring import StoryIdentity
from tools.motion_previs import read_previs_json


class AuthoredInput(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    authored_by: str = Field(min_length=1, max_length=200)
    duration_seconds: int = Field(strict=True, ge=4, le=600)
    intent: DirectorIntentSpec
    storyboard: dict


def prepare_authored_run(source: dict, output_root: Path, *, paid_take_budget: int = 0) -> RunWorkspace:
    """Offline import only; main --resume still runs every admission and video check."""
    authored = AuthoredInput.model_validate(source)
    identity = StoryIdentity.model_validate({k: v for k, v in authored.storyboard.items() if k != 'shots'})
    if not authored.authored_by.strip():
        raise ValueError('必须注明动作稿作者')
    names = [c.name for c in identity.characters]
    if len(names) != len(set(names)):
        raise ValueError('角色 ID 必须唯一')
    options = RunOptions(request=authored.intent.source_request, resolution='720p', aspect_ratio='16:9',
                         style='cinematic', paid_take_budget=paid_take_budget, take_policy='completion_first')
    plan = build_production_plan(authored.intent.content_focus, authored.duration_seconds,
                                 intent=authored.intent.model_dump())
    compiled = compile_authored_storyboard(authored.storyboard, plan)
    storyboard = _compile_storyboard_contract(compiled, options.aspect_ratio, options.resolution,
                                               options.style, production_plan=plan)
    ensure_storyboard_ready(storyboard)
    snapshot = json.dumps(source, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    workspace = RunWorkspace.create(output_root, options)
    try:
        workspace._atomic_write_json(workspace.path / 'authored_input.json', source)
        workspace._atomic_write_json(workspace.path / 'authorship.json', {
            'authored_by': authored.authored_by, 'mode': 'supplied_action_script',
            'source_sha256': hashlib.sha256(snapshot).hexdigest(),
            'semantic_admission': 'pending',
        })
        workspace.save_storyboard(storyboard)
    except Exception:
        workspace.mark_failed('动作稿导入失败；未提交任何任务')
        raise
    return workspace


def main() -> int:
    parser = argparse.ArgumentParser(description='导入有署名的动作稿；离线校验，不调用模型、不生成视频')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('output'))
    parser.add_argument('--paid-take-budget', type=int, default=0)
    args = parser.parse_args()
    try:
        workspace = prepare_authored_run(read_previs_json(args.source), args.output_root,
                                        paid_take_budget=args.paid_take_budget)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'动作稿未导入：{exc}\n')
    print(f'动作稿已导入，尚未完成语义审查，未提交视频任务：{workspace.path}')
    print(f'下一步：python main.py --resume {workspace.path} --plan-only')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

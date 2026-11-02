"""Offline, source-bound control clips and timing measurements. No model calls."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.motion_experiment import file_digest, write_new_json
from pipeline.temporal_diagnostics import measure_frames
from tools.frame_extractor import check_video_quality


def read_frames(source: Path) -> tuple[list, list, float]:
    if not source.is_file() or source.stat().st_size > 256 * 1024 * 1024:
        raise ValueError('输入必须是小于256MiB的本地视频文件')
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(fps) or not 1 <= fps <= 120 or not 2 <= count <= 900:
            raise ValueError('诊断支持2–900帧、1–120fps的短片，不截断长视频')
        frames, timestamps = [], []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(frames) >= 900:
                raise ValueError('解码帧数超过诊断上限')
            h, w = frame.shape[:2]
            scale = min(1., 320 / max(h, w))
            size = (max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if (not np.isfinite(timestamp) or timestamp < 0
                    or (timestamps and timestamp <= timestamps[-1])):
                raise ValueError('缺少可验证的递增帧时间，不能用推算时间冒充实际时间')
            frames.append(frame)
            timestamps.append(timestamp)
        if len(frames) != int(count) or len(frames) < 2:
            raise ValueError('实际解码帧数与容器记录不一致')
        return frames, timestamps, fps
    finally:
        capture.release()


def controls(count: int) -> dict:
    if type(count) is not int or not 12 <= count <= 900:
        raise ValueError('对照构造需要12–900帧')
    start, length = count // 2, max(2, count // 5)
    base = list(range(count))
    freeze = [start if start <= i < start + length else i for i in base]
    jump = base[:start] + base[start + length:]
    return {'baseline': {'frames': base, 'known_change': 'none; original quality is unlabelled'},
            'inserted_hold': {'frames': freeze, 'known_change': 'inserted frozen interval; not a new generated take'},
            'removed_interval': {'frames': jump, 'known_change': 'removed interval and joined nonadjacent frames'},
            'camera_only': {'frames': [start] * count,
                            'known_change': 'one fixed source image translated; no independent subject motion'}}


def encode_control(path: Path, frames: list, mapping: dict, fps: float, camera_only: bool) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    try:
        if not writer.isOpened():
            raise RuntimeError('本地视频编码器不可用')
        for i, index in enumerate(mapping['frames']):
            frame = frames[index]
            if camera_only:
                matrix = np.float32([[1, 0, w * .06 * i / max(1, len(mapping['frames']) - 1)], [0, 1, 0]])
                frame = cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    finally:
        writer.release()


def build_audit(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.absolute()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('输出须为已有父目录下的新目录，不覆盖任何原片或旧报告')
    source_hash = file_digest(source)
    frames, timestamps, fps = read_frames(source)
    mappings = controls(len(frames))
    reports = {}
    with tempfile.TemporaryDirectory(prefix='.performance-audit-', dir=output.parent) as temp:
        staging = Path(temp)
        shutil.copyfile(source, staging / 'original.mp4')
        # Keep originals at native rate. All synthetic controls use the same
        # bounded resolution/codec; their time maps disclose every source frame.
        for name, mapping in mappings.items():
            destination = staging / f'{name}.mp4'
            encode_control(destination, frames, mapping, fps, name == 'camera_only')
            rendered, times, _ = read_frames(destination)
            diagnostic = measure_frames([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in rendered], times)
            reports[name] = {'known_change': mapping['known_change'], 'source_frames': mapping['frames'],
                'sha256': file_digest(destination), 'technical_check': check_video_quality(str(destination)),
                'diagnostic': diagnostic, 'semantic_judgment': 'unreviewed'}
        source_diagnostic = measure_frames([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], timestamps)
        # All source frames, not just the four-per-second semantic sample. Pages
        # allow timing checks without pretending contact sheets are full-speed viewing.
        pages = []
        width, height = frames[0].shape[1], frames[0].shape[0]
        for start in range(0, len(frames), 24):
            sheet = Image.new('RGB', (width * 6, (height + 24) * 4), '#141923')
            draw = ImageDraw.Draw(sheet)
            for local, index in enumerate(range(start, min(start + 24, len(frames)))):
                x, y = local % 6 * width, local // 6 * (height + 24)
                sheet.paste(Image.fromarray(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)), (x, y))
                draw.text((x + 6, y + height + 4), f'frame {index} | {timestamps[index]:.3f}s', fill='white')
            name = f'frames_{start:04d}.jpg'
            sheet.save(staging / name, quality=90)
            pages.append(name)
        report = {'version': 'performance-audit-v1', 'source_sha256': source_hash,
                  'source_name': source.name, 'source_timestamps': timestamps,
                  'source_technical_check': check_video_quality(str(source)),
                  'source_diagnostic': source_diagnostic, 'controls': reports,
                  'remote_calls': 0, 'paid_submissions': 0,
                  'reviewer_calibration': 'not_established',
                  'note': 'Control labels describe editing operations, not expert aesthetic ground truth. No production verdict is changed.'}
        if file_digest(source) != source_hash or file_digest(staging / 'original.mp4') != source_hash:
            raise ValueError('原片在诊断期间发生变化，不能发布报告')
        write_new_json(staging / 'audit.json', report)
        cards = ''.join(f'<article><h2>{html.escape(name)}</h2><video controls preload="metadata" src="{name}.mp4"></video>'
                        f'<p>{html.escape(value["known_change"])}</p></article>' for name, value in reports.items())
        page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>动作时序离线复盘</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:16px;background:#121720;color:#e7edf3}'
                'video{width:100%;max-width:720px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}'
                'img{width:100%}a{color:#8cd8ff}</style><h1>动作时序离线复盘</h1>'
                '<p>先以原速看片，再逐帧检查。画面变化不等于人物交手；相机补偿残差也不是流畅度分数。</p>'
                '<video controls preload="metadata" src="original.mp4"></video>'
                '<p>本页对照由同一原片构造，答案仅说明编辑操作，不代表专业动作评价。没有运行远端验收或新增生成。</p>'
                '<p>分别记录：动作是否发生、衔接是否连贯、节奏是否符合意图、证据是否足够。不要因为姿态变化判错，也不要因为镜头在动判定交手。</p>'
                f'<section>{cards}</section><p><a href="audit.json">逐帧测量、原始哈希与变换来源</a></p>'
                + ''.join(f'<details><summary>原片逐帧 {i + 1}</summary><img alt="逐帧时间标注" src="{name}"></details>' for i, name in enumerate(pages))
                + '</html>')
        (staging / 'index.html').write_text(page, encoding='utf-8')
        # Exclusive publication; source workspace and ledgers are never opened for writing.
        output.mkdir()
        published = []
        try:
            for file in staging.iterdir():
                destination = output / file.name
                os.link(file, destination)
                published.append(destination)
        except BaseException:
            for file in published:
                file.unlink()
            output.rmdir()
            raise
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='本地原片与受控反例复盘，不调用模型、不修改验收')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_audit(args.video, args.output)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f'离线复盘未完成：{exc}')
        return 2
    print(json.dumps({'output': str(args.output.resolve()), 'source_sha256': report['source_sha256'],
                      'remote_calls': 0, 'reviewer_calibration': report['reviewer_calibration']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

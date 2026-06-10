"""主编排器 — 7 Stage 顺序执行的完整流水线"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

import config
from pipeline.storyboard import generate_storyboard
from pipeline.generator import VideoGenerator, ShotResult
from tools import ffmpeg_ops, beat_analyzer
from tools.tts import synthesize_voiceover, TTSSegment

console = Console()


class VideoPipeline:
    """Seedance 电影级视频生成流水线"""

    def __init__(
        self,
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
        style: str = "cinematic",
        music_path: str | None = None,
        platforms: list[str] | None = None,
    ):
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.style = style
        self.music_path = music_path
        self.platforms = platforms or ["youtube", "tiktok"]

        # 工作目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace = Path(config.OUTPUT_DIR) / timestamp
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def run(self, user_request: str) -> str:
        """执行完整流水线, 返回最终视频路径"""

        console.print(Panel(
            f"[bold]🎬 Seedance 电影级视频流水线[/bold]\n\n{user_request}",
            title="开始", border_style="cyan",
        ))

        # ─── Stage 1: 分镜生成 ───
        console.print("\n[bold cyan]📋 Stage 1: 生成分镜脚本...[/bold cyan]")
        target_dur = self._parse_duration(user_request)
        storyboard = generate_storyboard(
            user_request=user_request,
            target_duration=target_dur,
            aspect_ratio=self.aspect_ratio,
            resolution=self.resolution,
            style=self.style,
        )
        self._save_json("storyboard.json", storyboard)
        console.print(f"   ✓ {len(storyboard['shots'])} 个镜头, 风格: {storyboard['mood']}, 目标时长: {target_dur}s")

        # ─── Stage 1.5: 音乐卡点 (可选) ───
        music_analysis = None
        if self.music_path:
            console.print("\n[bold cyan]🎵 Stage 1.5: 音乐节拍分析...[/bold cyan]")
            music_analysis = beat_analyzer.analyze_beats(self.music_path)
            console.print(f"   ✓ BPM: {music_analysis['bpm']:.0f}, 节拍数: {len(music_analysis['beat_times'])}")

            # 对齐镜头时长到节拍
            aligned = beat_analyzer.align_durations_to_beats(
                music_analysis, len(storyboard["shots"])
            )
            for i, shot in enumerate(storyboard["shots"]):
                shot["duration"] = aligned[i]
            self._save_json("storyboard_aligned.json", storyboard)

        # ─── Stage 2: 视频生成 ───
        console.print("\n[bold cyan]🎥 Stage 2: 生成视频片段...[/bold cyan]")
        generator = VideoGenerator(str(self.workspace))
        results = await generator.generate_all(storyboard)
        self._save_json("generation_results.json", [r.__dict__ for r in results])

        success_results = [r for r in results if r.status == "success"]
        console.print(
            f"   ✓ 成功 {len(success_results)}/{len(results)} 个镜头"
        )
        if not success_results:
            raise RuntimeError("所有镜头生成失败!")

        # ─── Stage 2.5: 规格统一 ───
        console.print("\n[bold cyan]🎞️ Stage 2.5: 统一视频规格...[/bold cyan]")
        norm_dir = self.workspace / "normalized"
        norm_dir.mkdir(exist_ok=True)

        video_files = []
        res_map = {"1080p": "1920:1080", "720p": "1280:720", "480p": "854:480"}
        target_res = res_map.get(self.resolution, "1920:1080")

        for r in success_results:
            norm_path = str(norm_dir / Path(r.local_path).name)
            ffmpeg_ops.normalize_video(r.local_path, norm_path, resolution=target_res)
            video_files.append(norm_path)
        console.print(f"   ✓ {len(video_files)} 个视频已统一规格")

        # ─── Stage 3: 拼接 + 转场 ───
        console.print("\n[bold cyan]🔗 Stage 3: 拼接 & 转场...[/bold cyan]")
        # 根据相邻镜头运动方向智能推导转场类型 + 时长
        used_shots = storyboard["shots"][:len(video_files)]
        transitions = ffmpeg_ops.infer_transitions(used_shots)
        self._transitions = transitions  # 保存供后续口播时间计算使用
        t_info = ", ".join(f"{t[0]}({t[1]:.1f}s)" for t in transitions)
        console.print(f"   转场: {t_info}")
        concat_path = str(self.workspace / "concat.mp4")
        ffmpeg_ops.concat_with_transitions(video_files, transitions, concat_path)
        console.print(f"   ✓ 拼接完成 ({ffmpeg_ops.get_video_duration(concat_path):.1f}s)")

        # ─── Stage 4: 音频处理 (BGM + 口播) ───
        # 音频处理前置: 全程 -c:v copy, 不损耗视频画质
        console.print("\n[bold cyan]🔊 Stage 4: 音频处理...[/bold cyan]")
        audio_path = str(self.workspace / "with_audio.mp4")
        music_file = self.music_path

        # 如果没指定音乐, 根据分镜 mood 自动匹配
        if not music_file:
            music_file = self._auto_select_music(storyboard)

        if music_file and Path(music_file).exists():
            ffmpeg_ops.add_background_music(concat_path, music_file, audio_path)
            console.print(f"   ✓ 背景音乐已添加: {Path(music_file).name}")
        else:
            audio_path = concat_path
            console.print("   - 无外部音乐, 保留 Seedance 原生音频")

        # 口播合成 + ducking
        console.print("\n[bold cyan]🎙️ Stage 4.5: 口播合成...[/bold cyan]")
        vo_segments: list[TTSSegment] = []
        try:
            vo_segments = await synthesize_voiceover(storyboard, str(self.workspace))
        except Exception as e:
            console.print(f"   ⚠ TTS 合成失败 ({e}), 跳过口播")

        vo_path = audio_path  # 默认不变
        if vo_segments:
            console.print(f"   ✓ 合成 {len(vo_segments)} 段口播")

            # 计算每段口播在成片中的起始时间 (累加镜头时长 - 转场重叠)
            shot_start_times = self._calc_shot_start_times(
                storyboard, transitions=self._transitions
            )
            vo_mix_input = []
            for seg in vo_segments:
                start = shot_start_times.get(seg.shot_id, 0.0) + 0.5  # 延迟 0.5s 开口
                vo_mix_input.append({
                    "audio_path": seg.audio_path,
                    "start_time": start,
                    "duration": seg.duration,
                })

            vo_path = str(self.workspace / "with_voiceover.mp4")
            ffmpeg_ops.mix_voiceover_with_ducking(audio_path, vo_mix_input, vo_path)
            console.print("   ✓ 口播已混入 (ducking)")
        else:
            console.print("   - 无口播内容")

        # ─── Stage 5: 调色 + 字幕 (合并为单次编码, 减少画质损失) ───
        console.print("\n[bold cyan]🎨 Stage 5: 调色 & 字幕...[/bold cyan]")

        # 准备 LUT 路径
        mood_str = storyboard.get("mood", "cinematic").lower()
        lut_name = None
        for key, value in config.MOOD_LUT_MAP.items():
            if key in mood_str:
                lut_name = value
                break
        lut_name = lut_name or "IWLTBAP Coronado - Standard.cube"
        lut_path = str(config.LUTS_DIR / lut_name)
        if not Path(lut_path).exists():
            lut_path = None
            console.print(f"   ⚠ LUT 文件不存在 ({lut_name}), 跳过调色")

        # 准备 SRT 字幕
        srt_path = str(self.workspace / "subtitles.srt")
        self._generate_srt(storyboard, srt_path, vo_segments=vo_segments)

        # 合并 LUT + 字幕为单次编码 (原来分两步: LUT→字幕, 现在合并)
        visual_path = str(self.workspace / "visual_final.mp4")
        ffmpeg_ops.apply_visual_filters(
            vo_path, visual_path,
            lut_path=lut_path,
            srt_path=srt_path,
        )
        if lut_path:
            console.print(f"   ✓ LUT 调色: {lut_name}")
        if Path(srt_path).exists() and Path(srt_path).stat().st_size > 10:
            console.print("   ✓ 字幕已烧录")
        console.print("   ✓ 视觉滤镜合并完成 (单次编码)")

        # ─── Stage 7: 片头片尾 + 多平台导出 ───
        console.print("\n[bold cyan]📦 Stage 7: 包装 & 导出...[/bold cyan]")

        # 片头
        title_path = str(self.workspace / "title.mp4")
        ffmpeg_ops.generate_title_card(
            title=storyboard.get("title", ""),
            output_path=title_path,
        )

        # 最终合成
        final_path = str(self.workspace / "final.mp4")
        ffmpeg_ops.concat_simple([title_path, visual_path], final_path)
        validation_reports = {
            "final": ffmpeg_ops.validate_publish_ready(final_path),
            "exports": {},
        }
        final_duration = validation_reports["final"]["duration"]
        console.print("   ✓ final.mp4 发布校验通过")

        # 多平台导出
        exports_dir = self.workspace / "exports"
        exports_dir.mkdir(exist_ok=True)
        for platform in self.platforms:
            export_path = str(exports_dir / f"{platform}.mp4")
            ffmpeg_ops.export_for_platform(final_path, platform, export_path)
            validation_reports["exports"][platform] = ffmpeg_ops.validate_publish_ready(
                export_path,
                platform=platform,
                expected_duration=final_duration,
            )
            console.print(f"   📤 {platform}: {export_path}")
            console.print("      ✓ 发布校验通过")

        self._save_json("publish_validation.json", validation_reports)

        # ─── 完成 ───
        duration = final_duration
        console.print(Panel(
            f"[bold green]✅ 成片输出[/bold green]\n\n"
            f"  路径: {final_path}\n"
            f"  时长: {duration:.1f}s\n"
            f"  分辨率: {self.resolution}\n"
            f"  导出平台: {', '.join(self.platforms)}",
            title="完成", border_style="green",
        ))

        return final_path

    def _generate_srt(
        self,
        storyboard: dict,
        output_path: str,
        vo_segments: list[TTSSegment] | None = None,
    ):
        """根据分镜生成 SRT 字幕

        如果有 TTS 口播, 用语音真实时长控制字幕显示时长 (与语音同步);
        如果没有, 退回按镜头时长机械计算 (向下兼容)。
        """
        # 建立 shot_id → TTS 时长的映射
        vo_dur_map: dict[int, float] = {}
        if vo_segments:
            for seg in vo_segments:
                vo_dur_map[seg.shot_id] = seg.duration

        shot_starts = self._calc_shot_start_times(
            storyboard, transitions=getattr(self, '_transitions', None)
        )
        lines = []
        idx = 1

        for shot in storyboard["shots"]:
            text = shot.get("subtitle_text", "")
            if not text:
                continue

            base_start = shot_starts.get(shot["shot_id"], 0.0)
            start = base_start + 0.5  # 延迟 0.5s 出现

            if shot["shot_id"] in vo_dur_map:
                # 有口播 → 字幕跟语音时长走
                end = start + vo_dur_map[shot["shot_id"]]
            else:
                # 无口播 → 退回镜头时长 (兼容旧逻辑)
                end = base_start + shot["duration"] - 0.5

            lines.append(f"{idx}")
            lines.append(f"{self._fmt_time(start)} --> {self._fmt_time(end)}")
            lines.append(text)
            lines.append("")
            idx += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _calc_shot_start_times(
        self, storyboard: dict, transitions: list[tuple[str, float]] | None = None,
    ) -> dict[int, float]:
        """计算每个镜头在成片中的起始时间

        考虑 xfade 转场的时间重叠: 转场会让相邻镜头重叠 transition_duration,
        所以实际起始时间 = sum(prev_durations) - sum(prev_transition_overlaps)
        """
        result = {}
        t = 0.0
        shots = storyboard["shots"]
        for i, shot in enumerate(shots):
            result[shot["shot_id"]] = t
            t += shot["duration"]
            # 减去与下一镜头的转场重叠时长
            if transitions and i < len(transitions):
                t -= transitions[i][1]  # transitions[i] = (type, duration)
        return result

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _save_json(self, filename: str, data):
        path = self.workspace / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _auto_select_music(self, storyboard: dict) -> str | None:
        """根据分镜 mood/music_style 自动匹配本地音乐"""
        music_style = storyboard.get("music_style", "").lower()
        mood = storyboard.get("mood", "").lower()

        # 在 MUSIC_LIBRARY 中模糊匹配
        for key, filename in config.MUSIC_LIBRARY.items():
            if key in music_style or key in mood:
                full_path = str(config.MUSIC_DIR / filename)
                if Path(full_path).exists():
                    return full_path

        # 默认: 用 cinematic
        default = config.MUSIC_LIBRARY.get("cinematic")
        if default:
            full_path = str(config.MUSIC_DIR / default)
            if Path(full_path).exists():
                return full_path

        return None

    @staticmethod
    def _parse_duration(user_request: str) -> int:
        """从用户请求中解析目标时长 (秒)

        支持: "15秒", "15s", "30秒的", "1分钟", "2min" 等
        无法解析时默认 30 秒
        """
        import re

        # 秒: "15秒", "15s", "15 秒"
        m = re.search(r'(\d+)\s*(?:秒|s(?:ec(?:ond)?s?)?)', user_request, re.IGNORECASE)
        if m:
            return max(5, min(int(m.group(1)), 120))

        # 分钟: "1分钟", "2min"
        m = re.search(r'(\d+)\s*(?:分钟|min(?:ute)?s?)', user_request, re.IGNORECASE)
        if m:
            return max(5, min(int(m.group(1)) * 60, 120))

        return 30  # 默认

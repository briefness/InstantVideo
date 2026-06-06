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
        storyboard = generate_storyboard(
            user_request=user_request,
            target_duration=30,
            aspect_ratio=self.aspect_ratio,
            resolution=self.resolution,
            style=self.style,
        )
        self._save_json("storyboard.json", storyboard)
        console.print(f"   ✓ {len(storyboard['shots'])} 个镜头, 风格: {storyboard['mood']}")

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
        transitions = [
            s.get("transition_to_next", "crossfade")
            for s in storyboard["shots"][:len(video_files)]
        ]
        concat_path = str(self.workspace / "concat.mp4")
        ffmpeg_ops.concat_with_transitions(video_files, transitions, concat_path)
        console.print(f"   ✓ 拼接完成 ({ffmpeg_ops.get_video_duration(concat_path):.1f}s)")

        # ─── Stage 4: 调色 ───
        console.print("\n[bold cyan]🎨 Stage 4: 调色...[/bold cyan]")
        graded_path = str(self.workspace / "graded.mp4")
        
        # mood 可能是多个词 (如 "warm premium inviting"), 逐词匹配
        mood_str = storyboard.get("mood", "cinematic").lower()
        lut_name = None
        for key, value in config.MOOD_LUT_MAP.items():
            if key in mood_str:
                lut_name = value
                break
        lut_name = lut_name or "IWLTBAP Coronado - Standard.cube"
        
        lut_path = str(config.LUTS_DIR / lut_name)

        if Path(lut_path).exists():
            ffmpeg_ops.apply_lut(concat_path, lut_path, graded_path)
            console.print(f"   ✓ 已应用 LUT: {lut_name}")
        else:
            graded_path = concat_path
            console.print(f"   ⚠ LUT 文件不存在 ({lut_name}), 跳过调色")

        # ─── Stage 5: 音频 ───
        console.print("\n[bold cyan]🔊 Stage 5: 音频处理...[/bold cyan]")
        audio_path = str(self.workspace / "with_audio.mp4")
        music_file = self.music_path

        # 如果没指定音乐, 根据分镜 mood 自动匹配
        if not music_file:
            music_file = self._auto_select_music(storyboard)

        if music_file and Path(music_file).exists():
            ffmpeg_ops.add_background_music(graded_path, music_file, audio_path)
            console.print(f"   ✓ 背景音乐已添加: {Path(music_file).name}")
        else:
            audio_path = graded_path
            console.print("   - 无外部音乐, 保留 Seedance 原生音频")

        # ─── Stage 6: 字幕 ───
        console.print("\n[bold cyan]💬 Stage 6: 字幕...[/bold cyan]")
        srt_path = str(self.workspace / "subtitles.srt")
        subtitled_path = str(self.workspace / "subtitled.mp4")

        self._generate_srt(storyboard, srt_path)

        if Path(srt_path).stat().st_size > 10:
            ffmpeg_ops.burn_subtitles(audio_path, srt_path, subtitled_path)
            console.print("   ✓ 字幕已烧录")
        else:
            subtitled_path = audio_path
            console.print("   - 无字幕内容")

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
        ffmpeg_ops.concat_simple([title_path, subtitled_path], final_path)

        # 多平台导出
        exports_dir = self.workspace / "exports"
        exports_dir.mkdir(exist_ok=True)
        for platform in self.platforms:
            export_path = str(exports_dir / f"{platform}.mp4")
            ffmpeg_ops.export_for_platform(final_path, platform, export_path)
            console.print(f"   📤 {platform}: {export_path}")

        # ─── 完成 ───
        duration = ffmpeg_ops.get_video_duration(final_path)
        console.print(Panel(
            f"[bold green]✅ 成片输出[/bold green]\n\n"
            f"  路径: {final_path}\n"
            f"  时长: {duration:.1f}s\n"
            f"  分辨率: {self.resolution}\n"
            f"  导出平台: {', '.join(self.platforms)}",
            title="完成", border_style="green",
        ))

        return final_path

    def _generate_srt(self, storyboard: dict, output_path: str):
        """根据分镜生成 SRT 字幕"""
        lines = []
        current_time = 0.0
        idx = 1

        for shot in storyboard["shots"]:
            text = shot.get("subtitle_text", "")
            if not text:
                current_time += shot["duration"]
                continue

            start = current_time + 0.5
            end = current_time + shot["duration"] - 0.5

            lines.append(f"{idx}")
            lines.append(f"{self._fmt_time(start)} --> {self._fmt_time(end)}")
            lines.append(text)
            lines.append("")

            idx += 1
            current_time += shot["duration"]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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

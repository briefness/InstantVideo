"""主编排器 — 7 Stage 顺序执行的完整流水线"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

import config
from pipeline.storyboard import generate_storyboard
from pipeline.generator import RemoteTaskPendingError, VideoGenerator, ShotResult
from pipeline.models import RunOptions, RunStatus
from pipeline.readiness import ensure_storyboard_ready
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SemanticReviewUnavailableError
from tools import ffmpeg_ops, beat_analyzer
from tools.tts import synthesize_voiceover, TTSSegment

console = Console()


class VideoPipeline:
    """Seedance 电影级视频生成流水线"""

    def __init__(
        self,
        resolution: str = config.DEFAULT_RESOLUTION,
        aspect_ratio: str = config.DEFAULT_RATIO,
        style: str = "cinematic",
        music_path: str | None = None,
        platforms: list[str] | None = None,
        paid_take_budget: int | None = None,
        run_workspace: RunWorkspace | None = None,
    ):
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.style = style
        self.music_path = music_path
        self.platforms = platforms or ["youtube", "tiktok"]
        self.paid_take_budget = paid_take_budget
        self.run_workspace = run_workspace
        self.workspace = run_workspace.path if run_workspace else None
        self._resuming = run_workspace is not None

    @classmethod
    def from_workspace(cls, workspace: str | Path) -> "VideoPipeline":
        run_workspace = RunWorkspace.resume(workspace)
        options = run_workspace.options
        return cls(
            resolution=options.resolution,
            aspect_ratio=options.aspect_ratio,
            style=options.style,
            music_path=options.music_path,
            platforms=options.platforms,
            paid_take_budget=options.paid_take_budget,
            run_workspace=run_workspace,
        )

    async def run(self, user_request: str | None = None) -> str:
        """Execute a new or resumed one-click run and return the final video."""
        if self.run_workspace:
            stored_request = self.run_workspace.options.request
            if user_request and user_request.strip() != stored_request:
                raise ValueError("恢复运行不能修改原始视频需求")
            user_request = stored_request
        else:
            if not user_request or not user_request.strip():
                raise ValueError("视频需求不能为空")
            options = RunOptions(
                request=user_request.strip(),
                resolution=self.resolution,
                aspect_ratio=self.aspect_ratio,
                style=self.style,
                music_path=self.music_path,
                platforms=self.platforms,
                paid_take_budget=self.paid_take_budget,
            )
            self.run_workspace = RunWorkspace.create(config.OUTPUT_DIR, options)
            self.workspace = self.run_workspace.path
            user_request = options.request

        completed = self.run_workspace.completed_output()
        if completed:
            console.print(f"[green]✓ 运行已完成，复用现有成片: {completed}[/green]")
            return completed

        try:
            return await self._run(user_request)
        except RemoteTaskPendingError as exc:
            self.run_workspace.mark_interrupted(str(exc))
            raise
        except SemanticReviewUnavailableError as exc:
            self.run_workspace.mark_interrupted(str(exc))
            raise
        except asyncio.CancelledError:
            self.run_workspace.mark_interrupted()
            raise
        except Exception as exc:
            self.run_workspace.mark_failed(str(exc))
            raise

    async def _run(self, user_request: str) -> str:
        """Run pipeline stages after workspace identity has been established."""
        if self.workspace is None or self.run_workspace is None:
            raise RuntimeError("运行工作区尚未初始化")

        console.print(Panel(
            f"[bold]🎬 Seedance 电影级视频流水线[/bold]\n\n{user_request}",
            title="开始", border_style="cyan",
        ))

        # ─── Stage 1: 分镜生成 ───
        target_dur = self._parse_duration(user_request)
        if self._resuming:
            console.print("\n[bold cyan]📋 Stage 1: 恢复已确认分镜...[/bold cyan]")
            storyboard = self.run_workspace.load_storyboard()
        else:
            console.print("\n[bold cyan]📋 Stage 1: 生成分镜脚本...[/bold cyan]")
            storyboard = generate_storyboard(
                user_request=user_request,
                target_duration=target_dur,
                aspect_ratio=self.aspect_ratio,
                resolution=self.resolution,
                style=self.style,
            )
            self._save_json("storyboard_draft.json", storyboard)

            # ─── Stage 1.5: 音乐卡点 (可选) ───
            if self.music_path:
                console.print("\n[bold cyan]🎵 Stage 1.5: 音乐节拍分析...[/bold cyan]")
                music_analysis = beat_analyzer.analyze_beats(self.music_path)
                console.print(f"   ✓ BPM: {music_analysis['bpm']:.0f}, 节拍数: {len(music_analysis['beat_times'])}")
                aligned = beat_analyzer.align_durations_to_beats(
                    music_analysis, len(storyboard["shots"])
                )
                for i, shot in enumerate(storyboard["shots"]):
                    shot["duration"] = aligned[i]
                self._save_json("storyboard_aligned.json", storyboard)

            storyboard = self.run_workspace.save_storyboard(storyboard)

        console.print(f"   ✓ {len(storyboard['shots'])} 个镜头, 风格: {storyboard['mood']}, 目标时长: {target_dur}s")

        # ─── Stage 2: 视频生成 ───
        ensure_storyboard_ready(storyboard)
        unresolved_legacy_tasks = self.run_workspace.unresolved_legacy_provider_tasks()
        if unresolved_legacy_tasks:
            shot_ids = ", ".join(str(shot_id) for shot_id in sorted(unresolved_legacy_tasks))
            raise RuntimeError(
                "恢复运行包含缺少不可变提交描述的历史远端任务 "
                f"(Shot {shot_ids})；为避免重复付费生成，已停止且不会轮询或重新提交。"
            )
        terminal_materialization_tasks = self.run_workspace.terminal_materialization_tasks()
        if terminal_materialization_tasks:
            shot_ids = ", ".join(
                str(shot_id) for shot_id in sorted(terminal_materialization_tasks)
            )
            raise RuntimeError(
                "恢复运行包含已返回但本地物化或技术 QA 失败的远端镜头 "
                f"(Shot {shot_ids})；为避免重复付费生成，已停止且不会重新提交。"
            )
        console.print("\n[bold cyan]🎥 Stage 2: 生成视频片段...[/bold cyan]")
        def record_progress(result: ShotResult) -> None:
            self.run_workspace.record_shot(
                shot_id=result.shot_id,
                status=result.status,
                provider_task_id=result.provider_task_id,
                provider_error_type=result.provider_error_type,
                provider_error_code=result.provider_error_code,
                provider_error_message=result.provider_error_message,
                provider_error_locus=result.provider_error_locus,
                prompt_profile=result.prompt_profile,
                prompt_fingerprint=result.prompt_fingerprint,
                compiled_contract_version=result.compiled_contract_version,
                compiled_contract_fingerprint=result.compiled_contract_fingerprint,
                accepted_contract_version=result.accepted_contract_version,
                accepted_contract_fingerprint=result.accepted_contract_fingerprint,
                semantic_evaluator_version=result.semantic_evaluator_version,
                acceptance_policy=result.acceptance_policy,
                recovery_actions=result.recovery_actions,
                prompt_attempts=result.prompt_attempts,
                local_path=result.local_path,
                last_frame_url=result.last_frame_url,
                quality_score=result.quality_score,
                technical_quality_score=result.technical_quality_score,
                semantic_accepted=result.semantic_accepted,
                observed_end_state=result.observed_end_state,
                reference_chain_depth=result.reference_chain_depth,
                model_used=result.model_used,
                resolution_used=result.resolution_used,
                attempts=result.attempts,
                errors=result.errors,
            )

        generator = VideoGenerator(
            str(self.workspace),
            on_progress=record_progress,
            reserve_paid_take=self.run_workspace.reserve_paid_take,
            confirm_paid_take_submission=(
                self.run_workspace.confirm_paid_take_submission
            ),
            reconcile_paid_take=self.run_workspace.reconcile_paid_take,
            release_unsubmitted_paid_take=self.run_workspace.release_unsubmitted_paid_take,
            resume_tasks=self.run_workspace.resumable_pending_tasks(),
            accepted_shot_artifacts=self.run_workspace.accepted_shot_artifacts(),
        )
        results = await generator.generate_all(storyboard)
        self._save_json("generation_results.json", [r.__dict__ for r in results])

        success_results = [r for r in results if r.status == "success"]
        console.print(
            f"   ✓ 成功 {len(success_results)}/{len(results)} 个镜头"
        )
        if len(success_results) != len(results):
            failed_ids = [str(r.shot_id) for r in results if r.status != "success"]
            raise RuntimeError(
                f"镜头 {', '.join(failed_ids)} 生成失败；运行状态已保存，可使用 --resume 继续"
            )

        self.run_workspace.checkpoint("postprocessing", RunStatus.running)

        # ─── Stage 2.5: 规格统一 ───
        console.print("\n[bold cyan]🎞️ Stage 2.5: 统一视频规格...[/bold cyan]")
        norm_dir = self.workspace / "normalized"
        norm_dir.mkdir(exist_ok=True)

        video_files = []
        media_durations: dict[int, float] = {}
        target_res = config.SEEDANCE_OUTPUT_DIMENSIONS[self.resolution][
            self.aspect_ratio
        ]

        for r in success_results:
            norm_path = str(norm_dir / Path(r.local_path).name)
            ffmpeg_ops.normalize_video(r.local_path, norm_path, resolution=target_res)
            video_files.append(norm_path)
            media_durations[r.shot_id] = ffmpeg_ops.get_video_duration(norm_path)
        self._shot_media_durations = media_durations
        console.print(f"   ✓ {len(video_files)} 个视频已统一规格")

        # ─── Stage 3: 拼接 + 转场 ───
        console.print("\n[bold cyan]🔗 Stage 3: 拼接 & 转场...[/bold cyan]")
        # 根据相邻镜头运动方向智能推导转场类型 + 时长
        transitions = ffmpeg_ops.infer_transitions(storyboard["shots"])
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
                storyboard,
                transitions=self._transitions,
                media_durations=self._shot_media_durations,
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
            resolution=target_res,
        )

        # 最终合成
        final_path = str(self.workspace / "final.mp4")
        ffmpeg_ops.concat_simple([title_path, visual_path], final_path)
        expected_final_duration = (
            ffmpeg_ops.get_video_duration(title_path)
            + ffmpeg_ops.get_video_duration(visual_path)
        )
        validation_reports = {
            "final": ffmpeg_ops.validate_publish_ready(
                final_path,
                expected_duration=expected_final_duration,
            ),
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

        self.run_workspace.mark_succeeded(final_path)
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
            storyboard,
            transitions=getattr(self, '_transitions', None),
            media_durations=getattr(self, '_shot_media_durations', None),
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
                media_duration = getattr(self, '_shot_media_durations', {}).get(
                    shot["shot_id"], shot["duration"]
                )
                end = base_start + media_duration - 0.5

            lines.append(f"{idx}")
            lines.append(f"{self._fmt_time(start)} --> {self._fmt_time(end)}")
            lines.append(text)
            lines.append("")
            idx += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _calc_shot_start_times(
        self,
        storyboard: dict,
        transitions: list[tuple[str, float]] | None = None,
        media_durations: dict[int, float] | None = None,
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
            t += (
                media_durations.get(shot["shot_id"], shot["duration"])
                if media_durations
                else shot["duration"]
            )
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

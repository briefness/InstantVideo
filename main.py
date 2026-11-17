#!/usr/bin/env python3
"""Seedance 电影级视频生成工具 — CLI 入口

Usage:
    python main.py "制作一个30秒的智能手表宣传视频"
    python main.py "15秒咖啡品牌广告" --music music/upbeat.mp3
    python main.py "旅行Vlog开头" --ratio 9:16 --platforms tiktok
"""

import argparse
import asyncio
import sys

import config
from pipeline.orchestrator import VideoPipeline, MotionPreparationRequiredError
from pipeline.generator import RemoteTaskPendingError


def main():
    parser = argparse.ArgumentParser(
        description="Seedance 2.0 电影级视频自动生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "30秒智能手表宣传片，展示外观和功能"
  python main.py "科技感产品介绍" --resolution 720p --style futuristic
  python main.py "旅行短视频" --ratio 9:16 --music music/chill.mp3 --platforms tiktok
  python main.py --resume output/20260730_153000_123456
        """,
    )

    parser.add_argument("request", nargs="?", help="视频需求描述 (自然语言)")
    parser.add_argument("--resume", metavar="WORKSPACE", help="从已有 output 工作区继续")
    parser.add_argument('--authorize-paid-takes', type=int,
                        help='恢复已通过纯规划且无视频提交的工作区时，显式授权本次视频提交总上限')
    parser.add_argument("--plan-only", action="store_true", help="只生成并审查分镜，保存后停止，不提交视频任务；可用 --resume 继续")
    parser.add_argument('--no-planning-repair', dest='planning_repair', action='store_false', default=None,
                        help='规划失败立即停止，不自动修复 JSON 或重写分镜；随工作区保存')
    parser.add_argument('--take-policy', choices=['completion_first', 'single_pass'],
                        help='提交策略：默认为后续镜头保留首次额度；single_pass 每镜最多一次，失败即停')
    parser.add_argument('--delivery-mode', choices=['strict', 'reviewed', 'normal', 'preview'],
                        help='审查模式：strict（默认，兼容 reviewed）有问题即停；normal 非严重问题带警告成片；preview 为旧版未审查预览')
    parser.add_argument("--resolution", choices=config.SUPPORTED_RESOLUTIONS)
    parser.add_argument("--ratio", choices=config.SUPPORTED_ASPECT_RATIOS)
    parser.add_argument("--style",
                        help="视频风格 (cinematic/energetic/warm/cold/dramatic/futuristic)")
    parser.add_argument("--music", default=None, help="背景音乐文件路径")
    parser.add_argument("--motion-control", choices=["auto", "text_guided", "reference_motion"],
                        help="动作控制：auto 遵循需求；reference_motion 必须先准备可用动作参考")
    parser.add_argument(
        "--paid-take-budget",
        type=int,
        help="本次运行允许提交的付费 take 数量（省略则不设上限）",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=config.SUPPORTED_PLATFORMS,
        help="导出平台 (youtube tiktok bilibili instagram_reels instagram_feed)",
    )

    args = parser.parse_args()
    if args.authorize_paid_takes is not None and (not args.resume or args.plan_only):
        parser.error('--authorize-paid-takes 仅用于恢复纯规划工作区并开始生成')

    if args.resume:
        overrides = [
            args.resolution,
            args.ratio,
            args.style,
            args.music,
            args.platforms,
            args.paid_take_budget,
            args.motion_control,
            args.take_policy,
            args.delivery_mode,
            args.planning_repair,
        ]
        if any(value is not None for value in overrides):
            parser.error(
                "--resume 使用原运行参数，不能同时覆盖画幅、风格、音乐、平台、动作控制、付费 take 预算、提交策略或交付方式"
            )
    elif not args.request:
        parser.error("请提供视频需求，或使用 --resume WORKSPACE")

    try:
        if args.authorize_paid_takes is not None:
            from pipeline.run_state import RunWorkspace
            RunWorkspace.resume(args.resume).authorize_planned_generation(args.authorize_paid_takes)
        pipeline = (
            VideoPipeline.from_workspace(args.resume)
            if args.resume
            else VideoPipeline(
                resolution=args.resolution or config.DEFAULT_RESOLUTION,
                aspect_ratio=args.ratio or config.DEFAULT_RATIO,
                style=args.style or "cinematic",
                music_path=args.music,
                platforms=args.platforms or ["youtube", "tiktok"],
                paid_take_budget=args.paid_take_budget,
                motion_control=args.motion_control or "auto",
                take_policy=args.take_policy or 'completion_first',
                delivery_mode='reviewed' if args.delivery_mode in (None, 'strict') else args.delivery_mode,
                planning_repair=args.planning_repair if args.planning_repair is not None else True,
            )
        )
        pipeline.plan_only = args.plan_only
        result = asyncio.run(pipeline.run(args.request))
        if args.plan_only:
            print(f"\n📋 分镜准备完成，未提交视频任务: {result}")
        elif pipeline.delivery_mode == 'preview':
            print(f"\n🎬 预览成片已保存（人物、动作和运镜待验收）: {result}")
        elif pipeline.delivery_mode == 'normal':
            print(f"\n🎬 正常模式成片已保存，审查问题请查看交付报告: {result}")
        else:
            print(f"\n🎉 完成! 视频已保存到: {result}")
    except KeyboardInterrupt:
        print("\n\n⏹️ 已中断")
        sys.exit(1)
    except (RemoteTaskPendingError, MotionPreparationRequiredError) as e:
        print(f"\n⏸️ {e}")
        sys.exit(2)
    except Exception as e:
        from pipeline.operation_failure import failure_message
        print(f"\n❌ 错误: {failure_message(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
from pipeline.orchestrator import VideoPipeline
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
    parser.add_argument("--resolution", choices=config.SUPPORTED_RESOLUTIONS)
    parser.add_argument("--ratio", choices=config.SUPPORTED_ASPECT_RATIOS)
    parser.add_argument("--style",
                        help="视频风格 (cinematic/energetic/warm/cold/dramatic/futuristic)")
    parser.add_argument("--music", default=None, help="背景音乐文件路径")
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

    if args.resume:
        overrides = [
            args.resolution,
            args.ratio,
            args.style,
            args.music,
            args.platforms,
            args.paid_take_budget,
        ]
        if any(value is not None for value in overrides):
            parser.error(
                "--resume 使用原运行参数，不能同时覆盖画幅、风格、音乐、平台或付费 take 预算"
            )
    elif not args.request:
        parser.error("请提供视频需求，或使用 --resume WORKSPACE")

    try:
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
            )
        )
        result = asyncio.run(pipeline.run(args.request))
        print(f"\n🎉 完成! 视频已保存到: {result}")
    except KeyboardInterrupt:
        print("\n\n⏹️ 已中断")
        sys.exit(1)
    except RemoteTaskPendingError as e:
        print(f"\n⏸️ {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

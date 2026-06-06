#!/usr/bin/env python3
"""Seedance 电影级视频生成工具 — CLI 入口

Usage:
    python main.py "制作一个30秒的智能手表宣传视频"
    python main.py "15秒咖啡品牌广告" --music music/upbeat.mp3
    python main.py "旅行Vlog开头" --ratio 9:16 --platforms tiktok xiaohongshu
"""

import argparse
import asyncio
import sys

from pipeline.orchestrator import VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Seedance 2.0 电影级视频自动生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "30秒智能手表宣传片，展示外观和功能"
  python main.py "科技感产品介绍" --resolution 1080p --style futuristic
  python main.py "旅行短视频" --ratio 9:16 --music music/chill.mp3 --platforms tiktok
        """,
    )

    parser.add_argument("request", help="视频需求描述 (自然语言)")
    parser.add_argument("--resolution", default="1080p", choices=["480p", "720p", "1080p"])
    parser.add_argument("--ratio", default="16:9", choices=["16:9", "9:16", "4:3", "1:1", "21:9"])
    parser.add_argument("--style", default="cinematic",
                        help="视频风格 (cinematic/energetic/warm/cold/dramatic/futuristic)")
    parser.add_argument("--music", default=None, help="背景音乐文件路径")
    parser.add_argument("--platforms", nargs="+", default=["youtube", "tiktok"],
                        help="导出平台 (youtube tiktok bilibili xiaohongshu)")

    args = parser.parse_args()

    pipeline = VideoPipeline(
        resolution=args.resolution,
        aspect_ratio=args.ratio,
        style=args.style,
        music_path=args.music,
        platforms=args.platforms,
    )

    try:
        result = asyncio.run(pipeline.run(args.request))
        print(f"\n🎉 完成! 视频已保存到: {result}")
    except KeyboardInterrupt:
        print("\n\n⏹️ 已中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

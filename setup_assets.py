#!/usr/bin/env python3
"""素材下载脚本 — 一键获取免费 LUT 和背景音乐

Usage:
    python setup_assets.py          # 下载全部
    python setup_assets.py --luts   # 只下载 LUT
    python setup_assets.py --music  # 只下载音乐
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("需要 requests 库: pip install requests")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent
LUTS_DIR = PROJECT_ROOT / "luts"
MUSIC_DIR = PROJECT_ROOT / "music"


# ─── LUT 资源 (Pixabay CDN / 公开直链) ───
# 注: 大部分高质量 LUT 需要从网站手动下载
# 这里提供一些可直接下载的开源/免费 LUT
LUTS_TO_DOWNLOAD = {
    # 来自 FreshLUTs 和社区分享的公开 LUT
    # 如果下载失败, 手动从 README.md 中的链接获取
}

# ─── 音乐资源 (Pixabay 免费无版权音乐 API) ───
PIXABAY_MUSIC_SEARCHES = [
    {"filename": "cinematic_epic.mp3", "query": "epic cinematic orchestral", "duration_min": 60},
    {"filename": "upbeat_electronic.mp3", "query": "upbeat electronic energy", "duration_min": 30},
    {"filename": "calm_ambient.mp3", "query": "ambient calm peaceful", "duration_min": 60},
    {"filename": "corporate_modern.mp3", "query": "corporate modern technology", "duration_min": 30},
    {"filename": "acoustic_warm.mp3", "query": "acoustic warm guitar", "duration_min": 30},
]


def download_file(url: str, save_path: str, desc: str = "") -> bool:
    """下载文件并显示进度"""
    try:
        print(f"  ⬇️  下载: {desc or os.path.basename(save_path)}...", end=" ", flush=True)
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        size_kb = os.path.getsize(save_path) / 1024
        print(f"✓ ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def setup_luts():
    """下载/准备 LUT 文件"""
    print("\n🎨 准备 LUT 调色文件...\n")

    LUTS_DIR.mkdir(exist_ok=True)

    # 检查已有文件
    existing = list(LUTS_DIR.glob("*.cube"))
    if existing:
        print(f"  ℹ️  已有 {len(existing)} 个 LUT 文件:")
        for f in existing:
            print(f"     - {f.name}")
        print()

    # 生成占位文件 (帮助用户知道需要哪些)
    needed_luts = [
        ("Kodak2383_D65.cube", "电影级 (cinematic/premium)"),
        ("TealOrange.cube", "青橙对比 (energetic)"),
        ("BleachBypass.cube", "漂白跳过 (dramatic)"),
        ("WarmSunset.cube", "暖色调 (warm)"),
        ("NordicCold.cube", "北欧冷调 (cold)"),
        ("CyberpunkNeon.cube", "赛博朋克 (futuristic)"),
        ("DesaturatedDrama.cube", "低饱和 (documentary)"),
        ("VintageFilm.cube", "复古胶片 (vintage)"),
        ("TechClean.cube", "科技冷白 (modern_tech)"),
    ]

    missing = []
    for filename, desc in needed_luts:
        path = LUTS_DIR / filename
        if not path.exists():
            missing.append((filename, desc))

    if missing:
        print(f"  ⚠️  缺少 {len(missing)} 个 LUT 文件:")
        for filename, desc in missing:
            print(f"     - {filename} → {desc}")

        print(f"""
  📥 请从以下任一来源下载:

  1. IWLTBAP (推荐, 免费10个电影LUT):
     https://blog.iwltbap.com/the-10-free-cinematic-luts-by-iwltbap/

  2. Gumroad (Kodak 2393, 免费):
     https://filmborders.gumroad.com/l/Kodak2393LUT

  3. PresetPro (终极合集, 全免费):
     https://www.presetpro.com/the-ultimate-free-color-grading-lut-collection/

  4. DaVinci Resolve (安装免费版自带大量 LUT):
     macOS 路径: /Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/

  下载后将 .cube 文件重命名并放入: {LUTS_DIR}/
  
  ⚡ 如果暂时不下载, 流水线会自动跳过调色步骤, 不影响使用。
""")
    else:
        print("  ✅ 所有 LUT 文件已就绪!")


def setup_music():
    """下载免费背景音乐"""
    print("\n🎵 准备背景音乐...\n")

    MUSIC_DIR.mkdir(exist_ok=True)

    # 检查已有文件
    existing = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
    if existing:
        print(f"  ℹ️  已有 {len(existing)} 个音乐文件:")
        for f in existing:
            print(f"     - {f.name}")
        print()

    # Pixabay Music 不提供直链 API (需要网页下载)
    # 提供指引
    needed_music = [
        ("cinematic_epic.mp3", "epic cinematic orchestral"),
        ("upbeat_electronic.mp3", "upbeat electronic energy"),
        ("calm_ambient.mp3", "ambient calm peaceful"),
        ("corporate_modern.mp3", "corporate modern technology"),
        ("acoustic_warm.mp3", "acoustic warm guitar"),
    ]

    missing = []
    for filename, _ in needed_music:
        if not (MUSIC_DIR / filename).exists():
            missing.append((filename, _))

    if missing:
        print(f"  ⚠️  建议下载 {len(missing)} 个音乐文件:")
        for filename, query in missing:
            print(f"     - {filename}")

        print(f"""
  📥 从 Pixabay Music (免费, 无需署名, 可商用) 下载:

  🔗 https://pixabay.com/music/

  搜索以下关键词, 下载第一个高评分结果:
""")
        for filename, query in missing:
            print(f"     "{query}" → 保存为 {filename}")

        print(f"""
  其他免费来源:
  - Mixkit: https://mixkit.co/free-stock-music/
  - Chosic: https://www.chosic.com/free-music/cinematic/

  下载后放入: {MUSIC_DIR}/
  
  ⚡ 音乐是可选的。不提供音乐时, 流水线使用 Seedance 原生音效。
""")
    else:
        print("  ✅ 音乐文件已就绪!")


def print_final_status():
    """打印最终状态"""
    print("\n" + "=" * 60)
    print("📦 素材准备状态总览")
    print("=" * 60)

    lut_count = len(list(LUTS_DIR.glob("*.cube")))
    music_count = len(list(MUSIC_DIR.glob("*.mp3"))) + len(list(MUSIC_DIR.glob("*.wav")))

    print(f"  LUT 文件:  {lut_count} 个 {'✅' if lut_count > 0 else '⚠️ (可选, 跳过调色)'}")
    print(f"  音乐文件:  {music_count} 个 {'✅' if music_count > 0 else '⚠️ (可选, 用原生音效)'}")
    print()

    if lut_count == 0 and music_count == 0:
        print("  💡 没有素材也能跑! 流水线会自动跳过调色和外部音乐步骤。")
        print("     先试试: python main.py \"15秒咖啡广告\"")
    else:
        print("  🚀 准备就绪! 运行: python main.py \"你的视频需求\"")
    print()


def main():
    parser = argparse.ArgumentParser(description="下载/准备 LUT 和音乐素材")
    parser.add_argument("--luts", action="store_true", help="只准备 LUT")
    parser.add_argument("--music", action="store_true", help="只准备音乐")
    args = parser.parse_args()

    print("🎬 Seedance 素材准备工具")
    print("=" * 60)

    if args.luts:
        setup_luts()
    elif args.music:
        setup_music()
    else:
        setup_luts()
        setup_music()

    print_final_status()


if __name__ == "__main__":
    main()

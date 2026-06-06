"""全局配置 — 所有 API Key 和默认参数集中管理"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MUSIC_DIR = PROJECT_ROOT / "music"
LUTS_DIR = PROJECT_ROOT / "luts"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# ─── 火山引擎 Ark (统一平台) ───
ARK_API_KEY = os.getenv("ARK_API_KEY")                    # 唯一 Key, 同时调用 LLM 和视频生成
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")

# 视频生成模型
SEEDANCE_MODEL = os.getenv("SEEDANCE_MODEL", "doubao-seedance-2.0-fast")

# LLM 文本模型 (分镜生成)
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-2.0-lite")

# ─── 默认生成参数 ───
DEFAULT_RESOLUTION = "1080p"
DEFAULT_RATIO = "16:9"
DEFAULT_DURATION = 5          # 秒 (每镜头)
DEFAULT_FPS = 24
DEFAULT_GENERATE_AUDIO = True

# ─── 降级策略 ───
# 注意: doubao-seedance-2.0-fast 不支持 1080p, 最高支持 720p
DEGRADATION_CHAIN = [
    {"model": SEEDANCE_MODEL, "resolution": "720p", "max_duration": 15},
    {"model": "doubao-seedance-2.0-fast", "resolution": "480p", "max_duration": 10},
]

# ─── 调色 LUT 映射 ───
MOOD_LUT_MAP = {
    "cinematic": "IWLTBAP Coronado - Standard.cube",      # 电影感 (Kodak 风格)
    "premium": "IWLTBAP Coronado - Standard.cube",        # 高端质感
    "energetic": "Cliff-SLog3.cube",                      # 活力 (高对比)
    "dramatic": "Bat-SLog3.cube",                         # 戏剧化 (暗调)
    "warm": "Arrakis-SLog3.cube",                         # 暖色调
    "cold": "Cliff-SLog3.cube",                           # 冷色调
    "futuristic": "Bat-SLog3.cube",                       # 未来感
    "documentary": "IWLTBAP Coronado - LOG.cube",         # 纪录片
    "vintage": "Arrakis-SLog3.cube",                      # 复古
    "modern_tech": "Cliff-SLog3.cube",                    # 科技感
}
# 注: 以上映射基于你下载的 LUT 包。Seedance 输出为 Rec709 (标准色彩空间),
# 优先使用 "Standard" 或 "BMDFilm" 后缀的 LUT (非 Log 输入)。

# ─── 音乐库映射 ───
MUSIC_LIBRARY = {
    "cinematic": "tunetank-inspiring-cinematic-music-409347.mp3",
    "epic": "the_mountain-epic-508009.mp3",
    "upbeat": "jonasblakewood-upbeat-corporate-533853.mp3",
    "energetic": "jonasblakewood-upbeat-rock-524145.mp3",
    "ambient": "paulyudin-ambient-ambient-music-482398.mp3",
    "calm": "paulyudin-ambient-ambient-music-482398.mp3",
    "corporate": "jonasblakewood-upbeat-corporate-533853.mp3",
}

# ─── 并发/调度 ───
MAX_CONCURRENT_GENERATIONS = 3
GENERATION_TIMEOUT = 600       # 单个镜头最长等待 10 分钟 (I2V 比 T2V 慢, 需更多时间)
MAX_RETRIES_PER_SHOT = 3
DOWNLOAD_IMMEDIATELY = True    # 生成后立即下载 (URL 24h 过期)

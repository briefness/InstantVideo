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
ARK_API_KEY_SEEDANCE = os.getenv("ARK_API_KEY_SEEDANCE", ARK_API_KEY)  # 视频生成专用 Key (优先级高)
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")  # LLM
ARK_BASE_URL_SEEDANCE = os.getenv("ARK_BASE_URL_SEEDANCE", "https://ark.cn-beijing.volces.com/api/v3")  # 视频生成

# 视频生成模型
SEEDANCE_MODEL = os.getenv(
    "SEEDANCE_MODEL", "doubao-seedance-2-0-mini-260615"
)

# 豆包 Seed 多模态模型 (分镜生成 + 五点跨镜头语义验收)
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-2.0-lite")
SEMANTIC_REVIEW_MODEL = os.getenv("SEMANTIC_REVIEW_MODEL", LLM_MODEL)
SEMANTIC_REVIEW_ENABLED = os.getenv("SEMANTIC_REVIEW_ENABLED", "true").lower() == "true"
SEMANTIC_REVIEW_IMAGE_DETAIL = os.getenv(
    "SEMANTIC_REVIEW_IMAGE_DETAIL", "high"
).lower()

# ─── TTS ───
TTS_ENGINE = os.getenv("TTS_ENGINE", "macos").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "Tingting")
VOLCANO_TTS_API_KEY = os.getenv("VOLCANO_TTS_API_KEY")
VOLCANO_TTS_RESOURCE_ID = "seed-tts-1.0"
VOLCANO_TTS_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"

# ─── 默认生成参数 ───
DEFAULT_RESOLUTION = "480p"
DEFAULT_RATIO = "16:9"
DEFAULT_DURATION = 5          # 秒 (每镜头)
MIN_SHOT_DURATION = 4
MAX_SHOT_DURATION = 15
DEFAULT_FPS = 24
DEFAULT_GENERATE_AUDIO = True

# ─── Seedance 2.0 Mini 官方输出规格 ───
SUPPORTED_RESOLUTIONS = ("480p", "720p")
SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16", "4:3", "1:1", "3:4", "21:9")
SUPPORTED_PLATFORMS = (
    "youtube",
    "tiktok",
    "bilibili",
    "instagram_reels",
    "instagram_feed",
)

SEEDANCE_OUTPUT_DIMENSIONS = {
    "480p": {
        "16:9": "864:496",
        "4:3": "752:560",
        "1:1": "640:640",
        "3:4": "560:752",
        "9:16": "496:864",
        "21:9": "992:432",
    },
    "720p": {
        "16:9": "1280:720",
        "4:3": "1112:834",
        "1:1": "960:960",
        "3:4": "834:1112",
        "9:16": "720:1280",
        "21:9": "1470:630",
    },
}

# 默认只生成 480p；显式选择 720p 时才允许同模型降级到 480p。
GENERATION_CHAINS = {
    "480p": [
        {"model": SEEDANCE_MODEL, "resolution": "480p", "max_duration": 15},
    ],
    "720p": [
        {"model": SEEDANCE_MODEL, "resolution": "720p", "max_duration": 15},
        {"model": SEEDANCE_MODEL, "resolution": "480p", "max_duration": 15},
    ],
}

# Consecutive accepted output-tail handoffs before the plan schedules an
# intentional canonical identity re-anchor. The provider adapter only resets
# when an eligible canonical reference actually exists.
MAX_REFERENCE_CHAIN_DEPTH = 2

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

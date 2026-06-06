# 背景音乐库

将 MP3/WAV 背景音乐文件放在此目录。通过 `--music` 参数指定使用。

## 免费无版权音乐下载源

### 1. Pixabay Music (推荐首选)
- 🔗 https://pixabay.com/music/
- 完全免费, 无需署名, 可商用
- 搜索: cinematic, upbeat, ambient, corporate, epic
- 直接下载 MP3

### 2. Mixkit
- 🔗 https://mixkit.co/free-stock-music/
- 免费, 无需署名
- 分类清晰: Cinematic / Electronic / Ambient / Corporate

### 3. Chosic
- 🔗 https://www.chosic.com/free-music/cinematic/
- 免费, 可用于 YouTube / 社交媒体
- 按情绪/风格分类

### 4. Bensound
- 🔗 https://www.bensound.com/royalty-free-music/cinematic
- 部分免费 (需署名), 付费可免署名
- 质量很高, 适合品牌视频

### 5. Uppbeat
- 🔗 https://uppbeat.io/
- 免费版每月 10 首
- 100% 人工创作, 适合正式商业项目

### 6. FreeToUse
- 🔗 https://freetouse.com/blog/best-royalty-free-cinematic-background-music-for-videos-movies-trailers
- 免费电影级 BGM 合集

## 建议的音乐库结构

```
music/
├── cinematic_epic.mp3         ← 史诗/大气 (品牌大片)
├── upbeat_electronic.mp3      ← 活力电子 (科技/运动)
├── calm_ambient.mp3           ← 平静氛围 (生活方式)
├── corporate_modern.mp3       ← 现代企业 (产品介绍)
├── acoustic_warm.mp3          ← 温暖原声 (情感/美食)
├── cinematic_tension.mp3      ← 紧张悬疑 (预告片)
├── lofi_chill.mp3             ← Lo-fi 放松 (Vlog)
└── orchestral_inspiring.mp3   ← 管弦乐振奋 (励志/教育)
```

## 使用方式

```bash
# 指定音乐文件 (会自动分析 BPM + 卡点对齐)
python main.py "产品宣传" --music music/upbeat_electronic.mp3

# 不指定 → 使用 Seedance 原生音频 (环境音)
python main.py "产品宣传"
```

## 快速下载推荐曲目

访问 Pixabay Music 搜索以下关键词, 下载前几个高评分结果:

| 风格 | 搜索关键词 | 建议时长 |
|------|-----------|---------|
| 史诗 | "epic cinematic" | 60-120s |
| 活力 | "upbeat electronic" | 30-60s |
| 平静 | "ambient calm" | 60-120s |
| 企业 | "corporate modern" | 30-60s |
| 温暖 | "acoustic warm" | 30-60s |
| 悬疑 | "tension dramatic" | 30-60s |

## 注意事项

- 时长建议 ≥ 视频目标时长 (30-60 秒够用)
- MP3 / WAV 均可, 采样率 44.1kHz 或 48kHz
- 如果音乐短于视频, FFmpeg 会循环 (当前实现是截断)
- BPM 分析需要 librosa, 确保已安装: `pip install librosa`

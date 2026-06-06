# LUT 调色文件 (.cube)

将 .cube 格式的 3D LUT 文件放在此目录。文件名需要与 `config.py` 中的 `MOOD_LUT_MAP` 对应。

## 免费 LUT 下载源 (推荐)

### 1. IWLTBAP — 10 个免费电影级 LUT
- 🔗 https://blog.iwltbap.com/the-10-free-cinematic-luts-by-iwltbap/
- 包含: Aspen (Kodak Ektar/2383 风格), Sedona (暖调), Tahoe (冷调) 等
- 格式: .cube, 免费商用

### 2. Gumroad — Kodak 2393 LUT (免费)
- 🔗 https://filmborders.gumroad.com/l/Kodak2393LUT
- 经典 Kodak 电影胶片印刷模拟
- 格式: .cube, 免费

### 3. PresetPro — 终极免费 LUT 合集
- 🔗 https://www.presetpro.com/the-ultimate-free-color-grading-lut-collection/
- 包含 Kodak 模拟、Teal & Orange、Bleach Bypass 等数十款
- 格式: .cube, 免费

### 4. Uppbeat — 免费 LUT 单独下载
- Teal & Orange: https://uppbeat.io/luts/asset/cinematic-teal-orange-film-look-7076
- Kodak 模拟: https://uppbeat.io/luts/asset/kodak-film-emulation-lut-4065
- Cinematic Teal: https://uppbeat.io/luts/asset/cinematic-teal-lut-2616

### 5. FreshLUTs — 浏览器内预览 + 下载
- 🔗 https://freshluts.com/luts/109

## 需要下载的文件 (对应 config.py 映射)

```
luts/
├── Kodak2383_D65.cube        ← cinematic / premium
├── TealOrange.cube           ← energetic
├── BleachBypass.cube         ← dramatic
├── WarmSunset.cube           ← warm
├── NordicCold.cube           ← cold
├── CyberpunkNeon.cube        ← futuristic
├── DesaturatedDrama.cube     ← documentary
├── VintageFilm.cube          ← vintage
└── TechClean.cube            ← modern_tech
```

## 快速下载脚本

```bash
# 从 IWLTBAP 下载 (需手动从网站获取直链)
# 或者从 DaVinci Resolve 自带 LUT 目录复制:
# macOS: /Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/
# 找到 Film Looks 文件夹下的 Kodak 相关 .cube 文件

# 最简单的方式: 安装 DaVinci Resolve (免费版) 即自带大量专业 LUT
```

## 注意事项

- LUT 文件大小通常 1-5 MB
- 确保是 `.cube` 格式 (FFmpeg `lut3d` 滤镜需要)
- 33x33x33 或 65x65x65 精度均可
- 如果没有 LUT 文件, 流水线会自动跳过调色步骤

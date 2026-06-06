#!/bin/bash
# 测试拼接 — 直接运行: bash test_concat.sh

BASE="/Users/lucas/Desktop/seedance/output/20260605_172813/normalized"
OUTPUT="/Users/lucas/Desktop/seedance/output/20260605_172813/concat_test.mp4"

echo "🔗 测试 xfade 拼接 4 个镜头..."
echo ""

# 先查看每个视频时长
for f in "$BASE"/shot_00{1,2,3,4}.mp4; do
  dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
  echo "  $(basename $f): ${dur}s"
done

echo ""
echo "开始拼接 (crossfade 0.5s)..."

ffmpeg -y \
  -i "$BASE/shot_001.mp4" \
  -i "$BASE/shot_002.mp4" \
  -i "$BASE/shot_003.mp4" \
  -i "$BASE/shot_004.mp4" \
  -filter_complex "\
    [0:v][1:v]xfade=transition=fade:duration=0.5:offset=5.5[v0];\
    [v0][2:v]xfade=transition=fade:duration=0.5:offset=11.5[v1];\
    [v1][3:v]xfade=transition=fade:duration=0.5:offset=19.0[vout]" \
  -map "[vout]" \
  -c:v libx264 -preset medium -crf 18 \
  -movflags +faststart \
  "$OUTPUT"

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ 拼接成功: $OUTPUT"
  echo "   时长: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT")s"
  echo ""
  echo "打开预览: open \"$OUTPUT\""
  open "$OUTPUT"
else
  echo ""
  echo "❌ 拼接失败，尝试简单 concat 模式..."
  
  # fallback: 简单拼接 (无转场)
  echo "file '$BASE/shot_001.mp4'" > /tmp/concat_list.txt
  echo "file '$BASE/shot_002.mp4'" >> /tmp/concat_list.txt
  echo "file '$BASE/shot_003.mp4'" >> /tmp/concat_list.txt
  echo "file '$BASE/shot_004.mp4'" >> /tmp/concat_list.txt
  
  ffmpeg -y -f concat -safe 0 -i /tmp/concat_list.txt \
    -c:v libx264 -crf 18 -c:a aac \
    "$OUTPUT"
  
  if [ $? -eq 0 ]; then
    echo "✅ 简单拼接成功: $OUTPUT"
    open "$OUTPUT"
  fi
fi

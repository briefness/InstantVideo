"""仅运行 Stage 1 分镜生成, 用于快速验证 prompt 改进效果 (不消耗视频生成 API)"""

import json
import sys
from pipeline.storyboard import generate_storyboard

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "制作一个15秒的咖啡品牌宣传片"
    print(f"\n🧪 测试分镜生成: {prompt}\n")

    storyboard = generate_storyboard(
        user_request=prompt,
        target_duration=15,
    )

    # 输出分析摘要
    shots = storyboard["shots"]
    print(f"\n{'='*60}")
    print(f"📊 分镜分析摘要")
    print(f"{'='*60}")
    print(f"标题: {storyboard.get('title', 'N/A')}")
    print(f"镜头数: {len(shots)}")
    print(f"总时长: {sum(s['duration'] for s in shots)}s")
    print(f"情绪: {storyboard.get('mood', 'N/A')}")
    print()

    for s in shots:
        chars = s.get("characters", [])
        cam = s.get("camera", {})
        prompt_words = len(s.get("prompt_en", "").split())
        print(f"Shot {s['shot_id']} ({s['duration']}s):")
        print(f"  场景: {s.get('scene_description', 'N/A')[:60]}")
        print(f"  景别: {cam.get('start_framing', '?')} → {cam.get('end_framing', '?')}")
        print(f"  运镜: {cam.get('primary_movement', '?')}")
        print(f"  构图: {cam.get('composition', '?')}")
        print(f"  情绪: {s.get('mood', '?')}")
        print(f"  角色: {chars if chars else '(无 - insert shot)'}")
        print(f"  prompt_en 词数: {prompt_words}")
        print(f"  字幕: {s.get('subtitle_text', 'N/A')}")
        print()

    # 保存完整 JSON
    out_path = "output/test_storyboard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)
    print(f"💾 完整 JSON 已保存: {out_path}")

if __name__ == "__main__":
    main()

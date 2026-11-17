"""用分布充分的特征点检验取景是否未变；像素变化不是镜头变化。

测量只提供独立反证，不能证明新景别、构图质量或人物身份正确。
纹理不足、遮挡过大、非平面变化等情况返回inconclusive，不冒充成功或失败。
"""
from pathlib import Path
import hashlib

import cv2
import numpy as np


VERSION = 'distributed-homography-v1'


def measure_composition(source: str | Path, candidate: str | Path) -> dict:
    frames, hashes = [], []
    for name in (source, candidate):
        path = Path(name)
        if not path.is_file() or path.stat().st_size > 20_000_000:
            raise ValueError('镜头测量缺少有界原始图像')
        data = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None or frame.size == 0 or frame.size > 40_000_000:
            raise ValueError('镜头测量图像不可读取或过大')
        frames.append(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))
        hashes.append(hashlib.sha256(data).hexdigest())
    result = {'version': VERSION, 'status': 'inconclusive', 'source_sha256': hashes[0],
              'candidate_sha256': hashes[1], 'inliers': 0, 'coverage': 0.0}
    detector = cv2.SIFT_create(nfeatures=2000)
    first, a = detector.detectAndCompute(frames[0], None)
    second, b = detector.detectAndCompute(frames[1], None)
    if a is None or b is None or min(len(a), len(b)) < 20:
        return result
    pairs = cv2.BFMatcher().knnMatch(a, b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7 * p[1].distance]
    if len(good) < 20:
        return result
    src = np.float32([first[m.queryIdx].pt for m in good])
    dst = np.float32([second[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return result
    valid = mask.ravel() > 0
    count = int(valid.sum())
    result['inliers'] = count
    if count < 20 or count / len(good) < .6:
        return result
    coverage = min(cv2.contourArea(cv2.convexHull(points[valid])) / (640 * 480)
                   for points in (src, dst))
    result['coverage'] = round(coverage, 6)
    # 局部主体/道具匹配不能代表整幅摄影。两侧都须有广泛分布的几何证据。
    occupied = [len({(min(2, int(x * 3 / 640)), min(2, int(y * 3 / 480)))
                     for x, y in points[valid]}) for points in (src, dst)]
    if coverage < .5 or min(occupied) < 6:
        return result
    grid = np.float32([[(x, y) for y in (48, 240, 432) for x in (64, 320, 576)]])
    projected = cv2.perspectiveTransform(grid, matrix)
    if not np.isfinite(projected).all():
        return result
    displacement = float(np.percentile(np.linalg.norm((projected - grid) / [640, 480], axis=2), 90))
    result['displacement_fraction'] = round(displacement, 6)
    # 1.5%仅是“几何近乎没动”的保守噪声带，不是合格景别变化的评分阈值。
    result['status'] = 'unchanged' if displacement <= .015 else 'changed'
    return result


def measure_video_composition(source: str | Path, video: str | Path, timestamp: float) -> dict:
    from tempfile import TemporaryDirectory
    from tools.frame_extractor import extract_frame
    if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError('镜头测量时间无效')
    with TemporaryDirectory(prefix='camera-measure-') as directory:
        frame = Path(directory) / 'sample.jpg'
        extract_frame(str(video), str(frame), timestamp=timestamp)
        return {**measure_composition(source, frame), 'timestamp': timestamp}


def composition_verdict(observed: bool, evidence: dict) -> bool:
    """几何变化不授予合格；几何未变可否定模型的切镜肯定判断。"""
    if type(observed) is not bool or evidence.get('version') != VERSION or evidence.get('status') not in {
            'unchanged', 'changed', 'inconclusive'}:
        raise ValueError('镜头观察或测量协议无效')
    return observed and evidence['status'] != 'unchanged'

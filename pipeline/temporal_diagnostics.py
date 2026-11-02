"""Image-change measurements, explicitly not action or aesthetic judgments.

Dominant affine registration is an estimate, not camera/actor segmentation.
Texture loss, lighting, parallax and occlusion can invalidate that estimate.
"""
import math

import cv2
import numpy as np


def frame_change(first: np.ndarray, second: np.ndarray, delta_seconds: float) -> dict:
    if (not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray)
            or first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 2 or second.shape != first.shape
            or min(first.shape) < 32 or max(first.shape) > 640
            or type(delta_seconds) not in (int, float) or not math.isfinite(delta_seconds)
            or delta_seconds <= 0):
        raise ValueError('需要同尺寸、有界灰度图和正的实际时间间隔')
    raw = float(cv2.absdiff(first, second).mean())
    result = {'raw_change': raw, 'exact_duplicate': bool(np.array_equal(first, second)),
              'registration': 'unavailable', 'aligned_change': None,
              'dominant_speed_heights_per_second': None, 'inlier_fraction': None}
    points = cv2.goodFeaturesToTrack(first, maxCorners=250, qualityLevel=.02, minDistance=6)
    if points is None or len(points) < 20:
        return result
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    if moved is None:
        return result
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(second, first, moved, None)
    if returned is None:
        return result
    valid = (forward.ravel() == 1) & (backward.ravel() == 1)
    valid &= np.linalg.norm(returned - points, axis=2).ravel() <= 1.
    src, dst = points[valid].reshape(-1, 2), moved[valid].reshape(-1, 2)
    if len(src) < 20:
        return result
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                ransacReprojThreshold=1.5, maxIters=1000)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return result
    result['inlier_fraction'] = float(inliers.mean())
    # These are estimator-support bounds, not movement quality thresholds.
    if result['inlier_fraction'] < .6:
        return result
    height, width = first.shape
    mask = cv2.warpAffine(np.ones_like(first), matrix, (width, height), flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    if float(mask.mean()) < .7:
        return result
    aligned = cv2.warpAffine(first, matrix, (width, height))
    center = np.array([width / 2, height / 2, 1.])
    displacement = matrix @ center - center[:2]
    result.update(registration='estimated',
                  aligned_change=float(cv2.absdiff(aligned, second)[mask].mean()),
                  dominant_speed_heights_per_second=float(np.linalg.norm(displacement) / height / delta_seconds))
    return result


def measure_frames(frames: list[np.ndarray], timestamps: list[float]) -> dict:
    if (not 2 <= len(frames) <= 900 or len(frames) != len(timestamps)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError('需要2–900帧及严格递增的逐帧时间来源')
    pairs = [{'start_seconds': a, 'end_seconds': b,
              **frame_change(first, second, b - a)}
             for first, second, a, b in zip(frames, frames[1:], timestamps, timestamps[1:])]
    valid = [p['aligned_change'] for p in pairs if p['registration'] == 'estimated']
    return {'measurement': 'temporal-diagnostics-v1', 'scope': 'image_change_only',
            'action_fluency': 'not_evaluated', 'pairs': pairs,
            'summary': {'pairs': len(pairs), 'estimated_pairs': len(valid),
                        'exact_duplicate_pairs': sum(p['exact_duplicate'] for p in pairs),
                        'raw_change_median': float(np.median([p['raw_change'] for p in pairs])),
                        'aligned_change_median': float(np.median(valid)) if valid else None},
            'limitations': ['Dominant image motion is not proven camera motion.',
                            'Aligned residual includes articulation, parallax, cloth, lighting and occlusion.',
                            'Neither image-change magnitude nor duplicate frames proves artistic quality.',
                            'Unavailable registration is missing evidence, never zero movement.']}

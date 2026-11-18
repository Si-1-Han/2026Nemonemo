"""5단계: 원근 변환 (담당: 채성우)

촬영 시 한 번, 원본 고해상도 이미지에 적용한다.
앱(Dart)의 lib/pipeline/warp.dart 와 같은 동작을 유지할 것.
"""
from __future__ import annotations

import cv2
import numpy as np


def warp_document(
    img: np.ndarray,
    corners: np.ndarray,
    aspect_ratio: float | None = None,
) -> np.ndarray:
    """정렬된 꼭짓점으로 문서를 펴고, 선택하면 출력 가로/세로 비율을 맞춘다.

    ``aspect_ratio`` 는 width / height (예: 세로 A4 는 1 / 1.4142)이다.
    지정하지 않으면 검출한 변 길이로 출력 크기를 정한다.
    """
    tl, tr, br, bl = np.asarray(corners, dtype=np.float32)

    measured_width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    measured_height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if aspect_ratio is None:
        width = int(round(measured_width))
        height = int(round(measured_height))
    else:
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("aspect_ratio 는 유한한 양수여야 합니다 (width / height).")
        # 측정된 면적을 유지하며 표준 비율로 출력 크기를 조정한다.
        area = measured_width * measured_height
        width = int(round(np.sqrt(area * aspect_ratio)))
        height = int(round(np.sqrt(area / aspect_ratio)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_LINEAR)

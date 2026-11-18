"""파이프라인 통합 (담당: 전성재)

앱과 같은 2갈래 구조:
  - scan_frame  : 라이브 루프. 1~4단계. 매 프레임.
  - scan_capture: 촬영 시. 5~6단계. 원본 해상도.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import find_document, order_corners
from .edges import detect_edges
from .enhance import enhance
from .preprocess import preprocess
from .warp import warp_document


@dataclass
class FrameResult:
    gray: np.ndarray            # 처리용 그레이스케일 (리사이즈됨)
    edges: np.ndarray           # 엣지 맵
    corners: np.ndarray | None  # 정렬된 꼭짓점, 처리 이미지 좌표계
    ratio: float                # 원본 / 처리 크기

    @property
    def corners_full(self) -> np.ndarray | None:
        """원본 해상도 좌표계의 꼭짓점"""
        return None if self.corners is None else self.corners * self.ratio


def scan_frame(img: np.ndarray) -> FrameResult:
    gray, ratio = preprocess(img)
    edges = detect_edges(gray)
    corners = find_document(edges)
    if corners is not None:
        corners = order_corners(corners)
    return FrameResult(gray=gray, edges=edges, corners=corners, ratio=ratio)


def scan_capture(
    img: np.ndarray,
    corners_full: np.ndarray,
    mode: str = "adaptive",
    aspect_ratio: float | None = None,
) -> np.ndarray:
    warped = warp_document(img, corners_full, aspect_ratio=aspect_ratio)
    return enhance(warped, mode)


def scan(img: np.ndarray, mode: str = "adaptive") -> tuple[np.ndarray | None, FrameResult]:
    """사진 한 장을 끝까지 처리한다. 검출 실패 시 (None, frame_result)."""
    fr = scan_frame(img)
    if fr.corners is None:
        return None, fr
    return scan_capture(img, fr.corners_full, mode), fr

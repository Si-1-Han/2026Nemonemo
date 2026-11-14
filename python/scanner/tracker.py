"""실시간 안정화: 프레임 간 꼭짓점 흔들림 보정 + 자동 촬영 판단 (담당: 신승준)

앱(Dart)의 lib/pipeline/tracker.dart 와 같은 동작을 유지할 것.
Python 에서는 동영상 파일이나 연속 사진으로 실험한다 (experiments/tracker_video.py).

- CornerTracker        : 지수 이동 평균(EMA)으로 부드럽게 만든다. (기본)
- KalmanCornerTracker  : 등속도 칼만 필터. 움직일 때 EMA 보다 덜 뒤처지고, 멈췄을 때 더 조용하다.

둘 다 '갑자기 다른 사각형으로 바뀌는 경우'를 처리한다.
한 프레임만 엉뚱한 사각형이 잡히면(손, 다른 종이, 오검출) 무시하고,
새 사각형이 jump_frames 프레임 연속으로 같은 자리에 잡혀야 그쪽으로 옮겨 간다.
"""
from __future__ import annotations

import numpy as np


class CornerTracker:
    """정렬된 꼭짓점(4, 2)을 프레임마다 받아 부드럽게 만들고, 안정 여부를 알려준다.

    Args:
        alpha: EMA 가중치. 클수록 새 프레임을 많이 반영(반응 빠름, 흔들림 큼).
        move_tol: 이 값(px, 처리 이미지 기준)보다 적게 움직이면 '정지'로 본다.
        stable_frames: 연속으로 정지한 프레임이 이만큼 쌓이면 is_stable=True.
        max_missed: 검출 실패가 이만큼 연속되기 전까지는 이전 결과를 유지.
        jump_tol: 현재 결과에서 이 값(px, 처리 이미지 기준)보다 멀리 떨어진 꼭짓점이 있으면
                  '다른 사각형'으로 본다.
        jump_frames: 다른 사각형이 이만큼 연속으로 같은 자리에 잡혀야 그쪽으로 옮겨 간다.
    """

    def __init__(self, alpha: float = 0.5, move_tol: float = 4.0,
                 stable_frames: int = 10, max_missed: int = 5,
                 jump_tol: float = 40.0, jump_frames: int = 3) -> None:
        self.alpha = alpha
        self.move_tol = move_tol
        self.stable_frames = stable_frames
        self.max_missed = max_missed
        self.jump_tol = jump_tol
        self.jump_frames = jump_frames
        self.reset()

    def reset(self) -> None:
        self.smoothed: np.ndarray | None = None
        self.still_count = 0
        self.missed = 0
        self._pending: np.ndarray | None = None  # 옮겨 갈 후보 사각형
        self._pending_count = 0

    @property
    def stability(self) -> float:
        """0~1, 자동 촬영까지 얼마나 안정됐는지 (UI 진행 표시용)"""
        return min(1.0, self.still_count / self.stable_frames)

    def update(self, corners: np.ndarray | None) -> tuple[np.ndarray | None, bool]:
        """새 프레임의 검출 결과를 반영한다.

        Returns:
            smoothed: 보정된 꼭짓점 (없으면 None)
            is_stable: 자동 촬영해도 될 만큼 안정적인지
        """
        if corners is None:
            self.missed += 1
            self.still_count = 0
            if self.missed > self.max_missed:
                self.reset()
            return self.smoothed, False

        corners = np.asarray(corners, dtype=np.float32)
        self.missed = 0

        if self.smoothed is None:
            self._start(corners)
            return self.smoothed, False

        movement = _max_distance(corners, self.smoothed)

        if movement > self.jump_tol:
            # 다른 사각형: 같은 자리에 연속으로 잡히는지 지켜본다. 그동안 결과는 그대로 둔다.
            if self._pending is not None and _max_distance(corners, self._pending) <= self.jump_tol:
                self._pending_count += 1
            else:
                self._pending_count = 1
            self._pending = corners
            self.still_count = 0
            if self._pending_count >= self.jump_frames:
                self._start(corners)
            return self.smoothed, False

        self._pending = None
        self._pending_count = 0
        self.smoothed = self._smooth(corners)
        self.still_count = self.still_count + 1 if movement < self.move_tol else 0
        return self.smoothed, self.still_count >= self.stable_frames

    # ---- 보정 방식별로 바꾸는 부분 ----

    def _start(self, corners: np.ndarray) -> None:
        """새 사각형으로 (다시) 시작한다."""
        self.smoothed = corners.copy()
        self.still_count = 0
        self._pending = None
        self._pending_count = 0

    def _smooth(self, corners: np.ndarray) -> np.ndarray:
        return (self.alpha * corners + (1.0 - self.alpha) * self.smoothed).astype(np.float32)


class KalmanCornerTracker(CornerTracker):
    """꼭짓점 좌표 8개(x, y × 4) 각각에 등속도 칼만 필터를 건다.

    상태 = [위치, 속도], 1프레임 = 1 시간 단위. 측정 잡음 분산을 1 로 두고
    process_noise 를 그에 대한 비율로 쓰기 때문에 좌표 단위(px 든 0~1 이든)와 상관없이 같은 값을 쓴다.

    Args:
        process_noise: 움직임 변화(가속) 잡음 / 측정 잡음. 클수록 반응 빠름, 흔들림 큼.
        (나머지는 CornerTracker 와 같다. alpha 는 쓰지 않는다.)
    """

    def __init__(self, process_noise: float = 0.02, move_tol: float = 4.0,
                 stable_frames: int = 10, max_missed: int = 5,
                 jump_tol: float = 40.0, jump_frames: int = 3) -> None:
        self.process_noise = process_noise
        super().__init__(move_tol=move_tol, stable_frames=stable_frames, max_missed=max_missed,
                         jump_tol=jump_tol, jump_frames=jump_frames)

    def reset(self) -> None:
        super().reset()
        self._vel: np.ndarray | None = None
        # 공분산 [[p00, p01], [p01, p11]] — 모든 좌표가 같은 잡음·같은 시점에 갱신되므로 하나만 둔다
        self._p = (1.0, 0.0, 1.0)

    def _start(self, corners: np.ndarray) -> None:
        super()._start(corners)
        self._vel = np.zeros_like(corners, dtype=np.float64)
        self._p = (1.0, 0.0, 1.0)

    def _smooth(self, corners: np.ndarray) -> np.ndarray:
        q = self.process_noise
        p00, p01, p11 = self._p
        # 예측: x' = x + v,  P' = F P Fᵀ + Q  (Q = q · [[1/4, 1/2], [1/2, 1]])
        pos = self.smoothed.astype(np.float64) + self._vel
        p00, p01, p11 = p00 + 2 * p01 + p11 + q / 4, p01 + p11 + q / 2, p11 + q
        # 갱신: 측정 잡음 R = 1
        s = p00 + 1.0
        k0, k1 = p00 / s, p01 / s
        resid = corners.astype(np.float64) - pos
        pos = pos + k0 * resid
        self._vel = self._vel + k1 * resid
        self._p = ((1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01)
        return pos.astype(np.float32)


def _max_distance(a: np.ndarray, b: np.ndarray) -> float:
    """두 사각형의 대응 꼭짓점 사이 최대 거리"""
    return float(np.max(np.linalg.norm(a - b, axis=1)))

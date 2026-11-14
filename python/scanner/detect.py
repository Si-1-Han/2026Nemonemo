"""3~4단계: 문서 영역 검출 + 꼭짓점 정렬 (담당: 신승준)

앱(Dart)의 lib/pipeline/detect.dart 와 같은 동작을 유지할 것.

검출은 두 단계로 한다.
  1) 기본: 윤곽선을 approxPolyDP 로 근사해서 꼭짓점 4개짜리 볼록 다각형을 찾는다.
  2) 대안(1이 실패할 때만): 손가락에 가려 변이 파이거나, 모서리가 가려지거나 접혀
     1)이 실패하는 경우를 위해 볼록 껍질(convexHull)로 근사한다.
     - 껍질을 epsilon 을 조금씩 키우며 approxPolyDP
         · 4각형 → 채택
         · 5각형 → 가장 짧은 변을 '잘린 모서리'로 보고 양옆 변의 교점으로 꼭짓점 복원
     - 그래도 안 되면 minAreaRect 로 회전된 직사각형을 씌운다
     모두 '얼마나 꽉 채우는가(FALLBACK_MIN_FILL)'를 확인해서 원·타원 같은 물체는 버린다.
     대안 단계 결과의 꼭짓점이 화면 가장자리에 닿으면 버린다. 문서가 같은 밝기의 배경(흰 책상 등)과
     붙어 버린 경우라 엉뚱한 사각형이 나오기 때문이다.
"""
from __future__ import annotations

import cv2
import numpy as np

# 문서로 인정할 최소 면적 (처리 이미지 면적 대비 비율)
MIN_AREA_RATIO = 0.10

# approxPolyDP 의 epsilon = APPROX_EPS * 둘레
APPROX_EPS = 0.02

# 면적 상위 몇 개의 윤곽선만 검사할지
TOP_K = 5

# [대안] 볼록 껍질 근사 epsilon 을 APPROX_EPS 부터 이 값까지 이 간격으로 키워 본다.
# 0.04 보다 크게 하면 원도 4각형으로 근사될 수 있다.
FALLBACK_EPS_MAX = 0.04
FALLBACK_EPS_STEP = 0.01

# [대안] 채움 비율 하한. 4각형 근사: 4각형 면적 / 껍질 면적, minAreaRect: 껍질 면적 / 직사각형 면적.
# 원은 0.785(π/4) 이므로 그보다 높게 둔다.
FALLBACK_MIN_FILL = 0.85

# [대안] 결과 꼭짓점이 화면 가장자리에서 이 거리(px, 처리 이미지 기준) 안에 있으면 버린다.
FALLBACK_BORDER_MARGIN = 2


def find_document(edges: np.ndarray) -> np.ndarray | None:
    """엣지 맵에서 문서(꼭짓점 4개인 볼록 다각형)를 찾는다.

    Returns:
        corners: float32 (4, 2) 배열, 처리 이미지 좌표계. 정렬되지 않은 상태.
        못 찾으면 None.
    """
    h, w = edges.shape[:2]
    min_area = MIN_AREA_RATIO * h * w

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    by_area = sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_K]
    for c in by_area:
        if cv2.contourArea(c) < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, APPROX_EPS * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return _fallback_quad(contours, min_area, w, h)


def _fallback_quad(contours, min_area: float, img_w: int, img_h: int) -> np.ndarray | None:
    """볼록 껍질로 문서를 근사한다. (find_document 의 대안 단계)

    끊긴 윤곽선은 contourArea 가 거의 0 이라 면적 순위에서 밀리므로,
    여기서는 껍질 면적으로 다시 순위를 매긴다.
    """
    hulls = []
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:  # 껍질 면적 <= 외접 사각형 면적이므로 미리 거른다
            continue
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area >= min_area:
            hulls.append((area, hull))
    hulls.sort(key=lambda t: t[0], reverse=True)

    steps = round((FALLBACK_EPS_MAX - APPROX_EPS) / FALLBACK_EPS_STEP) + 1
    for hull_area, hull in hulls[:TOP_K]:
        peri = cv2.arcLength(hull, True)
        for i in range(steps):
            eps = APPROX_EPS + i * FALLBACK_EPS_STEP
            approx = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2).astype(np.float32)
            if len(approx) < 4:
                break
            if len(approx) == 5:
                quad = _restore_cut_corner(approx)
                if quad is not None and _acceptable(quad, hull_area, img_w, img_h):
                    return quad
            if len(approx) == 4:
                if _acceptable(approx, hull_area, img_w, img_h):
                    return approx
                break

        box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        if _acceptable(box, hull_area, img_w, img_h):
            return box

    return None


def _acceptable(quad: np.ndarray, hull_area: float, img_w: int, img_h: int) -> bool:
    """대안 단계 결과를 받아들일지: 껍질을 충분히 채우고, 꼭짓점이 화면 가장자리에 닿지 않을 것"""
    m = FALLBACK_BORDER_MARGIN
    for x, y in quad:
        if x <= m or y <= m or x >= img_w - 1 - m or y >= img_h - 1 - m:
            return False
    return _fill(quad, hull_area) >= FALLBACK_MIN_FILL


def _fill(quad: np.ndarray, hull_area: float) -> float:
    """4각형과 껍질의 면적 비 (작은 쪽 / 큰 쪽, 0~1)"""
    qa = _polygon_area(quad)
    big = max(qa, hull_area)
    return min(qa, hull_area) / big if big > 0 else 0.0


def _polygon_area(pts: np.ndarray) -> float:
    """신발끈 공식. (Dart 쪽과 똑같이 계산하려고 cv2.contourArea 대신 직접 계산)"""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _restore_cut_corner(pent: np.ndarray) -> np.ndarray | None:
    """5각형에서 가장 짧은 변을 '잘린 모서리'로 보고, 양옆 변을 연장한 교점으로 꼭짓점을 복원한다.

    접힌 모서리, 손가락에 가려진 모서리처럼 꼭짓점 하나가 비스듬히 잘린 경우를 위한 것.
    복원한 4각형이 볼록하지 않으면 None.
    """
    p = pent.astype(np.float64)
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    k = int(np.argmin(lengths))  # 잘린 변: p[k] → p[k+1]
    a0, a1 = p[(k - 1) % 5], p[k]          # 잘린 변 앞쪽 변
    b0, b1 = p[(k + 1) % 5], p[(k + 2) % 5]  # 잘린 변 뒤쪽 변
    corner = _line_intersection(a0, a1, b0, b1)
    if corner is None:
        return None
    quad = np.array([p[(k + 2) % 5], p[(k + 3) % 5], p[(k + 4) % 5], corner], np.float32)
    return quad if _is_convex(quad) else None


def _line_intersection(a0, a1, b0, b1) -> np.ndarray | None:
    """직선 a0-a1 과 b0-b1 의 교점. 거의 평행하면 None."""
    da, db = a1 - a0, b1 - b0
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(denom) < 1e-9:
        return None
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denom
    return a0 + t * da


def _is_convex(quad: np.ndarray) -> bool:
    """네 꼭짓점에서 외적의 부호가 모두 같으면 볼록"""
    q = quad.astype(np.float64)
    signs = []
    for i in range(4):
        o, a, b = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        signs.append((a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0]))
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def order_corners(corners: np.ndarray) -> np.ndarray:
    """꼭짓점을 좌상, 우상, 우하, 좌하 순서로 정렬한다.

    좌상: x+y 최소, 우하: x+y 최대, 우상: y-x 최소, 좌하: y-x 최대
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )

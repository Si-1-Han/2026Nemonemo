"""3~4단계: 문서 영역 검출 + 꼭짓점 정렬 (담당: 신승준)

앱(Dart)의 lib/pipeline/detect.dart 와 같은 동작을 유지할 것.

면적(볼록 껍질 기준)이 큰 윤곽선부터 다음 두 방법을 차례로 시도한다.

  1) 긴 변 4개로 복원 (_quad_from_sides)
     윤곽선을 촘촘히 근사해 선분으로 나누고, 같은 직선 위의 선분끼리 묶어 길이가 가장 긴 직선 4개를
     문서의 네 변으로 본다. 이웃한 두 변의 교점이 꼭짓점이다.
     손가락이 변을 파거나(어두운 손가락) 변 밖으로 튀어나와도(피부색 손가락), 모서리 하나를 가리거나
     모서리가 접혀도 네 변의 나머지 부분은 직선으로 남아 있으므로 꼭짓점을 되찾을 수 있다.
  2) 기존 4각형 근사
     approxPolyDP 로 꼭짓점 4개짜리 볼록 다각형이 나오면 그대로 쓴다. 1)이 실패할 때만 쓴다.
     (문서 일부가 화면 밖으로 나가 변이 화면 가장자리에 붙은 경우 등)

1)의 결과는 '네 변이 실제 윤곽선으로 얼마나 덮여 있는가(SIDE_MIN_COVERAGE)'로 문서가 아닌 물체를 거른다.
원은 짧은 선분이 조금씩 꺾이며 이어지므로 같은 직선으로 묶이는 길이가 둘레의 일부밖에 안 되고,
다각형·L자 모양은 네 변으로 복원한 4각형의 둘레 중 비어 있는 부분이 많다.
(처음에는 '볼록 껍질 대비 면적'으로 걸렀는데, 손이 크게 들어오면 껍질이 커져서 맞게 찾은 사각형까지 버렸다.)
같은 직선 위의 선분이 꼭짓점 밖으로 많이 삐져나가면(SIDE_MAX_OVERHANG, L자 모양 등) 바깥 테두리가 아니므로 버리고,
꼭짓점이 화면 가장자리에 닿으면(문서가 같은 밝기의 배경과 붙어 버린 경우)도 버린다.
"""
from __future__ import annotations

import cv2
import numpy as np

# 문서로 인정할 최소 면적 (처리 이미지 면적 대비 비율)
MIN_AREA_RATIO = 0.10

# approxPolyDP 의 epsilon = APPROX_EPS * 둘레 (방법 2)
APPROX_EPS = 0.02

# 면적 상위 몇 개의 윤곽선만 검사할지
TOP_K = 5

# [방법 1] 선분으로 나눌 때의 촘촘한 근사 epsilon (둘레 대비)
SIDE_EPS = 0.005

# [방법 1] 두 선분을 같은 직선으로 묶는 기준: 각도 차(도), 직선까지 거리(처리 이미지 대각선 대비)
SIDE_ANGLE_TOL = 6.0
SIDE_DIST_TOL = 0.015

# [방법 1] 변으로 인정할 최소 길이 (둘레/4 대비). 이보다 짧은 직선만 남으면 4각형이 아니다.
SIDE_MIN_RATIO = 0.15

# [방법 1] 복원한 4각형의 네 변 위에 실제 선분이 덮고 있는 길이 / 4각형 둘레 의 하한.
# 틀린 사각형을 띄우는 것보다 못 찾는 편이 낫기 때문에 가린 문서가 통과하는 선에서 높게 잡는다.
SIDE_MIN_COVERAGE = 0.8

# [방법 1] 변과 같은 직선 위인데 꼭짓점 밖으로 삐져나간 선분 길이 / 변 길이 의 상한 (L자 모양은 0.5 이상)
SIDE_MAX_OVERHANG = 0.25

# [방법 1] 꼭짓점이 화면 가장자리에서 이 거리(px, 처리 이미지 기준) 안에 있으면 버린다.
SIDE_BORDER_MARGIN = 2


def find_document(edges: np.ndarray) -> np.ndarray | None:
    """엣지 맵에서 문서(꼭짓점 4개인 볼록 다각형)를 찾는다.

    Returns:
        corners: float32 (4, 2) 배열, 처리 이미지 좌표계. 정렬되지 않은 상태.
        못 찾으면 None.
    """
    h, w = edges.shape[:2]
    min_area = MIN_AREA_RATIO * h * w

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 끊기거나 파인 윤곽선은 contourArea 가 작게 나오므로 볼록 껍질 면적으로 순위를 매긴다.
    candidates = []
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:  # 껍질 면적 <= 외접 사각형 면적이므로 미리 거른다
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area >= min_area:
            candidates.append((hull_area, c))
    candidates.sort(key=lambda t: t[0], reverse=True)

    for _, c in candidates[:TOP_K]:
        quad = _quad_from_sides(c, w, h)
        if quad is not None:
            return quad
        if cv2.contourArea(c) >= min_area:
            approx = cv2.approxPolyDP(c, APPROX_EPS * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)

    return None


def _quad_from_sides(contour: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
    """윤곽선에서 가장 긴 직선 4개를 찾아 이웃한 직선의 교점을 꼭짓점으로 돌려준다. 실패하면 None."""
    peri = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, SIDE_EPS * peri, True).reshape(-1, 2).astype(np.float64)
    n = len(poly)
    if n < 4:
        return None

    # 1. 선분을 같은 직선끼리 묶는다. 묶음 = [단위 법선, 법선 방향 거리, 누적 길이, 선분 목록]
    angle_tol = np.cos(np.deg2rad(SIDE_ANGLE_TOL))
    dist_tol = SIDE_DIST_TOL * float(np.hypot(img_w, img_h))
    groups: list[list] = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        length = float(np.linalg.norm(q - p))
        if length < 1e-6:
            continue
        normal = np.array([p[1] - q[1], q[0] - p[0]]) / length
        mid = (p + q) / 2
        for g in groups:
            if abs(float(g[0] @ normal)) >= angle_tol and abs(float(g[0] @ mid) - g[1]) <= dist_tol:
                g[2] += length
                g[3].append((p, q))
                break
        else:
            groups.append([normal, float(normal @ mid), length, [(p, q)]])

    groups.sort(key=lambda g: g[2], reverse=True)
    if len(groups) < 4 or groups[3][2] < SIDE_MIN_RATIO * peri / 4:
        return None

    # 2. 각 묶음의 선분들에 직선을 맞춘다 (선분 위 균일 분포의 최소제곱 직선)
    lines = [(*_fit_line(g[3]), g[3]) for g in groups[:4]]

    # 3. 문서 중심에서 본 방향 순서로 변을 정렬하고, 이웃한 변의 교점을 구한다
    center = poly.mean(axis=0)

    def direction(line):
        point, d, _ = line
        foot = point + float((center - point) @ d) * d
        return np.arctan2(foot[1] - center[1], foot[0] - center[0])

    lines.sort(key=direction)
    corners = []
    for i in range(4):
        x = _intersect(lines[i], lines[(i + 1) % 4])
        if x is None:
            return None
        corners.append(x)
    quad = np.array(corners, np.float32)
    if not _is_convex(quad) or _touches_border(quad, img_w, img_h):
        return None

    # 4. 각 변(꼭짓점 i-1 ~ i) 위에 실제 선분이 얼마나 있는지 센다.
    #    같은 직선 위인데 꼭짓점 밖으로 많이 삐져나간 선분이 있으면 이 4각형은 윤곽의 바깥 테두리가 아니다
    #    (L자 모양 등). 손가락 테두리가 우연히 변과 한 직선에 놓이는 정도는 봐준다.
    covered = 0.0
    for i, (point, d, segments) in enumerate(lines):
        ta = float((corners[i - 1] - point) @ d)
        tb = float((corners[i] - point) @ d)
        lo, hi = min(ta, tb), max(ta, tb)
        overhang = 0.0
        for p, q in segments:
            t1, t2 = sorted((float((p - point) @ d), float((q - point) @ d)))
            covered += max(0.0, min(hi, t2) - max(lo, t1))
            overhang += max(0.0, min(t2, lo - dist_tol) - t1) + max(0.0, t2 - max(t1, hi + dist_tol))
        if overhang > SIDE_MAX_OVERHANG * (hi - lo):
            return None
    return quad if covered >= SIDE_MIN_COVERAGE * _perimeter(quad) else None


def _fit_line(segments) -> tuple[np.ndarray, np.ndarray]:
    """선분들 위에 고르게 퍼진 점들에 대한 최소제곱 직선 (지나는 점, 단위 방향).

    선분 p→q 위 균일 분포의 평균은 (p+q)/2, 2차 모멘트는 (ppᵀ + qqᵀ)/3 + (pqᵀ + qpᵀ)/6 이므로
    점을 뽑지 않고 길이 가중 합으로 바로 계산한다. (Dart 와 같은 식)
    """
    total = 0.0
    mean = np.zeros(2)
    second = np.zeros((2, 2))
    for p, q in segments:
        length = float(np.linalg.norm(q - p))
        total += length
        mean += length * (p + q) / 2
        second += length * ((np.outer(p, p) + np.outer(q, q)) / 3 + (np.outer(p, q) + np.outer(q, p)) / 6)
    mean /= total
    cov = second / total - np.outer(mean, mean)
    theta = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    return mean, np.array([np.cos(theta), np.sin(theta)])


def _intersect(a, b) -> np.ndarray | None:
    """두 직선 (점, 방향, ...) 의 교점. 거의 평행하면 None."""
    pa, da, pb, db = a[0], a[1], b[0], b[1]
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(denom) < 1e-6:
        return None
    t = ((pb[0] - pa[0]) * db[1] - (pb[1] - pa[1]) * db[0]) / denom
    return pa + t * da


def _touches_border(quad: np.ndarray, img_w: int, img_h: int) -> bool:
    m = SIDE_BORDER_MARGIN
    return any(x <= m or y <= m or x >= img_w - 1 - m or y >= img_h - 1 - m for x, y in quad)


def _perimeter(quad: np.ndarray) -> float:
    q = quad.astype(np.float64)
    return float(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).sum())


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

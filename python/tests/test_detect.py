"""문서 검출 테스트 (담당: 신승준)

기존 4각형 근사만으로는 실패하던 가림·접힘 상황과, 문서가 아닌 물체를 잘못 잡지 않는지를 확인한다.
장면은 experiments/detect_occlusion.py 와 같은 함수로 만든다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluate import SUCCESS_RATIO, corner_error
from scanner import detect_edges, find_document, order_corners, preprocess
from scanner.synthetic import make_scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from detect_occlusion import no_document, occlusion_scene  # noqa: E402

N = 10  # 장면당 장수 (실험 스크립트는 40장)


def detect(img: np.ndarray) -> np.ndarray | None:
    gray, ratio = preprocess(img)
    corners = find_document(detect_edges(gray))
    return None if corners is None else order_corners(corners) * ratio


def strict_only(img: np.ndarray) -> bool:
    """이전 방식(approxPolyDP 4각형만)으로 찾을 수 있는지"""
    gray, _ = preprocess(img)
    edges = detect_edges(gray)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.1 * edges.size:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return True
    return False


@pytest.mark.parametrize("scene", ["손가락 1개 (변)", "손가락 2개 (변)", "엄지 (모서리)", "접힌 모서리"])
def test_occluded_documents_found(scene):
    for i in range(N):
        img, gt = occlusion_scene(scene, i)
        found = detect(img)
        assert found is not None, f"{scene} #{i}: 못 찾음"
        assert corner_error(found, gt) <= SUCCESS_RATIO, f"{scene} #{i}: 오차 {corner_error(found, gt):.3f}"


def test_occlusion_breaks_previous_method():
    """위 테스트가 의미 있는지: 손가락 2개 장면은 이전 방식으로는 하나도 못 찾는다"""
    assert not any(strict_only(occlusion_scene("손가락 2개 (변)", i)[0]) for i in range(N))


def test_team_hand_condition():
    """조건별 합성 세트(make_scene)의 '손가림' (이전 방식 0/6)"""
    for i in range(6):
        img, gt, _ = make_scene("손가림", seed=i)
        found = detect(img)
        assert found is not None and corner_error(found, gt) <= SUCCESS_RATIO, f"손가림 #{i}"


@pytest.mark.parametrize("kind", ["빈 배경", "원·타원 물체", "무늬 배경"])
def test_no_document_scenes(kind):
    assert all(detect(no_document(kind, i)) is None for i in range(N))


def _shape_scene(pts: np.ndarray) -> np.ndarray:
    img = np.full((900, 1200, 3), 80, np.uint8)
    cv2.fillPoly(img, [pts.astype(np.int32)], (235, 235, 235))
    return img


@pytest.mark.parametrize("sides", [3, 5, 6, 8])
def test_polygons_rejected(sides):
    """직선 변이 있지만 4각형이 아닌 물체"""
    t = np.arange(sides) * 2 * np.pi / sides + 0.3
    assert detect(_shape_scene(np.stack([600 + 330 * np.cos(t), 450 + 330 * np.sin(t)], 1))) is None


def test_l_shape_rejected():
    """L자 모양은 네 변으로 복원한 4각형 둘레의 1/4 이 비어 있어 SIDE_MIN_COVERAGE 에 걸린다"""
    x0, y0, s = 330, 180, 540
    pts = np.array([[x0, y0], [x0 + s / 2, y0], [x0 + s / 2, y0 + s / 2], [x0 + s, y0 + s / 2],
                    [x0 + s, y0 + s], [x0, y0 + s]])
    assert detect(_shape_scene(pts)) is None


def test_quad_from_sides_geometry():
    """한 변 가운데가 파이고 모서리 하나가 잘린 사각형 윤곽선 → 원래 네 꼭짓점"""
    from scanner.detect import _quad_from_sides

    contour = np.array([
        [100, 100], [300, 100], [300, 150], [360, 150], [360, 100], [500, 100],  # 위 변이 파임
        [500, 380], [480, 400],                                                  # 우하단이 잘림
        [100, 400],
    ], np.int32).reshape(-1, 1, 2)
    quad = _quad_from_sides(contour, 600, 500)
    np.testing.assert_allclose(order_corners(quad), [[100, 100], [500, 100], [500, 400], [100, 400]], atol=0.5)

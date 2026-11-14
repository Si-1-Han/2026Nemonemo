"""문서 검출 대안 단계(convexHull / minAreaRect) 테스트 (담당: 신승준)

기본 단계(approxPolyDP 4각형)가 실패하는 대표 상황을 합성 사진으로 만들어 확인한다.
"""
import cv2
import numpy as np
import pytest

from scanner import detect_edges, find_document, order_corners, preprocess
from scanner.synthetic import DEFAULT_CORNERS, make_photo

FINGER_COLOR = (70, 70, 80)


def detect(img: np.ndarray) -> np.ndarray | None:
    gray, ratio = preprocess(img)
    corners = find_document(detect_edges(gray))
    return None if corners is None else order_corners(corners) * ratio


def finger_on_bottom_edge(img: np.ndarray) -> np.ndarray:
    """아래 변 가운데를 손가락이 누르고 있는 사진 → 윤곽선이 파여서 기본 단계는 실패한다."""
    _, _, br, bl = DEFAULT_CORNERS
    mid = ((br + bl) / 2).astype(int)
    cv2.ellipse(img, (int(mid[0]), int(mid[1]) + 40), (45, 110), 0, 0, 360, FINGER_COLOR, -1)
    return img


def test_strict_path_fails_on_finger_notch():
    """대안 단계가 실제로 쓰이는 상황인지 확인 (기본 단계만으로는 못 찾음)"""
    gray, _ = preprocess(finger_on_bottom_edge(make_photo(DEFAULT_CORNERS)))
    contours, _ = cv2.findContours(detect_edges(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        assert not (len(approx) == 4 and cv2.isContourConvex(approx))


@pytest.mark.parametrize("shadow", [False, True])
def test_finger_notch_recovered(shadow):
    found = detect(finger_on_bottom_edge(make_photo(DEFAULT_CORNERS, shadow=shadow)))
    assert found is not None
    err = np.linalg.norm(found - DEFAULT_CORNERS, axis=1).max()
    assert err < 15, f"꼭짓점 오차 {err:.1f}px"


def test_occluded_corner_restored():
    """엄지가 우하단 모서리를 가린 사진 → 껍질이 5각형이 되고, 양옆 변을 연장해 꼭짓점을 복원한다"""
    img = make_photo(DEFAULT_CORNERS, shadow=False)
    br = DEFAULT_CORNERS[2].astype(int)
    cv2.ellipse(img, (int(br[0]) - 10, int(br[1]) + 10), (70, 90), 30, 0, 360, FINGER_COLOR, -1)
    found = detect(img)
    assert found is not None
    err = np.linalg.norm(found - DEFAULT_CORNERS, axis=1).max()
    assert err < 15, f"꼭짓점 오차 {err:.1f}px"


def test_restore_cut_corner_geometry():
    from scanner.detect import _restore_cut_corner
    # 정사각형 (0,0)-(100,100) 의 우하단을 10px 잘라낸 5각형
    pent = np.array([[0, 0], [100, 0], [100, 90], [90, 100], [0, 100]], np.float32)
    quad = _restore_cut_corner(pent)
    np.testing.assert_allclose(order_corners(quad), [[0, 0], [100, 0], [100, 100], [0, 100]], atol=1e-4)


@pytest.mark.parametrize("shape", ["circle", "ellipse"])
def test_round_objects_rejected(shape):
    """대안 단계가 원·타원(접시, 시계 등)을 문서로 오인하지 않는다"""
    img = np.full((900, 1200, 3), 80, np.uint8)
    if shape == "circle":
        cv2.circle(img, (600, 450), 300, (240, 240, 240), -1)
    else:
        cv2.ellipse(img, (600, 450), (400, 250), 15, 0, 360, (240, 240, 240), -1)
    assert detect(img) is None


def test_document_merged_with_bright_border_region_rejected():
    """문서 오른쪽 배경이 문서와 같은 밝기라 화면 끝까지 붙어 버리면 엉뚱한 사각형 대신 None"""
    img = make_photo(DEFAULT_CORNERS, shadow=False)
    h, w = img.shape[:2]
    band = np.zeros((h, w), np.uint8)
    cv2.fillPoly(band, [np.array([[880, 0], [w, 0], [w, h], [960, h]], np.int32)], 255)
    doc = np.zeros((h, w), np.uint8)
    cv2.fillPoly(doc, [DEFAULT_CORNERS.astype(np.int32)], 255)
    img[(band > 0) & (doc == 0)] = 238
    assert detect(img) is None

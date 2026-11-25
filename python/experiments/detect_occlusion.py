"""[신승준] 가림·접힘 상황에서 문서 검출 비교

사용법:  python experiments/detect_occlusion.py [--n 40] [--img 저장할_그림.jpg]

1) 조건별 합성 세트 (dataset.py, 실제 사진이 있으면 data/raw) 조건별 성공률
2) 가림 세트: 합성 사진에 손가락·엄지·접힌 모서리를 넣은 장면별 성공 / 틀린 사각형 / 못 찾음
3) 문서가 없는 장면에서 잘못 검출한 수, find_document 처리 시간

성공 기준은 evaluate.py 와 같다 (꼭짓점 평균 오차 <= 문서 대각선의 2%).
'틀린 사각형'은 무언가를 검출했지만 기준을 넘은 경우로, 못 찾은 것보다 나쁘다 (화면에 엉뚱한 사각형이 뜨고
자동 촬영까지 될 수 있다).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_dataset, wants_synthetic  # noqa: E402
from evaluate import SUCCESS_RATIO, corner_error  # noqa: E402
from scanner import detect_edges, find_document, order_corners, preprocess  # noqa: E402
from scanner.synthetic import make_photo  # noqa: E402

SIZE = (1200, 900)
BASE = np.array([[310, 140], [880, 190], [960, 800], [230, 760]], np.float32)


def detect(img: np.ndarray) -> np.ndarray | None:
    gray, ratio = preprocess(img)
    corners = find_document(detect_edges(gray))
    return None if corners is None else order_corners(corners) * ratio


def judge(found, gt) -> str:
    if found is None:
        return "miss"
    return "ok" if corner_error(found, gt) <= SUCCESS_RATIO else "wrong"


# ---- 가림 장면 ------------------------------------------------------------

def random_corners(rng):
    c = BASE + rng.uniform(-50, 50, BASE.shape)
    center = c.mean(axis=0)
    c = (c - center) * rng.uniform(0.8, 1.05) + center + rng.uniform([-60, -40], [60, 40])
    return c.astype(np.float32)


def finger_color(rng):
    if rng.random() < 0.5:  # 피부색 (BGR): 배경보다 밝아 종이와 윤곽선이 이어진다
        return tuple(int(v) for v in rng.uniform([110, 140, 180], [150, 175, 215]))
    return tuple(int(v) for v in rng.uniform(55, 95, 3))  # 어두운 장갑·그림자: 종이 변이 파인다


def _outward(c, p):
    v = p - c.mean(axis=0)
    return v / np.linalg.norm(v)


def finger_on_edge(img, c, rng, k=None):
    k = int(rng.integers(4)) if k is None else k
    p = c[k] + (c[(k + 1) % 4] - c[k]) * rng.uniform(0.25, 0.75)
    out = _outward(c, p)
    pos = p + out * rng.uniform(20, 50)
    axes = (int(rng.uniform(90, 130)), int(rng.uniform(35, 48)))
    cv2.ellipse(img, (int(pos[0]), int(pos[1])), axes, float(np.degrees(np.arctan2(out[1], out[0]))),
                0, 360, finger_color(rng), -1)
    return img


def two_fingers(img, c, rng):
    k = int(rng.integers(4))
    finger_on_edge(img, c, rng, k)
    return finger_on_edge(img, c, rng, (k + 2) % 4)


def thumb_on_corner(img, c, rng):
    k = int(rng.integers(4))
    pos = c[k] + _outward(c, c[k]) * rng.uniform(-5, 20)
    axes = (int(rng.uniform(55, 75)), int(rng.uniform(75, 95)))
    cv2.ellipse(img, (int(pos[0]), int(pos[1])), axes, float(rng.uniform(0, 180)), 0, 360, finger_color(rng), -1)
    return img


def folded_corner(img, c, rng):
    k = int(rng.integers(4))
    p, prv, nxt = c[k], c[(k - 1) % 4], c[(k + 1) % 4]
    cut = rng.uniform(20, 45)
    a = p + (prv - p) / np.linalg.norm(prv - p) * cut
    b = p + (nxt - p) / np.linalg.norm(nxt - p) * cut
    tip = p + _outward(c, p) * 6
    cv2.fillPoly(img, [np.array([tip, a, b]).astype(np.int32)], tuple(int(v) for v in img[5, 5]))
    return img


SCENES = {
    "가림 없음": lambda img, c, rng: img,
    "손가락 1개 (변)": finger_on_edge,
    "손가락 2개 (변)": two_fingers,
    "엄지 (모서리)": thumb_on_corner,
    "접힌 모서리": folded_corner,
}


def occlusion_scene(name: str, i: int):
    rng = np.random.default_rng(1000 + i)
    c = random_corners(rng)
    img = make_photo(c, size=SIZE, seed=i, shadow=bool(i % 2))
    return SCENES[name](img, c, rng), c


def no_document(kind: str, i: int):
    rng = np.random.default_rng(2000 + i)
    w, h = SIZE
    img = cv2.GaussianBlur(rng.integers(60, 100, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    if kind == "원·타원 물체":
        axes = (int(rng.uniform(200, 380)), int(rng.uniform(180, 300)))
        color = tuple(int(v) for v in rng.uniform(200, 245, 3))
        cv2.ellipse(img, (w // 2, h // 2), axes, float(rng.uniform(0, 180)), 0, 360, color, -1)
    elif kind == "무늬 배경":
        img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), float(rng.uniform(3, 6)))
    return img


def save_examples(path: str) -> None:
    """가림 장면별 대표 1장씩: 초록 = 정답, 빨강 = 검출"""
    tiles = []
    for name in list(SCENES)[1:]:
        for i in range(40):
            img, gt = occlusion_scene(name, i)
            found = detect(img)
            if (i % 2 == 0) == (name != "접힌 모서리") or found is None:  # 피부색·어두운 손가락 섞어서 고른다
                continue
            cv2.polylines(img, [gt.astype(np.int32)], True, (60, 200, 60), 6)
            cv2.polylines(img, [found.astype(np.int32)], True, (40, 40, 230), 3)
            tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
            break
    cv2.imwrite(path, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40

    items, source = load_dataset(synthetic=wants_synthetic())
    print(f"## 1) 조건별 세트: {source}")
    by_tag: dict[str, list[str]] = {}
    for _, img, gt, tags in items:
        if gt is None:
            continue
        r = judge(detect(img), gt)
        for t in tags or ["(태그 없음)"]:
            by_tag.setdefault(t, []).append(r)
    total = [r for rs in by_tag.values() for r in rs]
    for t, rs in by_tag.items():
        print(f"  {t:8s} 성공 {rs.count('ok')}/{len(rs)}  틀림 {rs.count('wrong')}")
    print(f"  합계     성공 {total.count('ok')}/{len(total)}  틀림 {total.count('wrong')}")

    print(f"\n## 2) 가림 세트 (장면당 {n}장, 1200x900, 문서 위치·각도 무작위)")
    print("  장면            성공  틀린 사각형  못 찾음  성공 시 평균 오차(px)")
    for name in SCENES:
        rs, errs = [], []
        for i in range(n):
            img, gt = occlusion_scene(name, i)
            found = detect(img)
            r = judge(found, gt)
            rs.append(r)
            if r == "ok":
                errs.append(float(np.linalg.norm(found - gt, axis=1).mean()))
        mean_err = f"{np.mean(errs):.1f}" if errs else "-"
        print(f"  {name:14s} {rs.count('ok'):3d}  {rs.count('wrong'):9d}  {rs.count('miss'):7d}  {mean_err:>8s}")

    print(f"\n## 3) 문서 없는 장면 (종류당 {n}장) 잘못 검출 / find_document 시간")
    for kind in ["빈 배경", "원·타원 물체", "무늬 배경"]:
        fp = sum(detect(no_document(kind, i)) is not None for i in range(n))
        print(f"  {kind:10s} {fp}/{n}")
    for label, img in [("문서 있음", occlusion_scene("손가락 1개 (변)", 0)[0]), ("문서 없음", no_document("무늬 배경", 0))]:
        edges = detect_edges(preprocess(img)[0])
        find_document(edges)
        t = time.perf_counter()
        for _ in range(200):
            find_document(edges)
        print(f"  {label}: {(time.perf_counter() - t) / 200 * 1000:.2f} ms (480px)")

    if "--img" in sys.argv:
        save_examples(sys.argv[sys.argv.index("--img") + 1])


if __name__ == "__main__":
    main()

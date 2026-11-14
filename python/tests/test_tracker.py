"""CornerTracker / KalmanCornerTracker 테스트 (담당: 신승준)

app/test/tracker_test.dart 와 같은 상황을 확인한다.
"""
import numpy as np
import pytest

from scanner import CornerTracker, KalmanCornerTracker

Q = np.array([[120, 60], [350, 80], [380, 320], [90, 300]], np.float32)  # 처리 이미지(480px) 좌표
TRACKERS = [CornerTracker, KalmanCornerTracker]


@pytest.mark.parametrize("cls", TRACKERS)
def test_becomes_stable_when_still(cls):
    t = cls(stable_frames=5)
    rng = np.random.default_rng(0)
    stable = False
    for _ in range(10):
        _, stable = t.update(Q + rng.uniform(-1, 1, Q.shape))
    assert stable
    assert t.stability == 1.0


@pytest.mark.parametrize("cls", TRACKERS)
def test_dropouts(cls):
    t = cls(max_missed=2)
    t.update(Q)
    assert t.update(None)[0] is not None
    t.update(None)
    assert t.update(None)[0] is None


@pytest.mark.parametrize("cls", TRACKERS)
def test_single_frame_outlier_ignored(cls):
    """한 프레임만 엉뚱한 사각형(손, 다른 종이)이 잡히면 결과를 바꾸지 않는다"""
    t = cls(stable_frames=3)
    for _ in range(5):
        t.update(Q)
    out, stable = t.update(Q + 150)
    np.testing.assert_allclose(out, Q, atol=1e-3)
    assert not stable
    out, _ = t.update(Q)  # 원래 사각형으로 돌아오면 그대로 이어서
    np.testing.assert_allclose(out, Q, atol=1e-3)


@pytest.mark.parametrize("cls", TRACKERS)
def test_switches_after_jump_frames(cls):
    """다른 사각형이 jump_frames 프레임 연속으로 잡히면 그쪽으로 바로 옮겨 간다 (중간값으로 끌려가지 않음)"""
    t = cls(jump_frames=3)
    for _ in range(5):
        t.update(Q)
    other = Q + 150
    for _ in range(2):
        out, _ = t.update(other)
        np.testing.assert_allclose(out, Q, atol=1e-3)
    out, stable = t.update(other)
    np.testing.assert_allclose(out, other, atol=1e-3)
    assert not stable


@pytest.mark.parametrize("cls", TRACKERS)
def test_scattered_outliers_do_not_switch(cls):
    """매번 다른 곳에 튀는 오검출은 연속으로 쌓이지 않는다"""
    t = cls(jump_frames=2)
    t.update(Q)
    for k in range(6):
        out, _ = t.update(Q + (150 if k % 2 else -150))
    np.testing.assert_allclose(out, Q, atol=1e-3)


def test_kalman_lags_less_than_ema_while_moving():
    """등속도로 움직이는 문서를 칼만 필터가 EMA 보다 가깝게 따라간다"""
    ema, kal = CornerTracker(), KalmanCornerTracker()
    step = np.array([4.0, 2.0], np.float32)
    for i in range(30):
        cur = Q + step * i
        e, _ = ema.update(cur)
        k, _ = kal.update(cur)
    lag_ema = np.linalg.norm(e - cur, axis=1).max()
    lag_kal = np.linalg.norm(k - cur, axis=1).max()
    assert lag_kal < 1.0 < lag_ema


def test_kalman_is_unit_independent():
    """좌표를 480으로 나눠 넣어도(Dart 처럼 0~1) 결과가 같은 비율이다"""
    rng = np.random.default_rng(1)
    meas = [Q + rng.normal(0, 1.5, Q.shape) for _ in range(20)]
    a = KalmanCornerTracker()
    b = KalmanCornerTracker(move_tol=4.0 / 480, jump_tol=40.0 / 480)
    for m in meas:
        oa, sa = a.update(m)
        ob, sb = b.update(m / 480)
        assert sa == sb
    np.testing.assert_allclose(oa / 480, ob, atol=1e-5)

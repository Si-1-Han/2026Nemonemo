"""[신승준] 꼭짓점 안정화 비교 (EMA / 칼만 필터 / 보정 없음) — 시뮬레이션

사용법:  python experiments/tracker_compare.py [--seeds 30] [--img 그림.png]

실제 영상 비교는 experiments/tracker_video.py (--kalman) 로 한다. 여기서는 정답 꼭짓점을 알고 있는
시뮬레이션으로 같은 조건을 여러 번 반복해 수치를 낸다. 좌표는 처리 이미지(480px) 기준 px, 1프레임 = 1/15초.

장면 A (정지 → 이동 → 정지): 0~29 정지, 30~59 초당 45px(3px/프레임)로 이동, 60~119 정지.
  정지 시 떨림   : 90~119 프레임에서 출력 꼭짓점의 프레임 간 이동 평균 (보정 전 = 측정값 자체)
  이동 중 뒤처짐 : 35~59 프레임에서 출력과 실제 위치의 거리 평균
  멈춘 직후 오차 : 멈춘 뒤 10프레임(60~69) 동안 출력과 실제 위치의 거리 평균 (EMA 는 뒤처짐, 칼만은 지나침)
  자동 촬영      : 멈춘 뒤 2초(30프레임) 안에 is_stable 이 된 비율 / 움직이는 중(40~59) 잘못 is_stable 이 된 비율

[제안] 정지 판정을 '최근 stable_frames 프레임 동안 보정된 꼭짓점이 평균 위치에서 반경 tol 안' 으로 바꾼
방식(WindowStill)도 같이 비교한다. 현재 방식(측정값과 보정값의 차이 < move_tol 이 연속 10프레임)은
네 꼭짓점 중 하나만 잡음으로 튀어도 카운트가 0으로 돌아가서, 잡음이 크면 자동 촬영이 거의 안 된다.
장면 B (오검출 한 프레임): 40 프레임만 150px 떨어진 엉뚱한 사각형 → 출력이 실제 위치에서 가장 멀어진 거리
장면 C (다른 문서로 전환): 40 프레임부터 150px 떨어진 문서 → 새 위치 (2px + 2σ) 안에 들어오기까지 프레임,
  그 사이 두 문서 어디에도 가깝지 않은(10px 넘게 떨어진) '중간 위치' 프레임 수
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner.tracker import CornerTracker, KalmanCornerTracker  # noqa: E402


class WindowStill:
    """[제안, 실험용] 정지 판정만 '최근 n 프레임 보정 꼭짓점의 퍼짐 < tol' 로 바꾼 래퍼"""

    def __init__(self, base: CornerTracker, tol: float = 3.0) -> None:
        self.base, self.tol, self.buf = base, tol, []

    def update(self, corners):
        out, _ = self.base.update(corners)
        if corners is None or out is None or self.base._pending is not None:
            self.buf = []
        else:
            self.buf = (self.buf + [out.copy()])[-self.base.stable_frames:]
        window = np.array(self.buf)
        still = len(self.buf) == self.base.stable_frames and float(
            np.linalg.norm(window - window.mean(axis=0), axis=-1).max()) < self.tol
        return out, still

BASE = np.array([[120, 60], [350, 80], [380, 320], [90, 300]], np.float32)
FPS = 15
SPEED = 3.0  # px/프레임


def trackers():
    return {
        "보정 없음": lambda: CornerTracker(alpha=1.0, jump_tol=float("inf")),
        "EMA a=0.5 (점프 처리 없음, PR #2 이전)": lambda: CornerTracker(jump_tol=float("inf")),
        "EMA a=0.5 (현재 기본)": lambda: CornerTracker(),
        "EMA a=0.3": lambda: CornerTracker(alpha=0.3),
        "칼만 q=0.02": lambda: KalmanCornerTracker(),
        "[제안] EMA a=0.5 + 창 정지 판정": lambda: WindowStill(CornerTracker()),
        "[제안] 칼만 + 창 정지 판정": lambda: WindowStill(KalmanCornerTracker()),
    }


def run(tracker, measurements):
    outs, stables = [], []
    for m in measurements:
        o, s = tracker.update(m)
        outs.append(None if o is None else o.copy())
        stables.append(s)
    return np.array(outs), np.array(stables)


def dist(a, b):
    return np.linalg.norm(a - b, axis=-1).max(axis=-1)  # 네 꼭짓점 중 최대


def scene_a(rng, sigma):
    truth, pos = [], BASE.copy()
    for t in range(120):
        if 30 <= t < 60:
            pos = pos + np.array([SPEED, SPEED * 0.5], np.float32)
        truth.append(pos.copy())
    truth = np.array(truth)
    return truth, truth + rng.normal(0, sigma, truth.shape).astype(np.float32)


def eval_a(make, sigma, seeds):
    rows = []
    for seed in range(seeds):
        truth, meas = scene_a(np.random.default_rng(seed), sigma)
        out, st = run(make(), list(meas))
        err = dist(out, truth)
        rows.append({
            "jitter": float(np.linalg.norm(np.diff(out[90:], axis=0), axis=-1).mean()),
            "lag": float(err[35:60].mean()),
            "settle": float(err[60:70].mean()),
            "capture": bool(st[60:60 + 2 * FPS].any()),
            "capture_frame": int(np.argmax(st[60:])) if st[60:].any() else None,
            "false_capture": bool(st[40:60].any()),
        })
    return rows


def eval_b(make, sigma, seeds):
    worst = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        truth = np.repeat(BASE[None], 80, axis=0)
        meas = list(truth + rng.normal(0, sigma, truth.shape).astype(np.float32))
        meas[40] = BASE + 150
        out, _ = run(make(), meas)
        worst.append(float(dist(out[40:], truth[40:]).max()))
    return float(np.mean(worst))


def eval_c(make, sigma, seeds):
    frames, middle = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        new = BASE + 150
        truth = np.array([BASE if t < 40 else new for t in range(80)])
        out, _ = run(make(), list(truth + rng.normal(0, sigma, truth.shape).astype(np.float32)))
        d_new, d_old = dist(out[40:], new), dist(out[40:], BASE)
        near = d_new < 2.0 + 2 * sigma
        frames.append(int(np.argmax(near)) if near.any() else 40)
        middle.append(int(((d_new > 10) & (d_old > 10)).sum()))
    return float(np.mean(frames)), float(np.mean(middle))


def main() -> None:
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 30
    for sigma in (0.5, 1.5, 3.0):
        print(f"\n## 측정 잡음 σ={sigma}px (480px 기준), {seeds}회 평균")
        print("  방식                                정지 떨림  이동 뒤처짐  멈춘 직후 오차  자동 촬영(2초)  이동 중 잘못 촬영")
        for name, make in trackers().items():
            r = eval_a(make, sigma, seeds)
            m = lambda k: np.mean([x[k] for x in r])  # noqa: E731
            print(f"  {name:34s} {m('jitter'):6.2f}px  {m('lag'):8.2f}px  {m('settle'):9.2f}px"
                  f"  {m('capture') * 100:9.0f}%  {m('false_capture') * 100:12.0f}%")
    print(f"\n## 오검출 한 프레임 / 다른 문서로 전환 (σ=1.5px, {seeds}회 평균)")
    print("  방식                                한 프레임 오검출 시 최대 이탈  전환 완료까지  중간 위치 프레임")
    for name, make in trackers().items():
        worst = eval_b(make, 1.5, seeds)
        f, mid = eval_c(make, 1.5, seeds)
        print(f"  {name:34s} {worst:12.1f}px  {f:12.1f}프레임  {mid:10.1f}")

    if "--img" in sys.argv:
        save_plot(sys.argv[sys.argv.index("--img") + 1])


def save_plot(path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f or "NanumGothic" in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    truth, meas = scene_a(np.random.default_rng(3), 1.5)
    t = np.arange(len(truth))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(t, meas[:, 0, 0], ".", color="#bbbbbb", ms=4, label="검출값 (보정 전)")
    ax.plot(t, truth[:, 0, 0], "--", color="#555555", lw=1, label="실제 위치")
    for name, color in [("EMA a=0.5 (현재 기본)", "#1f77b4"), ("칼만 q=0.02", "#d62728")]:
        out, _ = run(trackers()[name](), list(meas))
        ax.plot(t, out[:, 0, 0], color=color, lw=1.6, label=name.replace(" (현재 기본)", ""))
    ax.axvspan(30, 60, color="#f3f3f3", zorder=0)
    ax.set_xlabel("프레임 (15fps)")
    ax.set_ylabel("좌상단 x 좌표 (px, 480 기준)")
    ax.set_title("정지 → 이동 → 정지 (측정 잡음 σ=1.5px)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    rng = np.random.default_rng(3)
    truth = np.repeat(BASE[None], 70, axis=0)
    meas = list(truth + rng.normal(0, 1.5, truth.shape).astype(np.float32))
    for k in range(40, 70):
        meas[k] = BASE + 150 + rng.normal(0, 1.5, BASE.shape).astype(np.float32)
    meas[25] = BASE + 150
    t = np.arange(70)
    ax.plot(t, [m[0, 0] for m in meas], ".", color="#bbbbbb", ms=4, label="검출값")
    for name, color in [("EMA a=0.5 (점프 처리 없음, PR #2 이전)", "#999999"), ("EMA a=0.5 (현재 기본)", "#1f77b4")]:
        out, _ = run(trackers()[name](), meas)
        ax.plot(t, out[:, 0, 0], color=color, lw=1.6, label="점프 처리 없음" if "이전" in name else "점프 처리 (현재)")
    ax.annotate("한 프레임 오검출", (25, BASE[0, 0] + 150), xytext=(3, BASE[0, 0] + 120), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("다른 문서로 전환", (40, BASE[0, 0] + 150), xytext=(44, BASE[0, 0] + 40), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xlabel("프레임")
    ax.set_title("갑자기 다른 사각형이 잡힐 때")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()

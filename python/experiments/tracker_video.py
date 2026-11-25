"""[신승준] 동영상/웹캠으로 실시간 검출 + 안정화 실험

휴대폰으로 문서를 찍은 짧은 동영상(data/videos/*.mp4)이나 웹캠으로 앱과 같은 라이브 루프를 돌린다.
원본 검출 좌표와 보정 좌표의 프레임 간 흔들림(jitter)을 비교해 출력한다.

사용법:
    python experiments/tracker_video.py data/videos/sample.mp4
    python experiments/tracker_video.py 0          # 웹캠
    python experiments/tracker_video.py data/videos/sample.mp4 --kalman   # 칼만 필터로 비교
키: q 종료, space 일시정지
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from scanner import CornerTracker, KalmanCornerTracker, scan_frame  # noqa: E402


def jitter(seq):
    """연속 프레임 사이 꼭짓점 이동량(px)의 평균"""
    d = [np.linalg.norm(b - a, axis=1).mean() for a, b in zip(seq, seq[1:]) if a is not None and b is not None]
    return float(np.mean(d)) if d else float("nan")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_kalman = "--kalman" in sys.argv
    src = args[0] if args else "0"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    tracker = KalmanCornerTracker() if use_kalman else CornerTracker()
    raw_seq, smooth_seq, frames, detected, stable_frames = [], [], 0, 0, 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            fr = scan_frame(frame)
            smoothed, stable = tracker.update(fr.corners)
            frames += 1
            detected += fr.corners is not None
            stable_frames += stable
            raw_seq.append(fr.corners)
            smooth_seq.append(None if smoothed is None else smoothed.copy())

            view = frame.copy()
            if fr.corners is not None:
                cv2.polylines(view, [fr.corners_full.astype(np.int32)], True, (0, 0, 255), 1)
            if smoothed is not None:
                color = (0, 255, 0) if stable else (0, 200, 255)
                cv2.polylines(view, [(smoothed * fr.ratio).astype(np.int32)], True, color, 3)
            cv2.putText(view, "STABLE - capture!" if stable else "", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("tracker (red=raw, green/orange=smoothed)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    if frames:
        print(f"[{type(tracker).__name__}] 프레임 {frames}, 검출률 {detected / frames:.1%}, 안정 프레임 {stable_frames}")
        print(f"jitter(px, 처리 이미지 기준)  원본 {jitter(raw_seq):.2f}  →  보정 {jitter(smooth_seq):.2f}")


if __name__ == "__main__":
    main()

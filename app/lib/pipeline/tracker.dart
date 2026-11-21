// 실시간 안정화: 프레임 간 흔들림 보정 + 자동 촬영 판단 (담당: 신승준)
// Python 레퍼런스: python/scanner/tracker.py — 동작과 파라미터를 맞출 것.
//
// 좌표는 0~1 로 정규화된 값을 넣는 것을 권장한다. (moveTol, jumpTol 도 그 기준)
//
// - CornerTracker       : 지수 이동 평균(EMA). 기본.
// - KalmanCornerTracker : 등속도 칼만 필터. 움직일 때 덜 뒤처진다.
// 둘 다 갑자기 다른 사각형이 잡히면 jumpFrames 프레임 연속으로 같은 자리에 잡힐 때까지 기다렸다가 옮겨 간다.
import 'quad.dart';

class CornerTracker {
  CornerTracker({
    this.alpha = 0.5,
    this.moveTol = 4.0 / 480, // 처리 이미지 480px 기준 4px
    this.stableFrames = 10,
    this.maxMissed = 5,
    this.jumpTol = 40.0 / 480, // 처리 이미지 480px 기준 40px
    this.jumpFrames = 3,
  });

  final double alpha;
  final double moveTol;
  final int stableFrames;
  final int maxMissed;

  /// 현재 결과에서 이보다 멀리 떨어진 꼭짓점이 있으면 '다른 사각형'으로 본다.
  final double jumpTol;

  /// 다른 사각형이 이만큼 연속으로 같은 자리에 잡혀야 그쪽으로 옮겨 간다.
  final int jumpFrames;

  Quad? smoothed;
  int _stillCount = 0;
  int _missed = 0;
  Quad? _pending; // 옮겨 갈 후보 사각형
  int _pendingCount = 0;

  void reset() {
    smoothed = null;
    _stillCount = 0;
    _missed = 0;
    _pending = null;
    _pendingCount = 0;
  }

  /// 0~1, 자동 촬영까지 얼마나 안정됐는지 (UI 진행 표시용)
  double get stability => (_stillCount / stableFrames).clamp(0.0, 1.0);

  /// 새 프레임의 검출 결과를 반영한다. 반환: (보정된 꼭짓점, 안정 여부)
  (Quad?, bool) update(Quad? corners) {
    if (corners == null) {
      _missed++;
      _stillCount = 0;
      if (_missed > maxMissed) reset();
      return (smoothed, false);
    }
    _missed = 0;

    final prev = smoothed;
    if (prev == null) {
      _start(corners);
      return (smoothed, false);
    }

    final movement = corners.maxDistance(prev);

    if (movement > jumpTol) {
      // 다른 사각형: 같은 자리에 연속으로 잡히는지 지켜본다. 그동안 결과는 그대로 둔다.
      final pending = _pending;
      _pendingCount = (pending != null && corners.maxDistance(pending) <= jumpTol) ? _pendingCount + 1 : 1;
      _pending = corners;
      _stillCount = 0;
      if (_pendingCount >= jumpFrames) _start(corners);
      return (smoothed, false);
    }

    _pending = null;
    _pendingCount = 0;
    smoothed = _smooth(prev, corners);
    _stillCount = movement < moveTol ? _stillCount + 1 : 0;
    return (smoothed, _stillCount >= stableFrames);
  }

  // ---- 보정 방식별로 바꾸는 부분 ----

  /// 새 사각형으로 (다시) 시작한다.
  void _start(Quad corners) {
    smoothed = corners;
    _stillCount = 0;
    _pending = null;
    _pendingCount = 0;
  }

  Quad _smooth(Quad prev, Quad corners) => Quad([
        for (var i = 0; i < 4; i++) corners.pts[i] * alpha + prev.pts[i] * (1 - alpha),
      ]);
}

/// 꼭짓점 좌표 8개 각각에 등속도 칼만 필터를 건다. (Python KalmanCornerTracker 와 같은 식)
///
/// 측정 잡음 분산을 1 로 두고 processNoise 를 그에 대한 비율로 쓰므로 좌표 단위와 상관없이 같은 값을 쓴다.
class KalmanCornerTracker extends CornerTracker {
  KalmanCornerTracker({
    this.processNoise = 0.02,
    super.moveTol,
    super.stableFrames,
    super.maxMissed,
    super.jumpTol,
    super.jumpFrames,
  });

  /// 움직임 변화(가속) 잡음 / 측정 잡음. 클수록 반응 빠름, 흔들림 큼.
  final double processNoise;

  List<Pt> _vel = const [Pt(0, 0), Pt(0, 0), Pt(0, 0), Pt(0, 0)];
  // 공분산 [[p00, p01], [p01, p11]] — 모든 좌표가 같은 시점에 갱신되므로 하나만 둔다
  double _p00 = 1, _p01 = 0, _p11 = 1;

  void _resetFilter() {
    _vel = const [Pt(0, 0), Pt(0, 0), Pt(0, 0), Pt(0, 0)];
    _p00 = 1;
    _p01 = 0;
    _p11 = 1;
  }

  @override
  void reset() {
    super.reset();
    _resetFilter();
  }

  @override
  void _start(Quad corners) {
    super._start(corners);
    _resetFilter();
  }

  @override
  Quad _smooth(Quad prev, Quad corners) {
    final q = processNoise;
    // 예측: x' = x + v,  P' = F P Fᵀ + Q  (Q = q · [[1/4, 1/2], [1/2, 1]])
    final p00 = _p00 + 2 * _p01 + _p11 + q / 4;
    final p01 = _p01 + _p11 + q / 2;
    final p11 = _p11 + q;
    // 갱신: 측정 잡음 R = 1
    final s = p00 + 1;
    final k0 = p00 / s, k1 = p01 / s;

    final pos = <Pt>[];
    final vel = <Pt>[];
    for (var i = 0; i < 4; i++) {
      final predicted = prev.pts[i] + _vel[i];
      final resid = corners.pts[i] - predicted;
      pos.add(predicted + resid * k0);
      vel.add(_vel[i] + resid * k1);
    }
    _vel = vel;
    _p00 = (1 - k0) * p00;
    _p01 = (1 - k0) * p01;
    _p11 = p11 - k1 * p01;
    return Quad(pos);
  }
}

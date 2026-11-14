// 3~4단계: 문서 영역 검출 + 꼭짓점 정렬 (담당: 신승준)
// Python 레퍼런스: python/scanner/detect.py — 동작과 파라미터를 맞출 것.
//
// 1) 기본: approxPolyDP 로 꼭짓점 4개짜리 볼록 다각형을 찾는다.
// 2) 대안(1 실패 시): 볼록 껍질(convexHull)을 epsilon 을 키우며 근사한다.
//    4각형 → 채택, 5각형 → 잘린 모서리 복원, 안 되면 minAreaRect.
//    채움 비율(kFallbackMinFill)과 화면 가장자리(kFallbackBorderMargin)로 엉뚱한 결과를 거른다.
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'quad.dart';

const double kMinAreaRatio = 0.10;
const double kApproxEps = 0.02;
const int kTopK = 5;

// [대안] 껍질 근사 epsilon 을 kApproxEps 부터 이 값까지 이 간격으로 키워 본다.
const double kFallbackEpsMax = 0.04;
const double kFallbackEpsStep = 0.01;

// [대안] 채움 비율 하한 (원은 π/4 ≈ 0.785)
const double kFallbackMinFill = 0.85;

// [대안] 결과 꼭짓점이 화면 가장자리에서 이 거리(px, 처리 이미지 기준) 안이면 버린다.
const int kFallbackBorderMargin = 2;

/// 엣지 맵에서 꼭짓점 4개인 볼록 다각형을 찾는다. (정렬 전, 처리 이미지 좌표계)
Quad? findDocument(cv.Mat edges) {
  final minArea = kMinAreaRatio * edges.rows * edges.cols;
  final (contours, hierarchy) = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  hierarchy.dispose();

  final byArea = <(double, cv.VecPoint)>[
    for (final c in contours) (cv.contourArea(c), c),
  ]..sort((a, b) => b.$1.compareTo(a.$1));

  Quad? found;
  for (final (area, c) in byArea.take(kTopK)) {
    if (area < minArea) break;
    final peri = cv.arcLength(c, true);
    final approx = cv.approxPolyDP(c, kApproxEps * peri, true);
    if (approx.length == 4 && cv.isContourConvex(approx)) {
      found = Quad(_toPts(approx));
      approx.dispose();
      break;
    }
    approx.dispose();
  }

  found ??= _fallbackQuad(contours, minArea, edges.cols, edges.rows);
  contours.dispose();
  return found;
}

/// 볼록 껍질로 문서를 근사한다. 끊긴 윤곽선은 contourArea 가 거의 0 이므로 껍질 면적으로 순위를 매긴다.
Quad? _fallbackQuad(cv.VecVecPoint contours, double minArea, int imgW, int imgH) {
  final hulls = <(double, cv.VecPoint)>[];
  for (final c in contours) {
    final r = cv.boundingRect(c);
    final tooSmall = r.width * r.height < minArea; // 껍질 면적 <= 외접 사각형 면적
    r.dispose();
    if (tooSmall) continue;

    final hullMat = cv.convexHull(c);
    final hull = cv.VecPoint.fromMat(hullMat);
    hullMat.dispose();
    final area = cv.contourArea(hull);
    if (area >= minArea) {
      hulls.add((area, hull));
    } else {
      hull.dispose();
    }
  }
  hulls.sort((a, b) => b.$1.compareTo(a.$1));

  bool ok(List<Pt> q, double hullArea) => _acceptable(q, hullArea, imgW, imgH);

  final steps = ((kFallbackEpsMax - kApproxEps) / kFallbackEpsStep).round() + 1;
  Quad? found;
  candidates:
  for (final (hullArea, hull) in hulls.take(kTopK)) {
    final peri = cv.arcLength(hull, true);
    for (var i = 0; i < steps; i++) {
      final eps = kApproxEps + i * kFallbackEpsStep;
      final approx = cv.approxPolyDP(hull, eps * peri, true);
      final pts = _toPts(approx);
      approx.dispose();
      if (pts.length < 4) break;
      if (pts.length == 5) {
        final q = restoreCutCorner(pts);
        if (q != null && ok(q, hullArea)) {
          found = Quad(q);
          break candidates;
        }
      }
      if (pts.length == 4) {
        if (ok(pts, hullArea)) {
          found = Quad(pts);
          break candidates;
        }
        break;
      }
    }

    final rect = cv.minAreaRect(hull);
    final box = cv.boxPoints(rect);
    final boxPts = [for (final p in box) Pt(p.x, p.y)];
    box.dispose();
    rect.dispose();
    if (ok(boxPts, hullArea)) {
      found = Quad(boxPts);
      break;
    }
  }

  for (final (_, h) in hulls) {
    h.dispose();
  }
  return found;
}

List<Pt> _toPts(cv.VecPoint v) => [for (final p in v) Pt(p.x.toDouble(), p.y.toDouble())];

/// 껍질을 충분히 채우고, 꼭짓점이 화면 가장자리에 닿지 않을 것
bool _acceptable(List<Pt> q, double hullArea, int imgW, int imgH) {
  const m = kFallbackBorderMargin;
  for (final p in q) {
    if (p.x <= m || p.y <= m || p.x >= imgW - 1 - m || p.y >= imgH - 1 - m) return false;
  }
  return fillRatio(q, hullArea) >= kFallbackMinFill;
}

/// 4각형과 껍질의 면적 비 (작은 쪽 / 큰 쪽, 0~1)
double fillRatio(List<Pt> q, double hullArea) {
  final qa = polygonArea(q);
  final big = qa > hullArea ? qa : hullArea;
  if (big <= 0) return 0;
  return (qa < hullArea ? qa : hullArea) / big;
}

/// 신발끈 공식
double polygonArea(List<Pt> pts) {
  var s = 0.0;
  for (var i = 0; i < pts.length; i++) {
    final a = pts[i], b = pts[(i + 1) % pts.length];
    s += a.x * b.y - a.y * b.x;
  }
  return s.abs() / 2;
}

/// 5각형에서 가장 짧은 변을 '잘린 모서리'로 보고 양옆 변을 연장한 교점으로 꼭짓점을 복원한다.
/// 복원한 4각형이 볼록하지 않으면 null.
List<Pt>? restoreCutCorner(List<Pt> p) {
  assert(p.length == 5);
  var k = 0;
  var minLen = double.infinity;
  for (var i = 0; i < 5; i++) {
    final len = (p[(i + 1) % 5] - p[i]).length;
    if (len < minLen) {
      minLen = len;
      k = i;
    }
  }
  final corner = _lineIntersection(p[(k + 4) % 5], p[k], p[(k + 1) % 5], p[(k + 2) % 5]);
  if (corner == null) return null;
  final q = [p[(k + 2) % 5], p[(k + 3) % 5], p[(k + 4) % 5], corner];
  return _isConvex(q) ? q : null;
}

/// 직선 a0-a1 과 b0-b1 의 교점. 거의 평행하면 null.
Pt? _lineIntersection(Pt a0, Pt a1, Pt b0, Pt b1) {
  final da = a1 - a0, db = b1 - b0;
  final denom = da.x * db.y - da.y * db.x;
  if (denom.abs() < 1e-9) return null;
  final t = ((b0.x - a0.x) * db.y - (b0.y - a0.y) * db.x) / denom;
  return a0 + da * t;
}

/// 네 꼭짓점에서 외적의 부호가 모두 같으면 볼록
bool _isConvex(List<Pt> q) {
  var pos = 0, neg = 0;
  for (var i = 0; i < 4; i++) {
    final o = q[i], a = q[(i + 1) % 4], b = q[(i + 2) % 4];
    final cross = (a.x - o.x) * (b.y - a.y) - (a.y - o.y) * (b.x - a.x);
    if (cross > 0) pos++;
    if (cross < 0) neg++;
  }
  return pos == 4 || neg == 4;
}

/// 좌상, 우상, 우하, 좌하 순서로 정렬 (좌표의 합과 차 이용)
Quad orderCorners(Quad q) {
  Pt pick(double Function(Pt) key, {required bool max}) {
    var best = q.pts.first;
    for (final p in q.pts) {
      if (max ? key(p) > key(best) : key(p) < key(best)) best = p;
    }
    return best;
  }

  double sum(Pt p) => p.x + p.y;
  double diff(Pt p) => p.y - p.x;
  return Quad([
    pick(sum, max: false), // 좌상
    pick(diff, max: false), // 우상
    pick(sum, max: true), // 우하
    pick(diff, max: true), // 좌하
  ]);
}

// 3~4단계: 문서 영역 검출 + 꼭짓점 정렬 (담당: 신승준)
// Python 레퍼런스: python/scanner/detect.py — 동작과 파라미터를 맞출 것.
//
// 면적(볼록 껍질 기준)이 큰 윤곽선부터 차례로
// 1) 긴 변 4개로 복원: 윤곽선을 선분으로 나눠 같은 직선끼리 묶고, 가장 긴 직선 4개의 교점을 꼭짓점으로 쓴다.
//    손가락이 변을 파거나 튀어나와도, 모서리를 가리거나 접혀도 네 변의 나머지는 남는다.
//    네 변이 윤곽선으로 덮인 비율(kSideMinCoverage), 꼭짓점 밖으로 삐져나간 선분(kSideMaxOverhang),
//    화면 가장자리(kSideBorderMargin)로 엉뚱한 결과를 거른다.
// 2) 1)이 실패하면 기존 approxPolyDP 4각형 근사
import 'dart:math' as math;

import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'quad.dart';

const double kMinAreaRatio = 0.10;
const double kApproxEps = 0.02;
const int kTopK = 5;

// [방법 1] 선분으로 나눌 때의 촘촘한 근사 epsilon (둘레 대비)
const double kSideEps = 0.005;

// [방법 1] 같은 직선으로 묶는 기준: 각도 차(도), 직선까지 거리(처리 이미지 대각선 대비)
const double kSideAngleTol = 6.0;
const double kSideDistTol = 0.015;

// [방법 1] 변으로 인정할 최소 길이 (둘레/4 대비)
const double kSideMinRatio = 0.15;

// [방법 1] 복원한 4각형의 네 변 위에 실제 선분이 덮고 있는 길이 / 4각형 둘레 의 하한
const double kSideMinCoverage = 0.8;

// [방법 1] 변과 같은 직선 위인데 꼭짓점 밖으로 삐져나간 선분 길이 / 변 길이 의 상한
const double kSideMaxOverhang = 0.25;

// [방법 1] 꼭짓점이 화면 가장자리에서 이 거리(px, 처리 이미지 기준) 안이면 버린다.
const int kSideBorderMargin = 2;

/// 엣지 맵에서 꼭짓점 4개인 볼록 다각형을 찾는다. (정렬 전, 처리 이미지 좌표계)
Quad? findDocument(cv.Mat edges) {
  final w = edges.cols, h = edges.rows;
  final minArea = kMinAreaRatio * w * h;
  final (contours, hierarchy) = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  hierarchy.dispose();

  // 끊기거나 파인 윤곽선은 contourArea 가 작게 나오므로 볼록 껍질 면적으로 순위를 매긴다.
  final candidates = <(double, cv.VecPoint)>[];
  for (final c in contours) {
    final r = cv.boundingRect(c);
    final tooSmall = r.width * r.height < minArea; // 껍질 면적 <= 외접 사각형 면적
    r.dispose();
    if (tooSmall) continue;
    final hullMat = cv.convexHull(c);
    final hull = cv.VecPoint.fromMat(hullMat);
    hullMat.dispose();
    final hullArea = cv.contourArea(hull);
    hull.dispose();
    if (hullArea >= minArea) candidates.add((hullArea, c));
  }
  candidates.sort((a, b) => b.$1.compareTo(a.$1));

  Quad? found;
  for (final (_, c) in candidates.take(kTopK)) {
    final q = quadFromSides(c, w, h);
    if (q != null) {
      found = Quad(q);
      break;
    }
    if (cv.contourArea(c) >= minArea) {
      final approx = cv.approxPolyDP(c, kApproxEps * cv.arcLength(c, true), true);
      final ok = approx.length == 4 && cv.isContourConvex(approx);
      final pts = _toPts(approx);
      approx.dispose();
      if (ok) {
        found = Quad(pts);
        break;
      }
    }
  }
  contours.dispose();
  return found;
}

List<Pt> _toPts(cv.VecPoint v) => [for (final p in v) Pt(p.x.toDouble(), p.y.toDouble())];

class _Group {
  _Group(this.nx, this.ny, this.offset, this.length, this.segments);
  final double nx, ny; // 단위 법선
  final double offset; // 법선 방향 거리
  double length; // 누적 길이
  final List<(Pt, Pt)> segments;
}

/// 직선 = 지나는 점 + 단위 방향 + 직선을 이루는 선분들
typedef _Line = (Pt, Pt, List<(Pt, Pt)>);

/// 윤곽선에서 가장 긴 직선 4개를 찾아 이웃한 직선의 교점을 돌려준다. 실패하면 null.
List<Pt>? quadFromSides(cv.VecPoint contour, int imgW, int imgH) {
  final peri = cv.arcLength(contour, true);
  final approx = cv.approxPolyDP(contour, kSideEps * peri, true);
  final poly = _toPts(approx);
  approx.dispose();
  final n = poly.length;
  if (n < 4) return null;

  // 1. 선분을 같은 직선끼리 묶는다
  final angleTol = math.cos(kSideAngleTol * math.pi / 180);
  final distTol = kSideDistTol * math.sqrt(imgW * imgW + imgH * imgH);
  final groups = <_Group>[];
  for (var i = 0; i < n; i++) {
    final p = poly[i], q = poly[(i + 1) % n];
    final length = (q - p).length;
    if (length < 1e-6) continue;
    final nx = (p.y - q.y) / length, ny = (q.x - p.x) / length;
    final mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
    _Group? match;
    for (final g in groups) {
      if ((g.nx * nx + g.ny * ny).abs() >= angleTol && (g.nx * mx + g.ny * my - g.offset).abs() <= distTol) {
        match = g;
        break;
      }
    }
    if (match != null) {
      match.length += length;
      match.segments.add((p, q));
    } else {
      groups.add(_Group(nx, ny, nx * mx + ny * my, length, [(p, q)]));
    }
  }
  groups.sort((a, b) => b.length.compareTo(a.length));
  if (groups.length < 4 || groups[3].length < kSideMinRatio * peri / 4) return null;

  // 2. 각 묶음의 선분들에 직선을 맞춘다
  final lines = [for (final g in groups.take(4)) _fitLine(g.segments)];

  // 3. 문서 중심에서 본 방향 순서로 정렬하고 이웃한 변의 교점을 구한다
  var cx = 0.0, cy = 0.0;
  for (final p in poly) {
    cx += p.x;
    cy += p.y;
  }
  final center = Pt(cx / n, cy / n);
  double direction(_Line l) {
    final (point, d, _) = l;
    final t = (center.x - point.x) * d.x + (center.y - point.y) * d.y;
    final foot = point + d * t;
    return math.atan2(foot.y - center.y, foot.x - center.x);
  }

  lines.sort((a, b) => direction(a).compareTo(direction(b)));
  final quad = <Pt>[];
  for (var i = 0; i < 4; i++) {
    final x = _intersect(lines[i], lines[(i + 1) % 4]);
    if (x == null) return null;
    quad.add(x);
  }
  if (!_isConvex(quad) || _touchesBorder(quad, imgW, imgH)) return null;

  // 4. 각 변(꼭짓점 i-1 ~ i) 위에 실제 선분이 얼마나 있는지 센다.
  //    꼭짓점 밖으로 많이 삐져나간 선분이 있으면 윤곽의 바깥 테두리가 아니다 (L자 모양 등).
  var covered = 0.0;
  for (var i = 0; i < 4; i++) {
    final (point, d, segments) = lines[i];
    double proj(Pt p) => (p.x - point.x) * d.x + (p.y - point.y) * d.y;
    final ta = proj(quad[(i + 3) % 4]), tb = proj(quad[i]);
    final lo = math.min(ta, tb), hi = math.max(ta, tb);
    var overhang = 0.0;
    for (final (p, q) in segments) {
      final a = proj(p), b = proj(q);
      final t1 = math.min(a, b), t2 = math.max(a, b);
      covered += math.max(0.0, math.min(hi, t2) - math.max(lo, t1));
      overhang += math.max(0.0, math.min(t2, lo - distTol) - t1) + math.max(0.0, t2 - math.max(t1, hi + distTol));
    }
    if (overhang > kSideMaxOverhang * (hi - lo)) return null;
  }
  return covered >= kSideMinCoverage * _perimeter(quad) ? quad : null;
}

/// 선분들 위에 고르게 퍼진 점들에 대한 최소제곱 직선. (Python _fit_line 과 같은 식)
_Line _fitLine(List<(Pt, Pt)> segments) {
  var total = 0.0, mx = 0.0, my = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (final (p, q) in segments) {
    final l = (q - p).length;
    total += l;
    mx += l * (p.x + q.x) / 2;
    my += l * (p.y + q.y) / 2;
    sxx += l * ((p.x * p.x + q.x * q.x) / 3 + (p.x * q.x) / 3);
    syy += l * ((p.y * p.y + q.y * q.y) / 3 + (p.y * q.y) / 3);
    sxy += l * ((p.x * p.y + q.x * q.y) / 3 + (p.x * q.y + q.x * p.y) / 6);
  }
  mx /= total;
  my /= total;
  final cxx = sxx / total - mx * mx;
  final cyy = syy / total - my * my;
  final cxy = sxy / total - mx * my;
  final theta = 0.5 * math.atan2(2 * cxy, cxx - cyy);
  return (Pt(mx, my), Pt(math.cos(theta), math.sin(theta)), segments);
}

/// 두 직선의 교점. 거의 평행하면 null.
Pt? _intersect(_Line a, _Line b) {
  final (pa, da, _) = a;
  final (pb, db, _) = b;
  final denom = da.x * db.y - da.y * db.x;
  if (denom.abs() < 1e-6) return null;
  final t = ((pb.x - pa.x) * db.y - (pb.y - pa.y) * db.x) / denom;
  return pa + da * t;
}

bool _touchesBorder(List<Pt> q, int imgW, int imgH) {
  const m = kSideBorderMargin;
  return q.any((p) => p.x <= m || p.y <= m || p.x >= imgW - 1 - m || p.y >= imgH - 1 - m);
}

double _perimeter(List<Pt> q) {
  var s = 0.0;
  for (var i = 0; i < q.length; i++) {
    s += (q[(i + 1) % q.length] - q[i]).length;
  }
  return s;
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

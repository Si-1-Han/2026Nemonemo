// 5단계: 원근 변환 (담당: 채성우)
// Python 레퍼런스: python/scanner/warp.py — 동작과 파라미터를 맞출 것.
import 'dart:math' as math;

import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'quad.dart';

/// 정렬된 꼭짓점(원본 좌표계)으로 문서를 펴고 선택적으로 출력 비율을 맞춘다.
/// [aspectRatio]는 width / height (세로 A4: 1 / 1.4142)이다.
cv.Mat warpDocument(cv.Mat img, Quad corners, {double? aspectRatio}) {
  final measuredWidth = math.max((corners.br - corners.bl).length, (corners.tr - corners.tl).length);
  final measuredHeight = math.max((corners.tr - corners.br).length, (corners.tl - corners.bl).length);
  late final int width;
  late final int height;
  if (aspectRatio == null) {
    width = measuredWidth.round();
    height = measuredHeight.round();
  } else {
    if (!aspectRatio.isFinite || aspectRatio <= 0) {
      throw ArgumentError.value(aspectRatio, 'aspectRatio', '유한한 양수여야 합니다 (width / height).');
    }
    // 측정된 면적을 유지하며 표준 비율로 출력 크기를 조정한다.
    final area = measuredWidth * measuredHeight;
    width = math.sqrt(area * aspectRatio).round();
    height = math.sqrt(area / aspectRatio).round();
  }
  final w = math.max(width, 1), h = math.max(height, 1);
  final src = cv.VecPoint2f.fromList([for (final p in corners.pts) cv.Point2f(p.x, p.y)]);
  final dst = cv.VecPoint2f.fromList([
    cv.Point2f(0, 0),
    cv.Point2f(w - 1.0, 0),
    cv.Point2f(w - 1.0, h - 1.0),
    cv.Point2f(0, h - 1.0),
  ]);
  final m = cv.getPerspectiveTransform2f(src, dst);
  final warped = cv.warpPerspective(img, m, (w, h), flags: cv.INTER_LINEAR);
  src.dispose();
  dst.dispose();
  m.dispose();
  return warped;
}

// 문서 검출 '긴 변 4개로 복원' 테스트 (담당: 신승준)
// Python: python/tests/test_detect.py 와 같은 상황
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:nemonemo/pipeline/detect.dart';
import 'package:nemonemo/pipeline/quad.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

cv.VecPoint contourOf(List<List<int>> pts) => cv.VecPoint.fromList([for (final p in pts) cv.Point(p[0], p[1])]);

void main() {
  test('한 변이 파이고 모서리 하나가 잘린 윤곽선에서 원래 네 꼭짓점을 복원한다', () {
    final contour = contourOf([
      [100, 100], [300, 100], [300, 150], [360, 150], [360, 100], [500, 100], // 위 변이 파임
      [500, 380], [480, 400], // 우하단이 잘림
      [100, 400],
    ]);
    final q = quadFromSides(contour, 600, 500);
    contour.dispose();
    expect(q, isNotNull);
    final ordered = orderCorners(Quad(q!));
    final expected = Quad(const [Pt(100, 100), Pt(500, 100), Pt(500, 400), Pt(100, 400)]);
    expect(ordered.maxDistance(expected), lessThan(0.5));
  });

  test('원은 4각형으로 복원하지 않는다', () {
    final contour = contourOf([
      for (var i = 0; i < 120; i++)
        [(300 + 150 * math.cos(i * math.pi / 60)).round(), (250 + 150 * math.sin(i * math.pi / 60)).round()],
    ]);
    expect(quadFromSides(contour, 600, 500), isNull);
    contour.dispose();
  });

  test('L자 모양은 둘레가 비어 있어 버린다', () {
    final contour = contourOf([
      [100, 100], [250, 100], [250, 250], [400, 250], [400, 400], [100, 400],
    ]);
    expect(quadFromSides(contour, 600, 500), isNull);
    contour.dispose();
  });
}

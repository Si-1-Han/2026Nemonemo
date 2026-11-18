// 파이프라인 통합 (담당: 전성재)
// Python 레퍼런스: python/scanner/pipeline.py
//
//  - detectFrame   : 라이브 루프. 1~4단계. 매 프레임 (저해상도)
//  - processCapture: 촬영 시. 고해상도 사진에서 다시 검출 → 5~6단계
import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'detect.dart';
import 'edges.dart';
import 'enhance.dart';
import 'preprocess.dart';
import 'quad.dart';
import 'warp.dart';

/// 1~4단계. 반환되는 꼭짓점은 입력 이미지(img) 좌표계.
Quad? detectFrame(cv.Mat img) {
  final (gray, ratio) = preprocess(img);
  final edges = detectEdges(gray);
  final found = findDocument(edges);
  gray.dispose();
  edges.dispose();
  if (found == null) return null;
  return orderCorners(found).scale(ratio);
}

class CaptureResult {
  const CaptureResult({required this.warpedJpeg, required this.corners, required this.redetected});

  /// 원근 변환까지 끝난 컬러 이미지(JPEG). 모드를 바꿀 때마다 여기서 다시 enhance 한다.
  /// (cv.Mat 은 Isolate 사이에 보낼 수 없어서 바이트로 주고받는다)
  final Uint8List warpedJpeg;
  final Quad corners;

  /// 고해상도 사진에서 다시 검출에 성공했는지 (false 면 프리뷰 좌표를 사용)
  final bool redetected;
}

/// 촬영한 JPEG 바이트를 처리한다. Isolate 안에서 호출할 것.
///
/// [previewCorners]: 프리뷰에서 추적하던 꼭짓점 (0~1 정규화). 재검출 실패 시 사용.
CaptureResult? processCapture(Uint8List jpeg, Quad? previewCorners, {double? aspectRatio}) {
  // imdecode 는 EXIF 회전 정보를 반영해 똑바로 세운 이미지를 준다.
  final img = cv.imdecode(jpeg, cv.IMREAD_COLOR);
  if (img.isEmpty) return null;

  var corners = detectFrame(img);
  final redetected = corners != null;
  corners ??= previewCorners?.scale(img.cols.toDouble(), img.rows.toDouble());
  if (corners == null) {
    img.dispose();
    return null;
  }

  final warped = warpDocument(img, corners, aspectRatio: aspectRatio);
  img.dispose();
  final bytes = encodeJpeg(warped, quality: 95);
  warped.dispose();
  return CaptureResult(warpedJpeg: bytes, corners: corners, redetected: redetected);
}

/// 결과 이미지를 JPEG 바이트로 인코딩
Uint8List encodeJpeg(cv.Mat img, {int quality = 92}) {
  final params = cv.VecI32.fromList([cv.IMWRITE_JPEG_QUALITY, quality]);
  final (ok, bytes) = cv.imencode('.jpg', img, params: params);
  params.dispose();
  if (!ok) throw StateError('JPEG 인코딩 실패');
  return bytes;
}

/// 원근 변환된 이미지에 모드를 적용해 JPEG 로 돌려준다. Isolate 안에서 호출할 것.
Uint8List renderResult(Uint8List warpedJpeg, ScanMode mode) {
  final warped = cv.imdecode(warpedJpeg, cv.IMREAD_COLOR);
  final out = enhance(warped, mode);
  final bytes = encodeJpeg(out);
  warped.dispose();
  out.dispose();
  return bytes;
}

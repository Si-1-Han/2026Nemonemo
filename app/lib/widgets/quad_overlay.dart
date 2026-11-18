// 프리뷰 위에 검출된 사각형을 그리는 위젯 (담당: 전성재)
import 'package:material_ui/material_ui.dart';

import '../pipeline/quad.dart';

class QuadOverlay extends StatelessWidget {
  const QuadOverlay({super.key, required this.quad, required this.stability});

  /// 0~1 정규화 좌표
  final Quad? quad;

  /// 0~1, 1이면 자동 촬영 직전
  final double stability;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _QuadPainter(quad, stability), size: Size.infinite);
  }
}

class _QuadPainter extends CustomPainter {
  _QuadPainter(this.quad, this.stability);

  final Quad? quad;
  final double stability;

  @override
  void paint(Canvas canvas, Size size) {
    final q = quad;
    if (q == null) return;

    final path = Path()..moveTo(q.tl.x * size.width, q.tl.y * size.height);
    for (final p in [q.tr, q.br, q.bl]) {
      path.lineTo(p.x * size.width, p.y * size.height);
    }
    path.close();

    final color = Color.lerp(Colors.amber, Colors.greenAccent, stability)!;
    canvas.drawPath(path, Paint()..color = color.withValues(alpha: 0.25));
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );
    for (final p in q.pts) {
      canvas.drawCircle(Offset(p.x * size.width, p.y * size.height), 7, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(_QuadPainter old) => old.quad != quad || old.stability != stability;
}

// 실시간 스캔 화면 (담당: 전성재)
import 'dart:io' show Platform;
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:material_ui/material_ui.dart';

import '../camera/frame_processor.dart';
import '../pipeline/quad.dart';
import '../pipeline/scanner.dart';
import '../pipeline/tracker.dart';
import '../widgets/quad_overlay.dart';
import 'result_screen.dart';

enum _PaperRatio { detected, a4Portrait, a4Landscape }

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  String? _error;

  final _processor = FrameProcessor();
  final _tracker = CornerTracker();
  Quad? _quad;
  bool _autoCapture = true;
  bool _capturing = false;
  bool _showStats = true;
  _PaperRatio _paperRatio = _PaperRatio.detected;
  double? get _aspectRatio => switch (_paperRatio) {
        _PaperRatio.detected => null,
        _PaperRatio.a4Portrait => 1 / 1.41421356237,
        _PaperRatio.a4Landscape => 1.41421356237,
      };

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final c = _controller;
    if (c == null || !c.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      c.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high, // 스트림 해상도. 처리 전에 480px 로 줄이므로 high 면 충분
        enableAudio: false,
        imageFormatGroup: Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      await controller.setFlashMode(FlashMode.off);
      if (!mounted) return;
      setState(() => _controller = controller);
      await _startStream();
    } on CameraException catch (e) {
      setState(() => _error = '카메라를 열 수 없습니다: ${e.description ?? e.code}');
    } catch (e) {
      setState(() => _error = '카메라 초기화 오류: $e');
    }
  }

  Future<void> _startStream() async {
    final c = _controller;
    if (c == null || c.value.isStreamingImages) return;
    _tracker.reset();
    _processor.resetStats();
    // 앱은 세로 고정. 센서 방향만큼 돌리면 똑바로 선 이미지가 된다.
    // TODO(전성재): 기종별로 회전이 맞는지 실기기에서 확인 (특히 iOS)
    final rotation = c.description.sensorOrientation;
    await c.startImageStream((image) async {
      final (processed, corners) = await _processor.process(image, rotation);
      if (!processed || !mounted || _capturing) return;
      final (smoothed, stable) = _tracker.update(corners);
      setState(() => _quad = smoothed);
      if (stable && _autoCapture) _capture();
    });
  }

  Future<void> _capture() async {
    final c = _controller;
    if (c == null || _capturing) return;
    setState(() => _capturing = true);
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
      final file = await c.takePicture();
      final jpeg = await file.readAsBytes();
      final preview = _quad;
      final aspectRatio = _aspectRatio;
      final result = await Isolate.run(() => processCapture(jpeg, preview, aspectRatio: aspectRatio));
      if (!mounted) return;
      if (result == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('문서를 찾지 못했어요. 다시 시도해 주세요.')),
        );
      } else {
        await Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => ResultScreen(capture: result)),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('촬영 오류: $e')));
      }
    } finally {
      if (mounted) {
        setState(() {
          _capturing = false;
          _quad = null;
        });
        await _startStream();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('네모네모 스캐너'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        actions: [
          PopupMenuButton<_PaperRatio>(
            tooltip: '문서 비율',
            initialValue: _paperRatio,
            onSelected: (ratio) => setState(() => _paperRatio = ratio),
            itemBuilder: (_) => const [
              PopupMenuItem<_PaperRatio>(value: _PaperRatio.detected, child: Text('검출 비율')),
              PopupMenuItem<_PaperRatio>(value: _PaperRatio.a4Portrait, child: Text('A4 세로')),
              PopupMenuItem<_PaperRatio>(value: _PaperRatio.a4Landscape, child: Text('A4 가로')),
            ],
            icon: const Icon(Icons.aspect_ratio),
          ),
          IconButton(
            tooltip: '성능 표시',
            icon: Icon(_showStats ? Icons.speed : Icons.speed_outlined),
            onPressed: () => setState(() => _showStats = !_showStats),
          ),
        ],
      ),
      body: _error != null
          ? Center(child: Text(_error!, style: const TextStyle(color: Colors.white)))
          : c == null || !c.value.isInitialized
              ? const Center(child: CircularProgressIndicator())
              : Column(
                  children: [
                    Expanded(
                      child: Center(
                        child: CameraPreview(
                          c,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              IgnorePointer(child: QuadOverlay(quad: _quad, stability: _tracker.stability)),
                              if (_showStats)
                                Positioned(
                                  left: 8,
                                  top: 8,
                                  child: _StatsBadge(
                                    fps: _processor.fps,
                                    ms: _processor.lastMs,
                                    dropped: _processor.dropped,
                                    detected: _quad != null,
                                  ),
                                ),
                              if (_capturing)
                                const ColoredBox(
                                  color: Colors.black45,
                                  child: Center(child: CircularProgressIndicator()),
                                ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    _BottomBar(
                      autoCapture: _autoCapture,
                      onAutoChanged: (v) => setState(() => _autoCapture = v),
                      onCapture: _capturing ? null : _capture,
                      aspectRatio: _aspectRatio,
                    ),
                  ],
                ),
    );
  }
}

class _StatsBadge extends StatelessWidget {
  const _StatsBadge({required this.fps, required this.ms, required this.dropped, required this.detected});
  final double fps;
  final double ms;
  final int dropped;
  final bool detected;

  @override
  Widget build(BuildContext context) => DecoratedBox(
        decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(6)),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Text(
            '${fps.toStringAsFixed(1)} fps · ${ms.toStringAsFixed(0)} ms · drop $dropped\n'
            '${detected ? '문서 검출됨' : '문서 찾는 중…'}',
            style: const TextStyle(color: Colors.white, fontSize: 12, height: 1.3),
          ),
        ),
      );
}

class _BottomBar extends StatelessWidget {
  const _BottomBar({required this.autoCapture, required this.onAutoChanged, required this.onCapture, required this.aspectRatio});
  final bool autoCapture;
  final ValueChanged<bool> onAutoChanged;
  final VoidCallback? onCapture;
  final double? aspectRatio;

  @override
  Widget build(BuildContext context) => SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Row(
            children: [
              Expanded(
                child: Row(children: [
                  Switch(value: autoCapture, onChanged: onAutoChanged),
                  const Text('자동 촬영', style: TextStyle(color: Colors.white)),
                ]),
              ),
              FloatingActionButton.large(
                onPressed: onCapture,
                backgroundColor: Colors.white,
                child: const Icon(Icons.camera_alt, color: Colors.black, size: 36),
              ),
              Expanded(
                child: Align(
                  alignment: Alignment.centerRight,
                  child: Text(
                    aspectRatio == null ? '검출 비율' : 'A4 ${aspectRatio! < 1 ? '세로' : '가로'}',
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
}

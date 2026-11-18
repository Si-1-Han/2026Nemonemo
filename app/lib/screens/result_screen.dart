// 스캔 결과 화면: 모드 선택, 저장/공유 (담당: 전성재, 화질 모드는 채성우)
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../pipeline/enhance.dart';
import '../pipeline/scanner.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.capture});
  final CaptureResult capture;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  ScanMode _mode = ScanMode.adaptive;
  final Map<ScanMode, Uint8List> _cache = {};
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _render(_mode);
  }

  Future<void> _render(ScanMode mode) async {
    setState(() => _mode = mode);
    if (_cache.containsKey(mode)) return;
    setState(() => _busy = true);
    final warped = widget.capture.warpedJpeg;
    final bytes = await Isolate.run(() => renderResult(warped, mode));
    if (!mounted) return;
    setState(() {
      _cache[mode] = bytes;
      _busy = false;
    });
  }

  Future<File> _saveFile() async {
    final dir = await getApplicationDocumentsDirectory();
    final scans = Directory('${dir.path}/scans')..createSync(recursive: true);
    final stamp = DateTime.now().toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
    final file = File('${scans.path}/scan_${stamp}_${_mode.name}.jpg');
    await file.writeAsBytes(_cache[_mode]!);
    return file;
  }

  Future<void> _share() async {
    final file = await _saveFile();
    await SharePlus.instance.share(ShareParams(files: [XFile(file.path)], text: '네모네모 스캔'));
  }

  @override
  Widget build(BuildContext context) {
    final image = _cache[_mode];
    return Scaffold(
      appBar: AppBar(
        title: const Text('스캔 결과'),
        actions: [
          IconButton(
            tooltip: '저장',
            icon: const Icon(Icons.save_alt),
            onPressed: image == null
                ? null
                : () async {
                    final f = await _saveFile();
                    if (context.mounted) {
                      ScaffoldMessenger.of(context)
                          .showSnackBar(SnackBar(content: Text('저장됨: ${f.path.split('/').last}')));
                    }
                  },
          ),
          IconButton(
            tooltip: '공유',
            icon: const Icon(Icons.share),
            onPressed: image == null ? null : _share,
          ),
        ],
      ),
      body: Column(
        children: [
          if (!widget.capture.redetected)
            const MaterialBanner(
              content: Text('고해상도 사진에서 다시 찾지 못해 프리뷰 위치로 보정했어요.'),
              actions: [SizedBox.shrink()],
            ),
          Expanded(
            child: Stack(
              alignment: Alignment.center,
              children: [
                if (image != null)
                  InteractiveViewer(maxScale: 5, child: Image.memory(image, gaplessPlayback: true)),
                if (_busy) const CircularProgressIndicator(),
              ],
            ),
          ),
          SafeArea(
            top: false,
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  for (final m in ScanMode.values)
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ChoiceChip(
                        label: Text(m.label),
                        selected: _mode == m,
                        onSelected: _busy ? null : (_) => _render(m),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

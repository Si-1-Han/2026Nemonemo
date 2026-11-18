import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

import 'screens/scan_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // 세로 고정: 카메라 프레임 회전 처리를 단순하게 하기 위함
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const NemoNemoApp());
}

class NemoNemoApp extends StatelessWidget {
  const NemoNemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '네모네모',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const ScanScreen(),
    );
  }
}

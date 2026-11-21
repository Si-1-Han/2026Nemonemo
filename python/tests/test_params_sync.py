"""Python ↔ Dart 파라미터 일치 검사

python/scanner/*.py 의 대문자 상수(예: SIGMA)와 app/lib/pipeline/*.dart 의 const(예: kSigma)가
같은 값인지, CornerTracker 기본값이 양쪽에서 같은지 확인한다.
한쪽만 바꾸면 이 테스트가 실패해서 PR 에서 바로 보인다. (CI 에서 자동 실행)

새 상수를 추가할 때: Python 은 UPPER_SNAKE, Dart 는 같은 이름의 kUpperCamel 로 만든다.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

from scanner.tracker import CornerTracker, KalmanCornerTracker

# scanner/__init__.py 가 preprocess 함수를 내보내서 모듈 이름이 가려지므로 importlib 로 가져온다
_pre = importlib.import_module("scanner.preprocess")

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python" / "scanner"
DART_DIR = ROOT / "app" / "lib" / "pipeline"
MODULES = ["preprocess", "edges", "detect", "enhance", "warp"]


def py_constants(name: str) -> dict[str, float]:
    tree = ast.parse((PY_DIR / f"{name}.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key.isupper() and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                out[key] = float(node.value.value)
    return out


def dart_constants(name: str) -> dict[str, float]:
    src = (DART_DIR / f"{name}.dart").read_text(encoding="utf-8")
    return {m[1]: float(m[2]) for m in re.finditer(r"const\s+(?:int|double)\s+(k\w+)\s*=\s*([-\d.]+)\s*;", src)}


def to_dart_name(py_name: str) -> str:
    return "k" + "".join(w.capitalize() for w in py_name.lower().split("_"))


def snake_to_camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w.capitalize() for w in rest)


@pytest.mark.parametrize("module", MODULES)
def test_module_constants_match(module):
    py, dart = py_constants(module), dart_constants(module)
    for key, value in py.items():
        dk = to_dart_name(key)
        assert dk in dart, f"{module}.dart 에 {dk} 가 없다 (Python {key} = {value})"
        assert dart[dk] == pytest.approx(value), f"{module}: Python {key}={value}, Dart {dk}={dart[dk]}"


# Python 은 처리 이미지 픽셀 단위, Dart 는 0~1 정규화 단위인 파라미터
PIXEL_PARAMS = {"move_tol", "jump_tol"}


def dart_tracker_defaults() -> dict[str, float]:
    src = (DART_DIR / "tracker.dart").read_text(encoding="utf-8")
    dart = {}
    for m in re.finditer(r"this\.(\w+)\s*=\s*([-\d.\s/*]+?)\s*,", src):
        dart[m[1]] = float(eval(m[2], {"__builtins__": {}}))  # 숫자 계산식만 허용
    return dart


def python_defaults(cls) -> dict[str, float]:
    return {name: float(p.default) for name, p in inspect.signature(cls.__init__).parameters.items()
            if name != "self" and p.default is not inspect.Parameter.empty}


def test_tracker_defaults_match():
    dart = dart_tracker_defaults()
    for name, value in python_defaults(CornerTracker).items():
        dk = snake_to_camel(name)
        assert dk in dart, f"tracker.dart 에 {dk} 기본값이 없다"
        expected = value / _pre.PROCESS_SIZE if name in PIXEL_PARAMS else value
        assert dart[dk] == pytest.approx(expected), f"tracker: Python {name}={value}, Dart {dk}={dart[dk]}"


def test_kalman_tracker_defaults_match():
    # Dart 는 공통 파라미터를 super.xxx 로 넘겨 CornerTracker 기본값을 그대로 쓴다.
    # Python 도 공통 파라미터 기본값이 CornerTracker 와 같아야 한다.
    base, kalman = python_defaults(CornerTracker), python_defaults(KalmanCornerTracker)
    for name, value in kalman.items():
        if name in base:
            assert value == base[name], f"KalmanCornerTracker.{name}={value}, CornerTracker.{name}={base[name]}"
    dart = dart_tracker_defaults()
    assert dart.get("processNoise") == pytest.approx(kalman["process_noise"])

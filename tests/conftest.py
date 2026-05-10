"""
Pytest 公共配置 — 确保 tests/ 下的测试可以正常 import `core.*`

作用域：
1. 自动把项目根目录与 src/ 加入 sys.path，让 `from core.xxx` 在 pytest 与脚本两种
   执行方式下都能解析。
2. 强制 stdout 使用 UTF-8，规避 Windows GBK 终端打印 emoji/中文出错。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# 兼容 Windows GBK 终端
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
elif sys.stdout and getattr(sys.stdout, "buffer", None):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

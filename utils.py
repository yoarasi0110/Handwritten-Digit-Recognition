"""路徑與評估報告寫入的工具"""

from __future__ import annotations

from pathlib import Path

#確保必要的資料夾存在
def ensure_dirs() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

#把準確率報告寫入檔案，預設路徑是 results/accuracy.txt
def write_accuracy_report(content: str, output_path: str = "results/accuracy.txt") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

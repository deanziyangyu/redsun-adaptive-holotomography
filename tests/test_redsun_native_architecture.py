from __future__ import annotations

import ast
from pathlib import Path

from redsun.view.qt import QtView

from redsun_aht.view.multishot import MultishotScanWidget

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "redsun_aht"


def _python_trees() -> list[tuple[Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in SOURCE_ROOT.rglob("*.py")
    ]


def test_application_code_does_not_import_bluesky_run_engine() -> None:
    violations: list[str] = []
    for path, tree in _python_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"bluesky", "bluesky.run_engine"}:
                continue
            if any(alias.name == "RunEngine" for alias in node.names):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert violations == []


def test_qt_application_views_derive_from_redsun_qt_view() -> None:
    assert issubclass(MultishotScanWidget, QtView)

    direct_qwidget_subclasses = [
        f"{path.relative_to(SOURCE_ROOT)}:{node.name}"
        for path, tree in _python_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
        if isinstance(base, ast.Attribute)
        and isinstance(base.value, ast.Name)
        and base.value.id == "QtWidgets"
        and base.attr == "QWidget"
    ]

    assert direct_qwidget_subclasses == []

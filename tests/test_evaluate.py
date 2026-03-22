# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from contextbench import evaluate


def test_tree_sitter_install_command_switches_by_python_version() -> None:
    assert (
        evaluate._tree_sitter_install_command((3, 11))
        == 'pip install "tree-sitter==0.20.4" tree-sitter-languages'
    )
    assert (
        evaluate._tree_sitter_install_command((3, 13))
        == 'pip install "tree-sitter>=0.24,<0.25" tree-sitter-language-pack'
    )


def test_main_uses_tree_sitter_install_hint(monkeypatch, capsys) -> None:
    monkeypatch.setattr("contextbench.extractors.available", lambda: False)
    monkeypatch.setattr(
        evaluate,
        "_tree_sitter_install_command",
        lambda version_info=None: "pip install tree-sitter-test-package",
    )
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "contextbench.evaluate",
            "--gold",
            "gold.parquet",
            "--pred",
            "pred.jsonl",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        evaluate.main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "ERROR: Tree-sitter not available" in err
    assert "pip install tree-sitter-test-package" in err

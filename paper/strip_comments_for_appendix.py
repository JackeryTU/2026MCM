from __future__ import annotations

import ast
import io
import shutil
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "支撑材料" / "01_程序代码" / "py"
TARGET = Path(__file__).resolve().parent / "code_appendix"


def docstring_spans(tree: ast.AST) -> set[int]:
    spans: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            spans.update(range(first.lineno, first.end_lineno + 1))
    return spans


def strip_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    removed_lines = docstring_spans(ast.parse(text))
    comments: dict[int, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.start[1]
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    for number, line in enumerate(lines, start=1):
        if number in removed_lines:
            continue
        if number in comments:
            line = line[: comments[number]].rstrip() + ("\n" if line.endswith("\n") else "")
        output.append(line)
    return "".join(output)


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()
    for source_file in sorted(SOURCE.glob("*.py")):
        cleaned = strip_source(source_file)
        ast.parse(cleaned)
        (TARGET / source_file.name).write_text(cleaned, encoding="utf-8")


if __name__ == "__main__":
    main()

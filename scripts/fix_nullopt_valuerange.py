#!/usr/bin/env python3
"""Fix std::nullopt -> ValueRange{} / scf::YieldOp(loc) for MLIR API updates."""
import re
from pathlib import Path

MIDEND = Path(__file__).resolve().parent.parent / "midend"


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    text = text.replace("ValueRange{std::nullopt}", "ValueRange{}")

    # scf::YieldOp(loc, std::nullopt) — single line
    text = re.sub(
        r"(\w+\.)?create<scf::YieldOp>\(\s*(\w+)\s*,\s*std::nullopt\s*\)",
        r"\1create<scf::YieldOp>(\2)",
        text,
    )
    # Multiline yield
    text = re.sub(
        r"(\w+\.)?create<scf::YieldOp>\(\s*(\w+)\s*,\s*\n\s*std::nullopt\s*\)",
        r"\1create<scf::YieldOp>(\2)",
        text,
    )

    lines = text.split("\n")
    for i in range(len(lines) - 1):
        cur = lines[i]
        nxt = lines[i + 1]
        if "std::nullopt" not in cur:
            continue
        if "No mapping" in cur or "No mapping specified" in cur:
            continue
        if "return std::nullopt" in cur:
            continue
        if "[&" not in nxt and " [&]( " not in nxt:
            # Affine/GPU: lambda may start same line
            if "[&" not in cur:
                continue
        if i > 0 and lines[i - 1].rstrip().endswith("ValueRange{},"):
            continue
        if "create<scf::YieldOp>" in cur:
            continue
        lines[i] = cur.replace("std::nullopt", "ValueRange{}", 1)

    text = "\n".join(lines)

    # Remaining: std::nullopt, before [&]( on next line (indent variants)
    text = re.sub(
        r"(\n[ \t]*)std::nullopt,([ \t]*\n[ \t]*\[&)",
        r"\1ValueRange{},\2",
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    for p in sorted(MIDEND.rglob("*.cpp")):
        if fix_file(p):
            print("updated", p.relative_to(MIDEND.parent))
            n += 1
    print("files changed:", n)


if __name__ == "__main__":
    main()

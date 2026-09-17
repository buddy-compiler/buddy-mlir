#!/usr/bin/env python3
"""Generate Unicode tables from the pinned tokenizers engine used for validation.

The regex and normalizer can use different Unicode versions. Python's isspace()
or unicodedata alone is therefore not an equivalent source. Classify all Unicode
scalars with tokenizers.Regex; derive decomposition/composition from its NFD/NFC.
Canonical combining classes are ordered with the same NFD implementation.
No Unicode library is needed by the resulting freestanding C implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

MAX = 0x110000


def scalars():
    return (cp for cp in range(MAX) if not 0xD800 <= cp <= 0xDFFF)


def collect():
    import tokenizers
    from tokenizers import Regex, normalizers, pre_tokenizers

    nfd, nfc = normalizers.NFD(), normalizers.NFC()
    tables = {}
    for name, pattern in (("letter", r"\p{L}"), ("number", r"\p{N}"),
                          ("space", r"\s")):
        split = pre_tokenizers.Split(Regex("[^" + pattern + "]+"), behavior="removed")
        rows = []
        for start, end in ((0, 0xD800), (0xE000, MAX)):
            text = "".join(map(chr, range(start, end)))
            rows += [(start + a, start + b - 1)
                     for _, (a, b) in split.pre_tokenize_str(text)]
        tables[name] = rows

    # Stable classes are used only as names for the order. The actual engine
    # decides whether each scalar is a nonstarter and which class it matches.
    representatives = {}
    for cp in scalars():
        c = chr(cp)
        klass = unicodedata.combining(c)
        if klass and nfd.normalize_str(c) == c:
            if nfd.normalize_str("\u0345" + c + "\u0334") != "\u0345" + c + "\u0334":
                representatives.setdefault(klass, c)
    ordered = sorted(representatives.items())

    def combining(c):
        low, high = 0, len(ordered)
        while low < high:
            mid = (low + high) // 2
            klass, ref = ordered[mid]
            if nfd.normalize_str(c + ref) != c + ref:
                low = mid + 1
            elif nfd.normalize_str(ref + c) != ref + c:
                high = mid
            else:
                return klass
        raise ValueError(f"new combining class has no representative: U+{ord(c):04X}")

    canonical, ccc, composition = [], [], []
    for cp in scalars():
        c = chr(cp)
        expanded = nfd.normalize_str(c)
        if expanded != c:
            # Hangul has an algorithmic decomposition/composition in C.
            if 0xAC00 <= cp < 0xAC00 + 11172:
                continue
            sequence = list(map(ord, expanded))
            if len(sequence) > 4:
                raise ValueError("canonical decomposition no longer fits four scalars")
            canonical.append((cp, sequence))
            if len(sequence) >= 2 and nfc.normalize_str(c) == c:
                first = nfc.normalize_str(expanded[:-1])
                if len(first) != 1:
                    raise ValueError(f"composition prefix not a scalar: {cp:x}")
                composition.append((ord(first), sequence[-1], cp))
        elif nfd.normalize_str("\u0345" + c + "\u0334") != "\u0345" + c + "\u0334":
            ccc.append((cp, combining(c)))
    tables.update(canonical=canonical, ccc=ccc, composition=sorted(composition))
    tables["provenance"] = {
        "tokenizers_version": tokenizers.__version__,
        "unicode_scalar_values_examined": MAX - 0x800,
        "regex_source": "tokenizers.Regex; exhaustive scalar classification",
        "normalization_source": "tokenizers.normalizers.NFD/NFC; exhaustive scalar scan",
        "combining_class_names": "Python Unicode " + unicodedata.unidata_version,
        "combining_class_semantics": "validated/ordered by the reference NFD engine",
        "table_sha256": hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest(),
    }
    return tables


def emit(tables, header, source):
    lines = ["#ifndef QWEN_UNICODE_TABLES_H", "#define QWEN_UNICODE_TABLES_H",
             "#include <stdint.h>", "",
             "typedef struct { uint32_t first, last; } QwenRange;",
             "typedef struct { uint32_t code, count, sequence[4]; } QwenDecomposition;",
             "typedef struct { uint32_t code, combining; } QwenCombining;",
             "typedef struct { uint32_t first, second, composed; } QwenComposition;", ""]
    for name in ("letter", "number", "space"):
        lines += [f"extern const QwenRange qwen_{name}_ranges[];",
                  f"extern const uint32_t qwen_{name}_count;"]
    lines += ["extern const QwenDecomposition qwen_canonical[];",
              "extern const uint32_t qwen_canonical_count;",
              "extern const QwenCombining qwen_combining[];",
              "extern const uint32_t qwen_combining_count;",
              "extern const QwenComposition qwen_composition[];",
              "extern const uint32_t qwen_composition_count;", "", "#endif", ""]
    header.write_text("\n".join(lines))
    out = ["/* Generated by tools/gen_unicode_tables.py; do not edit. */",
           '/* Reference tokenizers: ' + tables["provenance"]["tokenizers_version"] + ' */',
           '#include "unicode_tables.h"', ""]
    for name in ("letter", "number", "space"):
        rows = tables[name]
        out += [f"const QwenRange qwen_{name}_ranges[] = {{"]
        out += [f"  {{0x{a:05x}u, 0x{b:05x}u}}," for a, b in rows]
        out += ["};", f"const uint32_t qwen_{name}_count = {len(rows)}u;", ""]
    out.append("const QwenDecomposition qwen_canonical[] = {")
    for cp, sequence in tables["canonical"]:
        items = ", ".join(f"0x{x:05x}u" for x in sequence + [0] * (4 - len(sequence)))
        out.append(f"  {{0x{cp:05x}u, {len(sequence)}u, {{{items}}}}},")
    out += ["};", f"const uint32_t qwen_canonical_count = {len(tables['canonical'])}u;", ""]
    out.append("const QwenCombining qwen_combining[] = {")
    out += [f"  {{0x{cp:05x}u, {c}u}}," for cp, c in tables["ccc"]]
    out += ["};", f"const uint32_t qwen_combining_count = {len(tables['ccc'])}u;", ""]
    out.append("const QwenComposition qwen_composition[] = {")
    out += [f"  {{0x{a:05x}u, 0x{b:05x}u, 0x{c:05x}u}}," for a, b, c in tables["composition"]]
    out += ["};", f"const uint32_t qwen_composition_count = {len(tables['composition'])}u;", ""]
    source.write_text("\n".join(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tables = collect()
    emit(tables, args.header, args.source)
    report = dict(tables["provenance"])
    report.update(letter_ranges=len(tables["letter"]), number_ranges=len(tables["number"]),
                  space_ranges=len(tables["space"]), canonical_decompositions=len(tables["canonical"]),
                  combining_marks=len(tables["ccc"]), composition_pairs=len(tables["composition"]),
                  table_bytes_estimate=sum(len(tables[n]) * 8 for n in ("letter", "number", "space"))
                  + len(tables["canonical"]) * 24 + len(tables["ccc"]) * 8
                  + len(tables["composition"]) * 12)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

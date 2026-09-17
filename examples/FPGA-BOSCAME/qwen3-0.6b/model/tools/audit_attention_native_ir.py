#!/usr/bin/env python3
"""Audit the actual post-bufferization key descriptors at native-QK callsites.

This is deliberately a narrow evidence checker for Buddy's unoptimized textual
LLVM IR, not a general LLVM parser. Unknown expression forms fail closed. The
graph matcher checks tensor semantics; this check verifies the resulting key
storage is really contiguous [1, H, capacity, head_dim] before linking it to a
kernel that uses physical strides instead of dynamically reading descriptors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(text, expected_layers):
    records = []
    for function in re.finditer(r"^define [^\n]*@([^ (]+)\([^\n]*\) \{\n(.*?)^\}", text,
                                re.M | re.S):
        name, body = function[1], function[2]
        if name not in ("forward_prefill", "forward_decode"):
            continue
        definitions = dict(re.findall(r"^\s*(%[a-zA-Z0-9_.]+) = (.*)$", body, re.M))

        def evaluate(value, field=(), depth=0):
            if depth > 100 or value not in definitions:
                if not field and re.fullmatch(r"-?\d+", value):
                    return int(value)
                raise ValueError(f"cannot establish static key layout: {value}, field {field}")
            expression = definitions[value]
            inserted = re.fullmatch(r"insertvalue \{.*\} (%\w+|poison), (?:i64|ptr) ([^, ]+), ([0-9, ]+)", expression)
            if inserted:
                index = tuple(map(int, inserted[3].split(",")))
                if index == field:
                    return evaluate(inserted[2], (), depth+1)
                return evaluate(inserted[1], field, depth+1)
            extracted = re.fullmatch(r"extractvalue \{.*\} (%\w+), ([0-9, ]+)", expression)
            if extracted:
                return evaluate(extracted[1], tuple(map(int, extracted[2].split(","))) + field, depth+1)
            raise ValueError(f"unknown descriptor expression for {value}: {expression}")

        pattern = (r"call void @([^ (]*attention_qk_position_native_"
                   r"(\d+)x(\d+)x(\d+)x(\d+))\(([^\n]*)\)")
        for call in re.finditer(pattern, body):
            symbol, heads, sequence, capacity, width, args = call.groups()
            h, s, cap, d = map(int, (heads, sequence, capacity, width))
            arguments = [part.strip().split()[-1] for part in args.split(",")]
            # Four ranked descriptors: Q rank4, K rank4, Position rank1, C rank4.
            if len(arguments) != 11 + 11 + 5 + 11:
                raise ValueError("unexpected native QK flattened ABI: " + symbol)
            key = arguments[11:22]
            offset = evaluate(key[2])
            sizes = [evaluate(value) for value in key[3:7]]
            strides = [evaluate(value) for value in key[7:11]]
            if sizes != [1,h,cap,d] or strides != [h*cap*d,cap*d,d,1]:
                raise ValueError(f"native QK requires contiguous key layout: {sizes}, {strides}")
            records.append({"function": name, "symbol": symbol,
                "key_sizes": sizes, "key_strides": strides, "key_offset_elements": offset,
                "kernel_reads": "[head, capacity, head_dim] using aligned + offset",
                "query_sequence": s})
        if re.search(r"call void @[^ (]*layout_k_", body):
            raise ValueError("full-cache key transpose call remains")
    if len(records) != expected_layers:
        raise ValueError(f"expected {expected_layers} native QK callsites, found {len(records)}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.llvm.read_text()
    records = audit(text, args.layers)
    report = {"status":"PASS", "scope":"actual lowered native-QK key descriptor ABI; no numerical execution",
              "llvm":str(args.llvm), "llvm_sha256":hashlib.sha256(text.encode()).hexdigest(),
              "callsite_count":len(records), "callsites":records}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"PASS: {len(records)} contiguous native-QK key descriptors")


if __name__ == "__main__":
    main()

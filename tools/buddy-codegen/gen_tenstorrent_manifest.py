#!/usr/bin/env python3
# ===- gen_tenstorrent_manifest.py ------------------------------------------
#
# Generate a small RHAL manifest for Tenstorrent TTNN flatbuffer artifacts.
# The manifest can be packed with rax-pack and dispatched by buddy-cli when
# the llama31_tt runner is linked in.
#
# ===------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path


def _file_uri(path: str | Path) -> str:
    text = str(path)
    if text.startswith(("file:", "payload:")):
        return text
    return f"file:{text}"


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an RHAL manifest for Llama 3.1 TTNN artifacts."
    )
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--model-name", default="llama31_tt")
    parser.add_argument("--prefill-ttnn", required=True)
    parser.add_argument("--decode-ttnn", required=True)
    parser.add_argument("--artifacts", default="chat_artifacts")
    parser.add_argument("--runner", default="llama31_chat_run.py")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF tokenizer path/id passed through to llama31_chat_run.py.",
    )
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--official-reference-npz", default="")
    parser.add_argument("--official-trace-out", default="")
    parser.add_argument("--ignore-system-desc", action="store_true", default=True)
    parser.add_argument("--no-ignore-system-desc", dest="ignore_system_desc", action="store_false")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--device-token-loop", action="store_true")
    args = parser.parse_args()

    attrs = {
        "version": "0.1.0",
        "model_name": args.model_name,
        "runner_uri": _file_uri(args.runner),
        "artifacts_uri": _file_uri(args.artifacts),
        "tokenizer_uri": args.tokenizer,
        "max_cache_len": str(args.max_cache_len),
        "ignore_system_desc": "true" if args.ignore_system_desc else "false",
        "ignore_eos": "true" if args.ignore_eos else "false",
        "device_token_loop": "true" if args.device_token_loop else "false",
    }
    if args.official_reference_npz:
        attrs["official_reference_uri"] = _file_uri(args.official_reference_npz)
    if args.official_trace_out:
        attrs["official_trace_uri"] = _file_uri(args.official_trace_out)

    attr_items = ", ".join(
        f'{key} = "{_quote(value)}"' for key, value in attrs.items()
    )

    text = f"""rhal.module @llama31_tt attributes {{{attr_items}}} {{
  rhal.codeobj @prefill_ttnn {{id = 1 : i32, kind = "raw_bytes",
                              backend = "ttnn",
                              uri = "{_quote(_file_uri(args.prefill_ttnn))}"}}
  rhal.codeobj @decode_ttnn {{id = 2 : i32, kind = "raw_bytes",
                             backend = "ttnn",
                             uri = "{_quote(_file_uri(args.decode_ttnn))}"}}
}}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

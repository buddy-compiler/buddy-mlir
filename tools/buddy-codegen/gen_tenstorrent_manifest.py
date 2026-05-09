#!/usr/bin/env python3
# ===- gen_tenstorrent_manifest.py ------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===------------------------------------------------------------------------
#
# Generate an RHAL manifest for Tenstorrent TTNN flatbuffer artifacts.
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


def _artifact_constants(artifacts: str | Path) -> list[tuple[str, Path]]:
    root = Path(artifacts)
    shared_weights = root / "prefill" / "weights.npz"
    required = [
        "slot_roles.json",
        "shapes.json",
        "dtypes.json",
        "summary.json",
        "weights.npz",
    ]
    optional = ["inv_freq.npy"]
    out: list[tuple[str, Path]] = []

    for phase in ("prefill", "decode"):
        phase_dir = root / phase
        for filename in required:
            # The model weights are identical across prefill and decode. Keep
            # both phase-local manifest symbols, but point them at one source
            # file so rax-pack embeds the large weight archive only once.
            if (
                phase == "decode"
                and filename == "weights.npz"
                and shared_weights.is_file()
            ):
                path = shared_weights
            else:
                path = phase_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing Llama artifact: {path}")
            sym = f"artifact_{phase}_{filename.replace('.', '_')}"
            out.append((sym, path))
        for filename in optional:
            path = phase_dir / filename
            if path.is_file():
                sym = f"artifact_{phase}_{filename.replace('.', '_')}"
                out.append((sym, path))
    return out


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
    parser.add_argument(
        "--ignore-system-desc", action="store_true", default=True
    )
    parser.add_argument(
        "--no-ignore-system-desc",
        dest="ignore_system_desc",
        action="store_false",
    )
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
  rhal.codeobj @runner_py {{id = 3 : i32, kind = "raw_bytes",
                           backend = "python",
                           uri = "{_quote(_file_uri(args.runner))}"}}
"""

    for idx, (sym, path) in enumerate(
        _artifact_constants(args.artifacts), start=1
    ):
        text += (
            f'  rhal.constant @{sym} {{id = {idx} : i32, storage = "external",\n'
            f"                           type = tensor<1xi8>,\n"
            f'                           uri = "{_quote(_file_uri(path))}"}}\n'
        )

    text += "}\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import os
from pathlib import Path


def _file_uri(path: str | Path) -> str:
    text = str(path)
    if text.startswith(("file:", "payload:")):
        return text
    return f"file:{Path(text).resolve()}"


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _artifact_constants(artifacts: str | Path) -> list[tuple[str, Path]]:
    root = Path(artifacts)
    shared_weights = root / "prefill" / "weights.bin"
    required = [
        "slot_roles.json",
        "shapes.json",
        "dtypes.json",
        "summary.json",
        "weights.bin",
    ]
    optional = ["inv_freq.npy"]
    out: list[tuple[str, Path]] = []

    for phase in ("prefill", "decode"):
        phase_dir = root / phase
        for filename in required:
            # The model weights are identical across prefill and decode. Keep
            # both phase-local manifest symbols, but point them at one source
            # file so rax-pack embeds the large binary blob only once.
            if (
                phase == "decode"
                and filename == "weights.bin"
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


def _local_path_from_uri(path: str | Path) -> Path | None:
    text = str(path)
    if text.startswith("file:"):
        return Path(text[5:])
    if text.startswith("payload:"):
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate
    return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


def _resolve_hf_snapshot(model_id: str) -> Path | None:
    if model_id.startswith(("file:", "payload:")):
        return None
    if "/" not in model_id:
        return None
    if Path(model_id).is_absolute():
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to package tokenizer files from "
            f"remote model id: {model_id}"
        ) from exc

    local_only = _env_flag("HF_HUB_OFFLINE") or _env_flag("TRANSFORMERS_OFFLINE")
    patterns = [
        "tokenizer.json",
        "tokenizer.model",
        "original/tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    snapshot = snapshot_download(
        repo_id=model_id,
        allow_patterns=patterns,
        local_files_only=local_only,
    )
    return Path(snapshot)


def _tokenizer_constants(tokenizer: str | Path) -> list[tuple[str, Path]]:
    root = _local_path_from_uri(tokenizer)
    if root is None:
        root = _resolve_hf_snapshot(str(tokenizer))
    if root is None:
        raise FileNotFoundError(
            "tokenizer path is not local and could not be resolved from the "
            f"Hugging Face cache: {tokenizer}"
        )

    if root.is_file():
        if root.name not in ("tokenizer.json", "tokenizer.model"):
            raise FileNotFoundError(
                f"tokenizer file must be tokenizer.json or tokenizer.model: {root}"
            )
        return [(f"tokenizer_{root.name.replace('.', '_')}", root)]

    if not root.is_dir():
        raise FileNotFoundError(f"tokenizer path is not a directory: {root}")

    files = [
        ("tokenizer_tokenizer_json", root / "tokenizer.json"),
        ("tokenizer_tokenizer_model", root / "tokenizer.model"),
        ("tokenizer_tokenizer_model", root / "original" / "tokenizer.model"),
        ("tokenizer_tokenizer_config_json", root / "tokenizer_config.json"),
        ("tokenizer_special_tokens_map_json", root / "special_tokens_map.json"),
        ("tokenizer_generation_config_json", root / "generation_config.json"),
    ]
    out: list[tuple[str, Path]] = []
    seen_symbols: set[str] = set()
    for symbol, path in files:
        if symbol in seen_symbols or not path.is_file():
            continue
        seen_symbols.add(symbol)
        out.append((symbol, path))

    if not any(
        symbol in ("tokenizer_tokenizer_json", "tokenizer_tokenizer_model")
        for symbol, _ in out
    ):
        raise FileNotFoundError(
            f"missing tokenizer.json/tokenizer.model under tokenizer path: {root}"
        )
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
    parser.add_argument("--runner", default="")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Local tokenizer/model path recorded in the package manifest.",
    )
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "completion"),
        default="chat",
        help=(
            "Default prompt formatting used by the native Llama runner when "
            "--chat-template is not provided."
        ),
    )
    parser.add_argument(
        "--prefill-kv-output-order",
        choices=("key_value", "value_key"),
        default="key_value",
        help="Order of KV cache tensors returned by the prefill graph.",
    )
    parser.add_argument(
        "--disable-static-reuse",
        action="store_true",
        help="Bake BUDDY_LLAMA31_DISABLE_STATIC_REUSE into the package manifest.",
    )
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
        "artifacts_uri": _file_uri(args.artifacts),
        "tokenizer_uri": args.tokenizer,
        "max_cache_len": str(args.max_cache_len),
        "batch_size": str(args.batch_size),
        "ignore_system_desc": "true" if args.ignore_system_desc else "false",
        "ignore_eos": "true" if args.ignore_eos else "false",
        "device_token_loop": "true" if args.device_token_loop else "false",
        "prompt_format": args.prompt_format,
        "prefill_kv_output_order": args.prefill_kv_output_order,
        "disable_static_reuse": "true"
        if args.disable_static_reuse
        else "false",
    }
    if args.official_reference_npz:
        attrs["official_reference_uri"] = _file_uri(args.official_reference_npz)
    if args.official_trace_out:
        attrs["official_trace_uri"] = _file_uri(args.official_trace_out)

    attr_items = ", ".join(
        f'{key} = "{_quote(value)}"' for key, value in attrs.items()
    )

    module_symbol = args.model_name.replace("-", "_").replace(".", "_")
    text = f"""rhal.module @{module_symbol} attributes {{{attr_items}}} {{
  rhal.codeobj @prefill_ttnn {{id = 1 : i32, kind = "raw_bytes",
                              backend = "ttnn",
                              uri = "{_quote(_file_uri(args.prefill_ttnn))}"}}
  rhal.codeobj @decode_ttnn {{id = 2 : i32, kind = "raw_bytes",
                             backend = "ttnn",
                             uri = "{_quote(_file_uri(args.decode_ttnn))}"}}
"""

    constants = _artifact_constants(args.artifacts)
    constants.extend(_tokenizer_constants(args.tokenizer))

    for idx, (sym, path) in enumerate(constants, start=1):
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

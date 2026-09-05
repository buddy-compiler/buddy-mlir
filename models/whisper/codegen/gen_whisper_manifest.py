#!/usr/bin/env python3
# ===- gen_whisper_manifest.py - RHAL .mlir manifest for Whisper -----------===//
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
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys


def _normalize_uri(raw: str) -> str:
    value = raw.strip()
    if ":" in value:
        return value
    return f"file:{value}"


def gen_manifest(
    spec: dict, runner_library: str, transcription_library: str
) -> str:
    model_id = spec.get("model_id", f"{spec['model_family']}_{spec['variant']}")
    params_size = spec["params_size"]
    vocab_size = spec["vocab_size"]
    max_token_len = spec["max_token_len"]
    mel_bins = spec["mel_bins"]
    audio_frames = spec["audio_frames"]
    so_name = spec.get("so_name", "whisper_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    vocab_file = spec.get("vocab_file", "vocab.txt")

    lines = []
    emit = lines.append
    emit("rhal.module @whisper attributes {")
    emit('    version = "0.1.0",')
    emit(f'    model_name = "{model_id}",')
    emit(f'    vocab_uri = "file:{vocab_file}",')
    emit(f'    runner_library = "{runner_library}",')
    emit(f'    transcription_library = "{transcription_library}",')
    emit(f'    params_size = "{params_size}",')
    emit(f'    vocab_size = "{vocab_size}",')
    emit(f'    max_token_len = "{max_token_len}",')
    emit(f'    mel_bins = "{mel_bins}",')
    emit(f'    audio_frames = "{audio_frames}",')
    emit(f'    enc_seq = "{spec["enc_seq"]}",')
    emit(f'    enc_dim = "{spec["enc_dim"]}",')
    emit(f'    sot_token = "{spec["sot_token"]}",')
    emit(f'    eot_token = "{spec["eot_token"]}"')
    emit("} {")
    emit("")

    emit('  rhal.constant @params {id = 1 : i32, storage = "external",')
    emit(f"                         type = tensor<{params_size}xf32>,")
    emit(f'                         uri = "file:{weight_file}"}}')
    emit("")

    emit(
        '  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",'
    )
    emit('                                backend = "cpu",')
    emit(f'                                uri = "file:{so_name}"}}')
    emit("")

    emit(
        f'  rhal.buffer @audio_features {{space = "host", '
        f"type = tensor<1x{mel_bins}x{audio_frames}xf32>}}"
    )
    emit(
        f'  rhal.buffer @decoder_tokens {{space = "host", '
        f"type = tensor<1x{max_token_len}xi64>}}"
    )
    emit(
        f'  rhal.buffer @logits {{space = "host", '
        f"type = tensor<1x{max_token_len}x{vocab_size}xf32>}}"
    )
    emit("")

    emit("  rhal.func @forward {")
    emit('    inputs   = ["audio_features", "decoder_tokens"],')
    emit('    outputs  = ["logits"],')
    emit('    dispatch = "model_kernels",')
    emit('    args     = ["audio_features", "decoder_tokens", "logits"]}')
    emit("}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate RHAL .mlir manifest for the Whisper model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="whisper_runner.so",
        help="Runner plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "--transcription-library",
        default="whisper_transcription.so",
        help="Audio transcription plugin URI/name for module attrs.",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    text = gen_manifest(
        spec,
        _normalize_uri(args.runner_library),
        _normalize_uri(args.transcription_library),
    )

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(text)
        print(f"[gen_whisper_manifest] Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

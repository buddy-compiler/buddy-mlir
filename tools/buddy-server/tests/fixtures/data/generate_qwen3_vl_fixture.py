#!/usr/bin/env python3
"""Generate small resources for the production Qwen3-VL runtime tests.

The kernel library implements the actual dlopen ABI, with deterministic outputs.
No model weights, SDK, network access, or Python numerical packages are needed.
"""

import argparse
import json
from array import array
from pathlib import Path

CONTEXT = 512
HIDDEN = 2048
IMAGE_TOKENS = 98
LOGITS_VOCAB = 264
SPECIALS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
)


def byte_alphabet():
    extra = 256
    result = []
    for byte in range(256):
        if 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255:
            result.append(chr(byte))
        else:
            result.append(chr(extra))
            extra += 1
    return result


def write_array(path, code, values):
    with path.open("wb") as output:
        array(code, values).tofile(output)


def quote(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    vocabulary = byte_alphabet() + list(SPECIALS)
    (target / "vocab.txt").write_text(
        "\n".join(vocabulary) + "\n", encoding="utf-8"
    )

    # Every embedding row exposes its ID to the test decoder. Image rows must
    # instead contain the pooled vision output, including at the last row.
    write_array(
        target / "embed_table.bin",
        "f",
        (float(token) for token in range(LOGITS_VOCAB) for _ in range(HIDDEN)),
    )
    write_array(target / "weights.bin", "f", [0.25])
    write_array(target / "cos.bin", "f", [1.0] * (CONTEXT * 128))
    write_array(target / "sin.bin", "f", [0.0] * (CONTEXT * 128))
    write_array(target / "cmask.bin", "f", [-42.0] * (CONTEXT * CONTEXT))
    write_array(target / "pixel_values.bin", "f", [0.5] * (392 * 1536))

    # Byte tokens have their own numeric byte IDs; special spellings are
    # recognized as one token. This matches the resident's default template.
    prefix = [257, *b"user\n", 259]
    prompt = (
        prefix
        + [261] * IMAGE_TOKENS
        + [260, *b"Hi", 258, 10, 257, *b"assistant\n"]
    )
    image_offset = len(prefix)
    (target / "prompt.txt").write_text(
        "<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * IMAGE_TOKENS
        + "<|vision_end|>Hi<|im_end|>\n<|im_start|>assistant\n",
        encoding="utf-8",
    )
    (target / "prompt.ids").write_text(" ".join(map(str, prompt)) + "\n")
    (target / "custom-template.json").write_text(
        '{"roles":{"system":"S:","user":"U:","assistant":"A:"},'
        '"turn_suffix":"<|im_end|>","add_bos":false,'
        '"add_generation_prompt":true}\n'
    )
    (target / "canonical-template.json").write_text(
        json.dumps(
            {
                "roles": {
                    role: f"<|im_start|>{role}\n"
                    for role in ("system", "user", "assistant")
                },
                "role_suffix": "<|im_end|>",
                "turn_suffix": "\n",
                "add_bos": False,
                "add_generation_prompt": True,
            }
        )
        + "\n"
    )
    # An actual tiny local image exercises preprocessing as well as the bundled
    # tensor path. The kernel checks its dimensions but accepts arbitrary pixels.
    (target / "image.ppm").write_bytes(b"P6\n1 1\n255\n\x20\x40\x60")

    resources = {
        "vision_weights": "weights.bin",
        "decoder_weights": "weights.bin",
        "embed_table": "embed_table.bin",
        "cos": "cos.bin",
        "sin": "sin.bin",
        "cmask": "cmask.bin",
        "pixel_values": "pixel_values.bin",
    }
    for profile, prompt_length, position in (
        ("canonical", len(prompt), image_offset),
        ("shifted", len(prompt), image_offset + 1),
        ("length", len(prompt) + 1, image_offset),
        ("full", CONTEXT, image_offset),
        ("near", CONTEXT - 1, image_offset),
    ):
        meta = f"{profile}-meta.txt"
        positions = f"{profile}-img_pos.bin"
        (target / meta).write_text(
            f"{prompt_length} {CONTEXT} {IMAGE_TOKENS} {HIDDEN} {LOGITS_VOCAB}\n"
        )
        write_array(
            target / positions, "q", range(position, position + IMAGE_TOKENS)
        )
        constants = dict(resources, img_pos=positions, meta=meta)
        manifest = (
            'rhal.module @qwen3_vl_test attributes {version = "0.1.0", '
            'model_name = "qwen-fixture", vocab_uri = "file:vocab.txt"} {\n'
        )
        for index, name in enumerate(("vision_kernels", "decoder_kernels"), 1):
            manifest += (
                f"  rhal.codeobj @{name} {{id = {index} : i32, "
                'kind = "host_shared_lib", backend = "cpu", '
                f'uri = "file:{quote(args.kernels.resolve())}"}}\n'
            )
        for index, (name, filename) in enumerate(constants.items(), 1):
            manifest += (
                f"  rhal.constant @{name} {{id = {index} : i32, "
                f'storage = "external", type = tensor<1xi8>, uri = "file:{filename}"}}\n'
            )
        (target / f"{profile}.mlir").write_text(manifest + "}\n")


if __name__ == "__main__":
    main()

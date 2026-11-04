#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.

"""Compare the compiled MMS vocoder with real VITS latents and PyTorch audio."""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


def memref_type(rank):
    class MemRef(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.POINTER(ctypes.c_float)),
            ("aligned", ctypes.POINTER(ctypes.c_float)),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRef


Params = memref_type(1)
Latents = memref_type(3)
Waveform = memref_type(2)


def descriptor(array, kind):
    pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return kind(
        pointer,
        pointer,
        0,
        (ctypes.c_int64 * array.ndim)(*array.shape),
        (ctypes.c_int64 * array.ndim)(
            *(stride // array.itemsize for stride in array.strides)
        ),
    )


class CompiledVocoder:
    def __init__(self, model_dir, spec):
        self.library = ctypes.CDLL(str(model_dir / spec["so_name"]))
        self.forward = self.library._mlir_ciface_forward
        self.forward.argtypes = [
            ctypes.POINTER(Waveform),
            ctypes.POINTER(Params),
            ctypes.POINTER(Latents),
        ]
        self.forward.restype = None
        self.weights = np.fromfile(
            model_dir / spec["weight_file"], dtype=np.float32
        )
        if self.weights.size != spec["params_size"]:
            raise ValueError("weight file does not match the spec")
        self.params = descriptor(self.weights, Params)
        self.samples = spec["max_seq_len"] * int(
            np.prod(spec["upsample_rates"])
        )
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None

    def __call__(self, latents):
        inputs = descriptor(latents, Latents)
        outputs = Waveform()
        self.forward(
            ctypes.byref(outputs),
            ctypes.byref(self.params),
            ctypes.byref(inputs),
        )
        try:
            if tuple(outputs.sizes) != (1, self.samples):
                raise AssertionError(
                    f"unexpected waveform shape: {tuple(outputs.sizes)}"
                )
            if (
                not outputs.aligned
                or outputs.offset < 0
                or outputs.strides[1] != 1
            ):
                raise AssertionError("invalid waveform memref")
            data = np.ctypeslib.as_array(
                outputs.aligned, shape=(outputs.offset + self.samples,)
            )
            return data[outputs.offset :].copy()
        finally:
            self.libc.free(outputs.allocated)


def cli_run(cli, package, latent_file, env):
    return subprocess.run(
        [
            str(cli),
            "--model",
            str(package),
            "--prompt",
            str(latent_file),
            "--no-stats",
        ],
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=180,
        env=env,
        cwd=package.parent,
    )


def check_invalid_manifests(cli, model_dir, output, latent_file, env, spec):
    manifest = (model_dir / "generated/mms.mlir").read_text()
    # Keep the real compiled kernel and weights; change only their metadata.
    manifest = manifest.replace('"file:', f'"file:{model_dir}/')
    cases = {
        "missing_flow_size": ("flow_size", None),
        "empty_flow_size": ("flow_size", ""),
        "negative_flow_size": ("flow_size", "-1"),
        "zero_params_size": ("params_size", "0"),
        "trailing_characters": ("max_seq_len", f"{spec['max_seq_len']}x"),
        "tensor_size_overflow": ("flow_size", str(np.iinfo(np.intp).max)),
        "inconsistent_waveform": (
            "audio_buffer_size",
            str(spec["max_seq_len"] * int(np.prod(spec["upsample_rates"])) + 1),
        ),
    }
    for label, (key, value) in cases.items():
        lines = manifest.splitlines(keepends=True)
        matching = [
            i
            for i, line in enumerate(lines)
            if line.strip().startswith(key + " =")
        ]
        if len(matching) != 1:
            raise AssertionError(
                f"expected one {key} attribute in the built manifest"
            )
        lines[matching[0]] = (
            "" if value is None else f'    {key} = "{value}",\n'
        )
        source = output / f"invalid_{label}.mlir"
        source.write_text("".join(lines))
        package = source.with_suffix(".rax")
        subprocess.run(
            [str(cli.with_name("rax-pack")), str(source), "-o", str(package)],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        result = cli_run(cli, package, latent_file, env)
        (output / f"invalid_{label}.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode != 1 or "MMSRunner:" not in result.stderr:
            raise AssertionError(
                f"invalid {label} was not rejected cleanly: {result}"
            )
    return list(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path(__file__).parent / "specs/f32.json"
    )
    parser.add_argument("--local-model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--buddy-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--text", action="append", help="Repeat for additional English texts"
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.set_num_threads(4)
    spec = json.loads(args.spec.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    compiled = CompiledVocoder(model_dir, spec)
    reference = VitsModel.from_pretrained(
        args.local_model, local_files_only=True, dtype=torch.float32
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_model, local_files_only=True
    )
    corpus = json.loads(
        (Path(__file__).parent / "validation_texts.json").read_text()
    )
    samples = (
        [
            {"id": f"custom-{i}", "text": text}
            for i, text in enumerate(args.text)
        ]
        if args.text
        else corpus["samples"]
    )
    frames = int(spec["max_seq_len"])
    rate = int(reference.config.sampling_rate)
    package_dir = output / "relocated"
    package_dir.mkdir(exist_ok=True)
    package = package_dir / "mms.rax"
    shutil.copy2(model_dir / "mms.rax", package)
    env = dict(os.environ, BUDDY_RAX_PAYLOAD_DIR=str(output / "payload"))
    env.pop("LD_LIBRARY_PATH", None)

    captured = []

    def capture_latents(_module, inputs):
        captured.append(inputs[0].detach().clone())

    hook = reference.decoder.register_forward_pre_hook(capture_latents)
    cases = []
    last_latents = None
    try:
        with torch.no_grad():
            for text_index, sample in enumerate(samples):
                text = sample["text"]
                captured.clear()
                torch.manual_seed(text_index)
                synthesis = reference(**tokenizer(text, return_tensors="pt"))
                if (
                    not torch.isfinite(synthesis.waveform).all()
                    or len(captured) != 1
                ):
                    raise AssertionError(
                        "VITS did not produce a finite waveform and one latent tensor"
                    )
                full_latents = captured.pop()
                if full_latents.shape[-1] < frames:
                    raise ValueError(
                        "text is too short for the requested latent window"
                    )
                last_start = full_latents.shape[-1] - frames
                for start in sorted({0, last_start // 2, last_start}):
                    name = f"text{text_index}_frame{start}"
                    window = full_latents[
                        :, :, start : start + frames
                    ].contiguous()
                    expected = reference.decoder(window, None).squeeze().numpy()
                    actual_latents = window.numpy().copy()
                    actual = compiled(actual_latents)
                    if not np.isfinite(actual).all():
                        raise AssertionError(
                            f"{name}: non-finite compiled waveform"
                        )
                    np.testing.assert_allclose(
                        actual, expected, atol=args.atol, rtol=args.rtol
                    )
                    latent_file = output / f"{name}.f32"
                    actual_latents.tofile(latent_file)
                    last_latents = actual_latents
                    result = cli_run(
                        args.buddy_cli.resolve(), package, latent_file, env
                    )
                    (output / f"{name}.cli.log").write_text(
                        result.stdout + result.stderr
                    )
                    if result.returncode:
                        raise RuntimeError(
                            f"{name}: buddy-cli failed: {result.stderr}"
                        )
                    summary = json.loads(result.stdout)
                    assert summary["n_samples"] == actual.size
                    np.testing.assert_allclose(
                        summary["first"], actual[:8], atol=1e-6, rtol=1e-5
                    )
                    for key, value in (
                        ("min", actual.min()),
                        ("max", actual.max()),
                        ("mean", actual.astype(np.float64).mean()),
                        (
                            "rms",
                            np.sqrt(np.mean(actual.astype(np.float64) ** 2)),
                        ),
                    ):
                        np.testing.assert_allclose(
                            summary[key], value, atol=1e-6, rtol=1e-5
                        )
                    error = actual.astype(np.float64) - expected
                    cases.append(
                        {
                            "name": name,
                            "transcript_id": sample["id"],
                            "text": text,
                            "samples": actual.size,
                            "max_abs_error": float(np.max(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                        }
                    )
                    sf.write(
                        output / f"{name}.reference.wav",
                        expected,
                        rate,
                        subtype="FLOAT",
                    )
                    sf.write(
                        output / f"{name}.compiled.wav",
                        actual,
                        rate,
                        subtype="FLOAT",
                    )
                    print(json.dumps(cases[-1]), flush=True)
    finally:
        hook.remove()

    # Invalid inputs are derived from the real model's latents.
    invalid_cases = {
        "truncated": last_latents.ravel()[:-1],
        "extra_sample": np.append(
            last_latents.ravel(), last_latents.ravel()[-1]
        ),
    }
    for label, value in (("nan", np.nan), ("infinity", np.inf)):
        invalid = last_latents.copy()
        invalid.flat[0] = value
        invalid_cases[label] = invalid
    for label, invalid in invalid_cases.items():
        path = output / f"invalid_{label}.f32"
        invalid.astype(np.float32).tofile(path)
        result = cli_run(args.buddy_cli.resolve(), package, path, env)
        if result.returncode != 1 or "MMSRunner:" not in result.stderr:
            raise AssertionError(
                f"invalid {label} was not rejected cleanly: {result}"
            )

    bad_weights = output / "truncated-weights.data"
    shutil.copyfile(model_dir / spec["weight_file"], bad_weights)
    with bad_weights.open("r+b") as stream:
        stream.truncate(
            bad_weights.stat().st_size - np.dtype(np.float32).itemsize
        )
    bad_weight_result = subprocess.run(
        [
            str(args.buddy_cli.resolve()),
            "--runner-so",
            str(model_dir / "mms_runner.so"),
            "--model-so",
            str(model_dir / spec["so_name"]),
            "--weights",
            str(bad_weights),
            "--prompt",
            str(latent_file),
            "--no-stats",
        ],
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
    )
    if (
        bad_weight_result.returncode != 1
        or "weight file size does not match params_size"
        not in bad_weight_result.stderr
    ):
        raise AssertionError(
            f"truncated weights were not rejected: {bad_weight_result}"
        )

    invalid_manifests = check_invalid_manifests(
        args.buddy_cli.resolve(), model_dir, output, latent_file, env, spec
    )

    report = {
        "model_id": spec["model_id"],
        "model_revision": spec.get("model_revision"),
        "dataset": "custom" if args.text else corpus["dataset"],
        "dataset_split": None if args.text else corpus["split"],
        "max_seq_len": frames,
        "sampling_rate": rate,
        "torch": torch.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": cases,
        "invalid_inputs_rejected": list(invalid_cases),
        "truncated_weights_rejected": True,
        "invalid_manifests_rejected": invalid_manifests,
        "relocated_package": "passed",
    }
    (output / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: {len(cases)} waveform comparisons, relocated CLI package, "
        f"{len(invalid_cases)} invalid inputs, truncated weights and "
        f"{len(invalid_manifests)} invalid manifests"
    )


if __name__ == "__main__":
    main()

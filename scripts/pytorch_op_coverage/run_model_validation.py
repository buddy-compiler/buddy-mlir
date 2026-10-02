#!/usr/bin/env python3
"""Isolated model checks, independent of the fixed operator denominator."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from run_coverage import provenance, worker_environment
from worker import environment, write_result

HERE = Path(__file__).resolve().parent
BERT_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
CASES = (
    "bert-tiny",
    "mixtral-block",
    "mixtral-block-nonstrict",
    "mixtral-decoder",
    "mixtral-decoder-mask",
    "olmoe-block",
)


def lifted_inputs(exported, args):
    """Follow the export signature, including non-persistent buffers."""
    user = iter(args)
    values = []
    for spec in exported.graph_signature.input_specs:
        kind = spec.kind.name
        if kind == "USER_INPUT":
            values.append(next(user))
        elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
            owner = (
                exported.state_dict
                if spec.target in exported.state_dict
                else exported.constants
            )
            values.append(owner[spec.target])
        else:
            raise ValueError(f"Unsupported exported input kind: {kind}")
    if list(user):
        raise ValueError("Unused model inputs")
    return values


def build_case(name, metadata):
    import torch
    from transformers import (
        BertModel,
        BertTokenizer,
        MixtralConfig,
        MixtralForCausalLM,
    )
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralSparseMoeBlock,
    )

    if name == "bert-tiny":
        from huggingface_hub import snapshot_download

        metadata.update(model="prajjwal1/bert-tiny", revision=BERT_REVISION)
        folder = Path(
            snapshot_download(
                metadata["model"],
                revision=BERT_REVISION,
                allow_patterns=[
                    "config.json",
                    "pytorch_model.bin",
                    "vocab.txt",
                ],
            )
        )
        metadata["artifacts_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir())
            if p.is_file()
        }
        model = BertModel.from_pretrained(
            folder, local_files_only=True, attn_implementation="eager"
        ).eval()
        tokenizer = BertTokenizer.from_pretrained(folder, local_files_only=True)

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, ids, mask, types):
                return self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=types,
                    return_dict=False,
                )

        batches = []
        for texts in (
            [
                "The compiler executes the model.",
                "Sparse experts route tokens.",
            ],
            ["A different input checks compilation reuse.", "Hello."],
        ):
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=16,
                return_tensors="pt",
            )
            batches.append(
                tuple(
                    encoded[k]
                    for k in ("input_ids", "attention_mask", "token_type_ids")
                )
            )
        metadata["scope"] = (
            "Pretrained encoder, f32, batch=2, length=16; hidden states and pooler"
        )
        implementation = BertModel
        module = Encoder().eval()
    elif name == "olmoe-block":
        from transformers import OlmoeConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        config = OlmoeConfig(
            hidden_size=16,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        metadata["config"] = config.to_dict()
        metadata["scope"] = (
            "Random-weight standard OLMoE block, four experts/top-2; not pretrained validation"
        )
        implementation = OlmoeSparseMoeBlock
        module = OlmoeSparseMoeBlock(config).eval()
        batches = [(torch.randn(2, 4, 16),) for _ in range(2)]
        batches.extend(
            (torch.full((2, 4, 16), value),) for value in (0.0, 1.0, -1.0)
        )
        with torch.no_grad():
            metadata["expert_token_counts"] = [
                torch.bincount(
                    module.gate(args[0].reshape(-1, 16))
                    .topk(2, dim=-1)
                    .indices.flatten(),
                    minlength=4,
                ).tolist()
                for args in batches
            ]
        if not any(0 in counts for counts in metadata["expert_token_counts"]):
            raise AssertionError("OLMoE validation must include unused experts")
    else:
        config = MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            router_jitter_noise=0.0,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        metadata["config"] = config.to_dict()
        metadata["scope"] = (
            "Random-weight standard Mixtral, four experts/top-2; not pretrained validation"
        )
        implementation = MixtralSparseMoeBlock
        if name.startswith("mixtral-block"):
            module = MixtralSparseMoeBlock(config).eval()
            batches = [(torch.randn(2, 4, 16),) for _ in range(2)]
        else:

            class Decoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = MixtralForCausalLM(config).eval()

                def forward(self, tokens, mask=None):
                    return self.model(
                        tokens,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=False,
                    )[0]

            module = Decoder().eval()
            batches = [(torch.randint(0, 32, (2, 4)),) for _ in range(2)]
            if name.endswith("-mask"):
                mask = torch.zeros(2, 1, 4, 4).masked_fill(
                    torch.ones(4, 4, dtype=torch.bool).triu(1),
                    torch.finfo(torch.float32).min,
                )
                batches = [(*args, mask) for args in batches]
    source = Path(inspect.getfile(implementation))
    metadata["model_source_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    return module, batches


def run_model(name, repo, path):
    result = {
        "name": name,
        "status": "running",
        "stage": "environment",
        "seed": 0,
    }

    def stage(value):
        result["stage"] = value
        write_result(path, result)

    try:
        stage("environment")
        import torch
        import transformers
        from buddy.compiler.frontend import DynamoCompiler
        from buddy.compiler.ops import tosa

        result["environment"] = environment([])
        result["transformers"] = transformers.__version__
        installed = result["environment"]["buddy_source_sha256"]
        source = repo / "frontend/Python"
        for p in source.rglob("*.py"):
            digest = hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if digest != installed.get(p.relative_to(source).as_posix()):
                raise ValueError(
                    f"Loaded Buddy source differs: {p.relative_to(repo)}"
                )
        torch.manual_seed(0)
        torch.set_num_threads(1)
        stage("load_model")
        model, batches = build_case(name, result)
        result["inputs"] = [
            [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in args]
            for args in batches
        ]
        with torch.no_grad():
            stage("eager")
            expected = [model(*args) for args in batches]
            expected = [x if isinstance(x, tuple) else (x,) for x in expected]
            if not all(
                torch.isfinite(t).all() for outputs in expected for t in outputs
            ):
                raise ValueError("Nonfinite eager reference")
            stage("export")
            strict = name != "mixtral-block-nonstrict"
            result["strict_export"] = strict
            exported = torch.export.export(model, batches[0], strict=strict)
            result["observed_ops"] = dict(
                Counter(
                    str(n.target)
                    for n in exported.graph.nodes
                    if n.op == "call_function"
                )
            )
            stage("import")
            compiler = DynamoCompiler(
                primary_registry=tosa.ops_registry, enable_external_calls=False
            )
            compiler._compile_fx(
                exported.graph_module,
                lifted_inputs(exported, batches[0]),
                tracing_inputs=[
                    n.meta["val"]
                    for n in exported.graph.nodes
                    if n.op == "placeholder"
                ],
            )
            if len(compiler.imported_graphs) != 1:
                raise ValueError("Expected one Buddy graph")
            stage("lower")
            compiler.imported_graphs[0].lower_to_top_level_ir()
            stage("compile")
            execute = compiler.dynamo_run()
            stage("execute_and_compare")
            errors = []
            for args, reference in zip(batches, expected):
                output = execute(*lifted_inputs(exported, args))
                if len(output) != len(reference):
                    raise ValueError("Output arity mismatch")
                for actual, target in zip(output, reference):
                    torch.testing.assert_close(
                        actual, target, rtol=1e-4, atol=1e-5, equal_nan=False
                    )
                errors.append(
                    [
                        (a - b).abs().max().item()
                        for a, b in zip(output, reference)
                    ]
                )
            result.update(
                status="passed",
                stage="complete",
                input_sets=len(batches),
                max_abs_errors=errors,
            )
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    write_result(path, result)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CASES, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.out_dir = args.out_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.model}.json"
    if args.worker:
        return run_model(args.model, args.repo_root, path)
    before = provenance(args.repo_root, HERE / "data/target_ops_v1.json")
    write_result(
        path, {"name": args.model, "status": "running", "stage": "starting"}
    )
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        args.model,
        "--out-dir",
        str(args.out_dir),
        "--repo-root",
        str(args.repo_root),
    ]
    with (args.out_dir / f"{args.model}.log").open(
        "w", encoding="utf-8"
    ) as log:
        try:
            completed = subprocess.run(
                command,
                env=worker_environment(args.repo_root),
                stdout=log,
                stderr=log,
                timeout=args.timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    result = json.loads(path.read_text(encoding="utf-8"))
    if returncode is None:
        result.update(status="timeout", reason="Model worker timeout")
    elif result.get("status") == "running" or (
        returncode != 0 and result.get("status") == "passed"
    ):
        result.update(
            status="failed", reason=f"Model worker exited {returncode}"
        )
    after = provenance(args.repo_root, HERE / "data/target_ops_v1.json")
    before["changed_during_run"] = (
        before["source_sha256"] != after["source_sha256"]
    )
    if before["changed_during_run"]:
        result.update(
            status="failed", reason="Source changed during model validation"
        )
    result.update(
        provenance=before,
        worker_returncode=returncode,
        coverage_credit=False,
        rtol=1e-4,
        atol=1e-5,
    )
    write_result(path, result)
    print(f"{args.model}: {result['status']} ({result['stage']}); {path}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

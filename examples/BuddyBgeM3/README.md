# Buddy Compiler BGE-M3 Example (SpacemiT K3 Benchmark & RVV Study)

This example contains the tooling for the BGE-M3 RAX performance study on
SpacemiT K3 (issue #888): artifact export on x86, native build / benchmark /
profile / experiment scripts on K3, and the full study report.

Summary of the study: on the same K3 hardware, RAX beats the HF reference for
seq 128/256 (0.81x / 0.75x latency) and matches it for seq 512 (1.05x). Eight
RVV optimization experiments were all negative; root causes are located and
documented (see the report).

## Directory Layout

```text
examples/BuddyBgeM3/
├── README.md                       # This file
├── docs/
│   ├── K3-benchmark-report.md      # Full study report (K3 data in 6.x)
│   ├── K3-workflow.md              # End-to-end workflow
│   └── K3-no-sudo-build-guide.md   # Building buddy-mlir on K3 without sudo
├── export/                         # x86 side (one-time, needs torch)
│   ├── export_artifacts.sh         # Generate MLIR + weights + manifest
│   ├── download_bge_m3.py          # Download BAAI/bge-m3 snapshot
│   ├── gen_specs.py                # Generate seq{128,256,512} specs
│   └── hf_bench.py                 # HF reference latency benchmark
└── k3/                             # K3 side (pure shell, no python packages)
    ├── env.sh                      # Environment (source once)
    ├── build_runtime.sh            # Runtime plugins (once, all seq)
    ├── build_seq.sh                # Compile one seq variant (optimization
    │                               # knob: SUB_PASSES)
    ├── build_seq_optimize.sh       # Experiment #6: fixed-16 vectorization
    ├── bench.sh                    # Benchmark (mode A cold + mode B steady)
    ├── profile.sh                  # RVV instruction census + VLEN
    ├── thread_scaling.sh           # OMP 16/8/4/1 scaling experiment
    ├── hf_ref.sh                   # Same-hardware HF reference on K3
    ├── exp_g2.sh                   # Experiment #8: +zvl256b backend
    └── cos.py                      # Cosine similarity gate (pure stdlib)
```

## Requirements

- x86 side (export only): python3 with `torch` and `transformers`
  (`pip install -r requirements.txt` from the repo root).
- K3 side: a native buddy-mlir build with LLVM (RuyiAI fork, riscv64 target)
  and `rax-pack`; no python packages are required. See
  `docs/K3-no-sudo-build-guide.md` for the user-mode build.

## 1. Export artifacts (x86, one-time)

```bash
$ cd buddy-mlir
$ export PY=/path/to/python-with-torch
$ export LOCAL_BGE_M3=/path/to/bge-m3-hf-snapshot
$ bash examples/BuddyBgeM3/export/export_artifacts.sh  # 128 256 512
```

This produces `examples/BuddyBgeM3/dist/` with `arg0.data` (2.27 GB),
`tokenizer.json` and `seq{L}/{forward.mlir, subgraph0.mlir, generated/...}`.

## 2. Transfer to K3

```bash
$ scp -r examples/BuddyBgeM3/dist \
    user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/src
```

## 3. Build the runtime (K3, once)

```bash
$ source ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/env.sh
$ bash $APP/k3/build_runtime.sh
```

## 4. Compile a seq variant

```bash
$ bash $APP/k3/build_seq.sh 128            # baseline
$ bash $APP/k3/build_seq.sh 128 exp1       # experiment (edit SUB_PASSES)
```

## 5. Benchmark

```bash
$ bash $APP/k3/bench.sh 128 baseline 20 10
```

## 6. Profile and experiments

```bash
$ bash $APP/k3/profile.sh 128 baseline     # RVV census + VLEN
$ nohup bash $APP/k3/thread_scaling.sh > $RESULTS/thread_scaling.txt 2>&1 &
$ bash $APP/k3/hf_ref.sh                   # same-hardware HF reference
$ bash $APP/k3/exp_g2.sh 128               # +zvl256b backend experiment
```

## Recording Rules

1. Every optimization result must pass the cosine gate first
   (`k3/cos.py` vs the baseline embedding, > 0.999) before recording speed.
2. Tag every number with `precision / baseline|optimized`.
3. Never compare across precisions.
4. Raw data goes to `results/`, RVV statistics to `profile/`; summaries go
   into `docs/K3-benchmark-report.md`.
5. Negative results are recorded too: they are part of the deliverable.

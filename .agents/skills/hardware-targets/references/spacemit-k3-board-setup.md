# SpacemiT K3: running and measuring on a board

## Running

- Buddy models: the process must be an AI process before its threads start,
  so that they run on the A100 cores:
  ```bash
  sh -c 'echo 0 > /proc/set_ai_thread && exec buddy-cli --model m.rax --prompt "..."'
  ```
  The same wrapper is needed for microbenchmarks of A100 kernels.
- llama.cpp (llama.cpp-tools-spacemit): run it *without* the AI switch; it
  places its threads itself and is ~2x slower as an AI process. Its
  libraries need `libspert.so.1` (spacemit-runtime) and `libspine_tcm`
  (spacemit-tcm) on `LD_LIBRARY_PATH`.
- `/tmp` is a RAM tmpfs (16 GB on the board measured): a 2.5 GB `.rax` there
  takes memory; delete old ones.
- `sudo` needs a password on the boards: never print it or store it in a
  repository; pass it through `sudo -S` from a file the developer chooses.

## Measuring

- `perf` is available; profile with `perf record -F 4000 -g` inside the AI
  wrapper (`... && exec perf record ... buddy-cli ...`).
- Alternate baseline and candidate runs (A B A B); single runs vary by
  ~1-2% in decode speed.
- Cross-compile microbenchmarks on the x86 host with the RISC-V GNU
  toolchain sysroot and clang (`--target=riscv64-unknown-linux-gnu
  -march=rv64gcv_zfh_zvfh`), copy them to the board, run them in the AI
  wrapper.
- Long jobs over ssh: start them detached (`setsid nohup ... &`) and poll
  for the result instead of keeping the connection open.

# Zhihe A210: cores and memory

Measured in 2026-10 on an A210 development board (Debian 13, vendor Linux
6.6 kernel, 7.6 GiB LPDDR4X), user space only.

## Cores

| CPUs | Core | L1 D / L2 | Max clock |
| --- | --- | --- | --- |
| 0-3 | XuanTie C908 (in-order) | 32 KiB / 512 KiB per core | 1.896 GHz |
| 4-7 | XuanTie C920 (out-of-order) | 64 KiB / 1 MiB per core | 1.896 GHz |

- All 8 cores: rv64gcv, RVV 1.0 with VLEN 128, Zba / Zbb / Zbs. Zfh and Zvfh
  work on both kinds (scalar and vector f16 instructions execute) although
  `/proc/cpuinfo` does not list them.
- `cpufreq` has one policy per cluster (`schedutil`); a loaded cluster runs
  at 1.896 GHz.
- LLVM has a scheduling model for `xt-c920v2` but none for the C908.

## Memory

Read bandwidth (each thread streams its own 256 MiB, RVV or scalar loads
alike; no prefetch hint helps):

| Threads | GB/s |
| --- | --- |
| 1 x C908 | 7.3-8.0 |
| 4 x C908 | 7.5-8.2 |
| 1 x C920 | 3.0 |
| 4 x C920 | 2.9-3.1 |
| all 8 | 11.0 |

The C920 cluster reaches far less DRAM bandwidth than the C908 cluster. A
DeepSeek R1 1.5B int4 decode kernel streamed 12-13 GB/s on all 8 cores.
There is no devfreq knob for the DRAM.

## Vector throughput

`vwmacc.vx` (e8 -> e16), independent accumulators, one core:

| Core | LMUL 1 | LMUL 2 |
| --- | --- | --- |
| C920 | 27.9 GMAC/s (one instruction per cycle) | 26.9 GMAC/s |
| C908 | 7.4 GMAC/s | 7.6 GMAC/s |

## Running

- The thread pool of the model libraries (`runtime/threadpool/
  BuddyThreadPool.c`) pins thread i to the i-th allowed CPU: with all 8
  allowed, threads 0-3 run on the C908 cores and 4-7 on the C920 cores.
- `perf` works for user-space counting and sampling of one's own processes
  (`perf_event_paranoid` 2).
- An 8-thread kernel waits for its slowest thread: a busy core (any other
  process) stretches single calls by 2-3x; time many calls and take the
  median.
- NPU: a VeriSilicon VIP of the A210 is reachable only through VIPLite
  (`/dev/vipcore0`), which runs precompiled network binaries (NBG); no
  public tool generates them for this chip.

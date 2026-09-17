# Transport/runtime review (2026-09-17)

The prior report's “UART RX is absent from the platform” conclusion is not
established. The deployed script `user_script/u2_set_partition.tcl` explicitly
assigns **both** `tx` (APCP4 pin 112) and `rx` (APCP4 pin 131), and `fe_run.tcl`
uses that partition file. The platform release has no corresponding RTL source
to inspect, so this alone does not establish actual bitstream wiring. ModelZoo's
`examples/buddy-qwen35-fpga/docs/validation/FPGA_FUNCTIONAL_ACCEPTANCE_20260915.md`
reports successful UTF-8 UART input on **FPGA0** with the same NR UART base and
divisor; its raw logs are not included in that repository.

`uv_shell_exec` CPU ticks measure the host control process, not RA execution
progress. They cannot distinguish a running model from a stalled FPGA program.

## Fixed defects

- RX state was in a NOLOAD section excluded from RA's BSS clear, but NH never
  initialized it. NH now initializes and publishes both input cursors.
- The NH-owned head and RA-owned tail shared a cache line; NH flush could
  overwrite RA updates. Each writer now owns a separate line. NH invalidates
  tail before reading it, and flushes payload before publishing head.
- RA called `cbo.inval` although this runtime's RA contract uses volatile DDR
  loads/stores and fences. `nr_getchar` now has no cache-management instructions.
- Debug code read RBR twice per received byte and sampled it while idle;
  reading RBR consumes input. One read now supplies both the byte and debug log.
- The TX console stopped permanently after 64 KiB. It is now a ring with
  separately owned producer/consumer cursors and producer backpressure. NH
  drains in bounded batches so RX and completion polling remain responsive.
- `nr_write` preserves byte length, including embedded NUL. Scoped
  `nr_heap_mark`/`nr_heap_reset` let the caller reclaim graph temporaries after
  persisting all live results. Heap exhaustion reports cursor/limit and fails.
- Multi-segment launch previously accepted any nonempty glob of readbacks;
  missing weight/tokenizer readbacks could be overlooked. The launcher now
  validates every plan/file/hash, rejects escaping/colliding names, uploads an
  explicit manifest and requires each declared readback to match the original.
- UART writes previously dropped partial writes and swallowed write errors.
  Unsent bytes now survive short writes/EAGAIN; missing scheduled bytes fail.
- TTY restore failure no longer skips exclusive-device release or result.json.
- `--interactive` adds raw stdin forwarding through the single existing UART
  worker. Atomic UUID requests persist across SSH acknowledgement loss; the
  worker writes each accepted request in server order and acknowledges complete
  OS writes. It never restarts a crashed worker to replay uncertain input.

## Hardware observations

| Probe | Run | Result and limit |
|---|---|---|
| UART RX | `run-dca587dbe36a45e1` | DDR readback matched. Host wrote all 16 bytes; FPGA5 printed ready but no received-byte line during 90 seconds. **Not received**, not proof of platform-wide RX absence. |
| TX ring | `run-33c1d41050c94a48` | 67,584 payload bytes matched exactly across the 64 KiB ring boundary; RA returned PASS. A real SSH outage reattached to the same worker without restarting it. |
| DDR addresses | `run-f58ee08065af4c65` | 19 separate 64-byte lines passed after all writes preceded all reads. Addresses: `0xb8000000` at 64 MiB increments through `0xfc000000`, plus `0xfffff000`. This is **1,216 sampled bytes**, not proof of all contiguous DDR capacity or reliability. |

Each subdirectory includes raw UART, worker/UVHS logs, result.json, a verification
summary and ELF audit. TX wrap ran before the final bounded-drain adjustment;
the DDR probe ran with that final runtime and completed normally. Live stdin
forwarding is validated with fake SSH and a real host PTY; board RX remains
unverified, so it is not an FPGA interactive inference result.

## Reproduction

```sh
make -C examples/FPGA-BOSCAME/common/nr/probes \
  build/uart_rx_probe.bin build/console_wrap_probe.bin build/ddr_probe.bin \
  RISCV_LD='/usr/bin/ld.lld-20 -m elf64lriscv'
python3 -m unittest discover -s examples/FPGA-BOSCAME/tools -p 'test_fpga_*.py'
```

The 30-test result is in `transport-tests.log`; it includes a lost SSH enqueue
acknowledgement, a retry of the same request UUID, partial UTF-8 writes, embedded
NUL bytes, session closure, output/readback validation and symlink protection.
Board commands are in the public `common/nr/probes/README.md`.

No platform source or external symlink target was modified. Remote generated
files are confined to `/home/hjuser/Desktop/fpga-tester-ISCAS`; the probes used
the public runner and released their UART/UVHS sessions normally. The staged
Git index was preserved.

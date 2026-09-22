# Incomplete profiling attempt — not acceptance evidence

Run `run-83f5ac200bbd4272`, identical profile image SHA256
`b207321b9a51ecb6a08c920665e95857bfa077f580c348ef25f6edbe6420ac26`.
Prefill and decode positions 16–18 completed with passing numerical comparisons.
After `[model] decode begin position=00000013`, UART remained at 23,988 bytes
for over ten minutes, versus about 68 seconds for the preceding decode steps.
Remote UART and local relay agreed; the worker and UVHS processes still existed,
and `/home` had 13 GB free. These facts do not establish the cause or CPU state.

The owning worker was explicitly sent SIGTERM after checking its command line
and working directory; it cleaned up and reported `INTERRUPTED`. No other task
was stopped. The same prepared image was then retried under a new run ID.
Do not present this partial run as a complete 16+8 profile.

The UART copy here normalizes CRLF to LF. Original bytes and worker/UVHS logs
remain in `examples/FPGA-BOSCAME/build/fpga-runs/run-83f5ac200bbd4272/`.

FPGA5 run `run-584aa2e8c308423f` executed the real C chat template, tokenizer
encoder and incremental decoder for eight literal-text fixtures. The verifier
compared computed token IDs and decoded bytes with the official host oracle;
all eight cases passed. NR completion and DDR readback also passed. The result
file's image SHA256 equals the saved build manifest's image SHA256.

This is an automatic board text-computation probe. Inputs were embedded literal
text and UART input bytes written were zero. It does not establish UART RX,
full model generation, or Stage E interactive acceptance.

Recheck from model/:

```bash
python -B tools/check_tokenizer_probe.py \
  --manifest validation/board/review/tokenizer-probe/probe.json \
  --uart-log validation/board/review/tokenizer-probe/uart.raw.log \
  --execution fpga --run-id run-584aa2e8c308423f \
  --output validation/board/review/tokenizer-probe/verification.json
```

# Bare-metal text path

`tokenizer_resource.c` provides immutable resource validation, incremental UTF-8
decoding and the optional-system/one-user chat template. `tokenizer_encode.c`
provides encoding without a filesystem, heap allocation or mutable global state.
The model launcher owns UART input, context limits and generation.

Encoding order is exact added-token matching on original UTF-8, then NFC,
Qwen's declared split regex and byte-level BPE on the ordinary spans. The packer
rejects unsupported tokenizer settings instead of dropping their semantics.
Malformed input UTF-8 is rejected. Output decoding replaces malformed UTF-8 as
the reference decoder does, buffering incomplete sequences across token calls.
A caller must finish/reset its decoder between conversations.

The tokenizer accepts at most 32768 input bytes. Each ordinary span must fit
8192 scalars including canonical decomposition. The caller separately supplies
the token capacity; any overrun returns `-1`, sets `written` to zero, and requires
discarding partial output. Embedded NUL is supported by the bounded API.
The model must separately reject prompts/generation exceeding its KV context.

Host GCC `-O2 -fstack-usage` measured 229888 bytes for the deepest encoder call
chain (`qwen_encode` + `encode_plain` + `qwen_bpe`), below the public NR runtime's
1 MiB RA stack. The generated Unicode tables occupy about 73848 bytes, in
addition to the approximately 4.98 MiB packed tokenizer resource. Target compiler
stack frames and the complete firmware memory budget still require image review.

## Reproducible host verification

Run from `model/`, with a Python environment containing `tokenizers==0.22.2` and
the model's compatible Transformers installation. These commands do not access a
board and their results are explicitly labelled **host only**.

```bash
python -B tools/check_tokenizer_encode.py \
  --assets assets/official --build build/text-encode-audit \
  --output validation/tokenizer-encode-audit.json
python -B tools/check_text.py \
  --assets assets/official --build build/text-audit \
  --output validation/text-check-audit.json
cc -std=c11 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/tokenizer_safety_check.c text/tokenizer_resource.c \
  text/tokenizer_encode.c text/unicode_tables.c \
  -o build/text-audit/safety-check
build/text-audit/safety-check build/text-audit/tokenizer.bin
```

The September 17 audit passed 5173 encoding and regex-boundary cases, 26464 NFC
cases (every decomposable scalar recognized by the reference, including Hangul),
144 chat-template-to-encoding cases, 18 encoding rejection cases, 1076 decoder
sequences, 1256 malformed UTF-8 decoding cases, 72 templates and 29 malformed
resource/ID checks. ASan/UBSan boundary checks passed. Saved fixture fields
`fpga_ids`/`fpga_prompt_ids` remain null because these are host results.

The older `validation/tokenizer-encode.json` and
`validation/stage-e-tokenizer-path.json` refer to historical host checks despite
using “on-board”/“Stage E complete” in their titles. They are not board evidence.
Use the new `*-audit.json` files for the current source and source hashes.
Full UART interaction and model inference must be validated separately on FPGA.

## Unicode table provenance

The regex and normalizer in one tokenizers package can use different Unicode
versions. In the tested engine, the regex recognizes Unicode 15 Kawi letters,
while NFD leaves U+11938 unchanged, unlike Python's Unicode 14 normalizer.
Python `isspace()` also includes controls that the tokenizer regex does not.
Consequently `gen_unicode_tables.py` queries the actual reference engines for
all 1112064 Unicode scalars, rather than mixing Python category/normalization
semantics with tokenizers semantics. Regenerate with:

```bash
python -B tools/gen_unicode_tables.py \
  --header text/unicode_tables.h --source text/unicode_tables.c \
  --report validation/unicode-tables.json
```

Normalization tables store full decomposition lengths, include algorithmic
Hangul handling, and use correct starter/blocking composition. The source tables
are committed resources; target firmware has no dependency on Python/tokenizers.

## Automatic board probe

`tests/tokenizer_probe.c` runs eight literal-text chat fixtures using the shared
NR runtime. It prints the IDs returned by the real encoder, incrementally
decodes those IDs, and compares both against host-derived oracle data. Tokenizer
resources are embedded read-only; it needs neither UART RX nor model weights.
This verifies the board text computation separately from the full interactive
model. Build locally from `model/`:

```bash
python -B tools/build_tokenizer_probe.py
```

The output is `build/tokenizer-probe/tokenizer_probe.bin`, with ELF, link map,
symbols, executable ISA audit, SHA256 manifest and a local `rebuild.sh` alongside.
When the board scheduler has made FPGA5 available, the authorized operator can
run and validate it (these commands are not part of the build):

```bash
../../fpga_run.sh build/tokenizer-probe/tokenizer_probe.bin --fpga=5 \
  --capture-seconds=120 --completion-marker='verify NR runtime: PASS' \
  > build/tokenizer-probe/uart.log
python -B tools/check_tokenizer_probe.py \
  --manifest build/tokenizer-probe/probe.json \
  --uart-log build/tokenizer-probe/uart.log --execution=fpga \
  --output validation/board/tokenizer-probe.json
```

The verifier requires exact computed IDs/decoded bytes for all eight cases and
NR completion. It rejects missing/duplicate records and wrong values even when
PASS labels are present. `--execution=host` is reserved for the host shim; such
reports explicitly set `board_tokenizer_verified` to false. The board probe's
build manifest itself remains `BUILT_AND_ISA_AUDITED_NOT_RUN`; only an actual
UART log plus its verification report establishes execution.

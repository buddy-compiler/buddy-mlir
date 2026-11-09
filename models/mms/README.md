# MMS — facebook/mms-tts-eng

Multilingual TTS model (`VitsModel`). This integration compiles the
**HiFi-GAN vocoder stage** of VITS as a `single_forward` model: it maps a
latent spectrogram `(1 x flow_size x max_seq_len)` to a raw waveform
`(1 x audio_buffer_size)`.

> **Scope note.** The full text-to-speech pipeline (text encoder + data-dependent
> duration alignment + flow decoder) cannot be AOT-traced into one static graph:
> `torch._dynamo` fragments on the token→frame duration expansion
> (`torch.repeat_interleave` over network-predicted durations).  The compiled
> kernel therefore exercises the deterministic vocoder stage (~14.3M of the
> model's ~36M parameters), which is a real, verifiable computation.  End-to-end
> synthesis needs a segmented pipeline (out of scope for `single_forward`).

## Build

```sh
cmake -B build \
  -DBUDDY_BUILD_MMS_MODEL=ON \
  -DBUDDY_MMS_MODEL_PATH=$HOME/.cache/huggingface/hub/models--facebook--mms-tts-eng/snapshots/<snapshot> \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mms_rax
```

## Run

```sh
# Deterministic (all-zero) latent:
build/bin/buddy-cli --model build/models/mms/mms.rax --prompt "" < /dev/null

# Custom latent spectrogram (raw f32 blob, 1 x flow_size x max_seq_len):
build/bin/buddy-cli --model build/models/mms/mms.rax --prompt latents.bin < /dev/null
```

The runner prints a JSON summary of the generated waveform (sample count,
min/max/mean/RMS and the first samples).

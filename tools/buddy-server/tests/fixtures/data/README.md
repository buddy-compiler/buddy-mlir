# Resident adapter metadata

`deepseek-r1-config.json` is a fixed copy of the Hugging Face
[DeepSeek-R1-Distill-Qwen-1.5B config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/e164738bb96d0c534f3634b1640208e07f5b1efc/config.json)
at revision `e164738bb96d0c534f3634b1640208e07f5b1efc`. It is used with the production
`models/deepseek_r1/specs/f32.json`, `gen_config.py`, and `gen_session.py` to
compile the real resident adapter and generated ModelSession offline. Generated
files stay in the build directory. The spec supplies parameter counts, so this
check does not need PyTorch, Transformers, model weights, or generated kernels.

The load tests use the production discovery descriptor and resident v2 loader.
They create each production resident model twice, verify its initial unloaded
state and empty capabilities, check that raw decoding rejects unloaded use,
and destroy it. They do not load weights or execute model kernels. The portable
Tenstorrent check explicitly compiles with `BUDDY_LLAMA31_TT_HAS_TTNN=0`; it does
not validate the TTNN SDK branch or inference on a device. See the
[build checks and validation boundaries](../../../README.md#build-checks).

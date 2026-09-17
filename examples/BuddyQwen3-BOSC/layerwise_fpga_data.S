    /*
     * Keep the large, immutable model payload in a dedicated PT_LOAD segment.
     * qwen3_layerwise_compact.ld places it immediately after code, static state
     * and the fixed arena, so objcopy can produce one compact flat image.
     */
    .section .model_blob,"a",@progbits
    .balign 64
    .globl _buddy_qwen3_prefill_ids
_buddy_qwen3_prefill_ids:
    .incbin QWEN3_PREFILL_IDS_FILE

    .balign 64
    .globl _buddy_qwen3_prompt_length
_buddy_qwen3_prompt_length:
    .incbin QWEN3_PROMPT_LENGTH_FILE

    .balign 64
    .globl _buddy_qwen3_prefill_rope_cos
_buddy_qwen3_prefill_rope_cos:
    .incbin QWEN3_PREFILL_ROPE_COS_FILE

    .balign 64
    .globl _buddy_qwen3_prefill_rope_sin
_buddy_qwen3_prefill_rope_sin:
    .incbin QWEN3_PREFILL_ROPE_SIN_FILE

    .balign 64
    .globl _buddy_qwen3_prefill_mask
_buddy_qwen3_prefill_mask:
    .incbin QWEN3_PREFILL_MASK_FILE

    .balign 64
    .globl _buddy_qwen3_decode_rope_cos
_buddy_qwen3_decode_rope_cos:
    .incbin QWEN3_DECODE_ROPE_COS_FILE

    .balign 64
    .globl _buddy_qwen3_decode_rope_sin
_buddy_qwen3_decode_rope_sin:
    .incbin QWEN3_DECODE_ROPE_SIN_FILE

    .balign 64
    .globl _buddy_qwen3_decode_mask
_buddy_qwen3_decode_mask:
    .incbin QWEN3_DECODE_MASK_FILE

    .balign 64
    .globl _buddy_qwen3_cache_write_mask
_buddy_qwen3_cache_write_mask:
    .incbin QWEN3_CACHE_WRITE_MASK_FILE

    /* Keep the frequently accessed runtime inputs below the large weights. */
    .balign 64
    .globl _buddy_qwen3_params_f32
_buddy_qwen3_params_f32:
    .incbin QWEN3_PARAMS_F32_FILE

    .balign 64
    .globl _buddy_qwen3_params_i8
_buddy_qwen3_params_i8:
    .incbin QWEN3_PARAMS_I8_FILE
